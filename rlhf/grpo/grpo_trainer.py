"""
Unsloth GRPO (Group Relative Policy Optimization) Trainer

This module provides GRPO training functionality using TRL's GRPOTrainer and Unsloth optimizations.
"""

import logging
import torch
import numpy as np
import json
import os
from typing import Dict, Any, List, Optional

# Import TRL components
from trl import GRPOTrainer, GRPOConfig

# Import Unsloth for model loading
from unsloth import FastLanguageModel

# Import transformers for callbacks
from transformers import TrainerCallback, TrainingArguments

# Import custom reward functions for TRL compatibility
from reward.reward_functions import MultiRewardFunction, SingleCustomRewardFunction

logger = logging.getLogger(__name__)


class GenerationLoggingCallback(TrainerCallback):
    """특정 step마다 생성된 텍스트를 로그로 출력하는 콜백"""

    def __init__(self, output_dir: str = "./generation_logs", max_samples: int = 5, log_every_n_steps: int = 50):
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.log_every_n_steps = log_every_n_steps
        self.eval_step_count = 0

        # 로그 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"📊 Generation logging callback initialized. Output dir: {output_dir}, Log every {log_every_n_steps} steps")

    def on_log(self, args: TrainingArguments, state, control, **kwargs):
        """로그 발생 시 호출 (특정 step마다)"""
        # 로깅 주기에 맞춰 생성 로그 출력
        if state.global_step % self.log_every_n_steps == 0 and state.global_step > 0:
            self._log_generations(args, state, **kwargs)

    def _log_generations(self, args: TrainingArguments, state, **kwargs):
        """실제 생성 로그 작성"""
        model = kwargs.get('model')
        tokenizer = kwargs.get('tokenizer')

        if not model or not tokenizer:
            return

        self.eval_step_count += 1
        current_step = state.global_step
        
        logger.info(f"🔍 Step {current_step} - Logging generations...")

        # 모델을 evaluation 모드로 전환
        model.eval()

        generation_logs = []

        # 샘플 프롬프트 정의
        sample_prompts = [
            "Start with 3, 5 4 6, mesial bleeding, middle of suppuration, mobility 2, 3 3 3, 3 2 3, repeat 8, 3 4 4, 3 5 4 furcation grade 2 ",
            "3 2 3, 3 2 3, 3 2 3, repeat, repeat, repeat, bleeding 1 on mesial, bleeding 2 mesial and distal three bleeding all",
            "number 1, 16 impacted, 17 and 32 missing, 5 4 3, 3 3 4, 3 2 3, repeat 9, 4 5 5, bleeding all, mobility class 3",
            "probing 3 3 3, 3 4 3, 4 3 3, 3 2 3, 2 2 3, repeat on 7, 3 2 3, 4 3 4, 3 3 4, 3 2 3, 3, distal suppuration and bleeding",
            "Mark number 18, furcation 2, 19, suppuration distal with proximal bleeding, 20, 4 3 3",
            "3 3 3, 3 4 3, 4 3 3, 3 2 3, 2 2 3, repeat on 7 quadrant 3 bleeding (bleeding all)",
            "3 3 3, 3 4 3, 4 3 3, 3 2 3, 2 2 3, repeat on 7 3 3 2, 3 3 3, 4 3 4, 4 3 5 quadrant 4 bleeding on proximal",
            "number 31, 6 5 5, 5 4 5, number 19, 5 5 4 other teeth all 3 3 3",
            "All probing 3 3 3, except number 21 and 25. 21 is 6 5 5. 25 is 5 5 4.",
            "13 suppuration and bleeding on the distopalatal, jump to 30 furaction 1",
            "number 11 crown 3 2 3, implant 3 3 4, bleeding",
        ]

        try:
            for i, prompt in enumerate(sample_prompts[:self.max_samples]):
                try:
                    # 생성 파라미터 설정
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    # 생성 실행
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            num_return_sequences=1,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                    # 생성된 텍스트 디코딩
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # 로그 데이터 구성
                    log_entry = {
                        "step": current_step,
                        "generation_step": self.eval_step_count,
                        "sample_index": i,
                        "prompt": prompt,
                        "generated": generated_text[len(prompt):].strip()[:200],
                        "full_response": generated_text[:300]
                    }

                    generation_logs.append(log_entry)

                    logger.info(f"📝 Sample {i+1}:")
                    logger.info(f"   Prompt: {prompt}")
                    logger.info(f"   Generated: {log_entry['generated']}")

                except Exception as e:
                    logger.error(f"❌ Error generating for sample {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"❌ Error during generation logging: {e}")

        # 로그 파일에 저장
        if generation_logs:
            log_file = os.path.join(self.output_dir, f"generation_log_step_{current_step}.json")
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(generation_logs, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Generation logs saved to {log_file}")
            except Exception as e:
                logger.error(f"❌ Failed to save generation logs: {e}")

        # 모델을 다시 training 모드로 전환
        model.train()


class CustomGRPOTrainer(GRPOTrainer):
    """TRL GRPOTrainer를 상속받은 커스텀 트레이너"""

    def __init__(
        self,
        reward_functions: List[MultiRewardFunction|SingleCustomRewardFunction] = None,
        enable_generation_logging: bool = True,
        generation_log_dir: str = "./generation_logs",
        max_generation_samples: int = 5,
        generation_log_every_n_steps: int = 50,
        *args,
        **kwargs
    ):
        kwargs['args'].model_init_kwargs = kwargs["model_init_kwargs"] if "model_init_kwargs" in kwargs else {}
        self.custom_reward_functions = reward_functions or []
        self.enable_generation_logging = enable_generation_logging

        # 생성 로깅 콜백 설정
        if self.enable_generation_logging:
            self.generation_callback = GenerationLoggingCallback(
                output_dir=generation_log_dir,
                max_samples=max_generation_samples,
                log_every_n_steps=generation_log_every_n_steps
            )
            # 콜백을 args에 추가 (Trainer가 콜백을 인식하도록)
            if 'callbacks' not in kwargs:
                kwargs['callbacks'] = []
            kwargs['callbacks'].append(self.generation_callback)
            logger.info("✅ Generation logging callback added to trainer")

        super().__init__(reward_funcs=self.custom_reward_functions, *args, **kwargs)

    def compute_rewards(
        self,
        completions,
        **kwargs
    ):
        """커스텀 보상 함수를 사용한 보상 계산"""
        if not self.custom_reward_functions:
            # 기본 TRL 보상 함수 사용
            return super().compute_rewards(completions, **kwargs)

        # 커스텀 보상 함수들 실행
        all_rewards = []
        for reward_func in self.custom_reward_functions:
            rewards = reward_func(completions, **kwargs)
            all_rewards.append(rewards)

        # 보상 평균 계산
        if all_rewards:
            final_rewards = []
            for i in range(len(completions)):
                avg_reward = sum(rewards[i] for rewards in all_rewards) / len(all_rewards)
                final_rewards.append(avg_reward)
            return final_rewards

        return super().compute_rewards(completions, **kwargs)


class UnslothGRPOTrainWorkflow:
    """GRPO Trainer using TRL's GRPOTrainer with Unsloth optimizations"""

    def __init__(
        self,
        config: GRPOConfig,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        reward_functions: List[MultiRewardFunction|SingleCustomRewardFunction] = None,
        enable_generation_logging: bool = True,
        generation_log_dir: str = None,
        max_generation_samples: int = 5,
        generation_log_every_n_steps: int = 50
    ):
        self.config = config
        self.model_init_kwargs = model_init_kwargs or {}
        self.model = None
        self.tokenizer = None
        self.reward_functions = reward_functions or []
        self.enable_generation_logging = enable_generation_logging
        self.generation_log_dir = generation_log_dir or os.path.join(config.output_dir, "generation_logs")
        self.max_generation_samples = max_generation_samples
        self.generation_log_every_n_steps = generation_log_every_n_steps
        self.trainer = None

        # Initialize components
        self._load_model()

        logger.info("✅ Unsloth GRPO Trainer initialized successfully")
    
    def _load_model(self):
        """Load model and tokenizer using Unsloth"""
        model_name = self.model_init_kwargs.get("model_name", "unsloth/Qwen3-0.6B-bnb-4bit")
        logger.info(f"🔄 Loading model: {model_name}")
        
        try:
            # Load model with Unsloth optimizations
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=getattr(self.config, 'max_prompt_length', 2048),
                dtype=None,
                load_in_4bit=True,
                device_map="balanced",
                use_gradient_checkpointing="unsloth",
            )
            
            # Apply PEFT
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                    ],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=getattr(self.config, 'seed', 42),
                use_rslora=False,
                qat_scheme = "int4",
                loftq_config={},
            )
            
            FastLanguageModel.for_training(self.model)
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def create_grpo_trainer(
        self,
        train_dataset,
        eval_dataset=None
    ):
        """Create TRL GRPOTrainer"""
        logger.info("🔄 Creating TRL GRPOTrainer")
        
        try:
            # Create custom trainer with reward functions and generation logging
            self.trainer = CustomGRPOTrainer(
                reward_functions=self.reward_functions,
                enable_generation_logging=self.enable_generation_logging,
                generation_log_dir=self.generation_log_dir,
                max_generation_samples=self.max_generation_samples,
                generation_log_every_n_steps=self.generation_log_every_n_steps,
                model=self.model,
                args=self.config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                model_init_kwargs=self.model_init_kwargs,
            )
            
            logger.info("✅ TRL GRPOTrainer created successfully")
            return self.trainer
            
        except Exception as e:
            logger.error(f"❌ Failed to create GRPOTrainer: {e}")
            raise
    
    def train(
        self,
        train_dataset,
        eval_dataset=None
    ):
        """Start GRPO training using TRL"""
        logger.info("🚀 Starting GRPO training with TRL")
        
        try:
            # Create trainer
            trainer = self.create_grpo_trainer(train_dataset, eval_dataset)
            
            # Start training
            training_result = trainer.train()
            
            logger.info("✅ GRPO training completed successfully")
            return training_result
            
        except Exception as e:
            logger.error(f"❌ GRPO training failed: {e}")
            raise

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset=None
    ):
        """Evaluate the model on the given dataset"""
        logger.info("📊 Starting evaluation")

        if self.trainer is None:
            logger.error("❌ Trainer not initialized. Call create_grpo_trainer() first.")
            raise RuntimeError("Trainer not initialized")

        try:
            # Use provided output_dir or default from config
            if output_dir is None:
                output_dir = self.config.output_dir

            # Run evaluation using TRL trainer
            eval_results = self.trainer.evaluate(eval_dataset)

            logger.info(f"📊 Evaluation completed: {eval_results}")
            return eval_results

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise

    def save_model(
        self,
        output_dir: Optional[str] = None
    ):
        """Save the trained model"""
        if output_dir is None:
            output_dir = self.config.output_dir

        logger.info(f"💾 Saving model to {output_dir}")

        try:
            # Save model and tokenizer
            self.model.save_pretrained(output_dir)
            logger.info(f"✅ Model saved to {output_dir}")

        except Exception as e:
            self.model.save_pretrained(output_dir)
            logger.error(f"❌ Failed to save model: {e}")
            print(f"Svae model")
        finally:
            self.tokenizer.save_pretrained(output_dir)


    def evaluate(
        self,
        eval_dataset=None,
        output_dir: Optional[str] = None
    ):
        """Evaluate the model on the given dataset"""
        logger.info("📊 Starting evaluation")

        if self.trainer is None:
            logger.error("❌ Trainer not initialized. Call create_grpo_trainer() first.")
            raise RuntimeError("Trainer not initialized")

        try:
            # Use provided output_dir or default from config
            if output_dir is None:
                output_dir = self.config.output_dir

            # Run evaluation using TRL trainer
            eval_results = self.trainer.evaluate(eval_dataset)

            logger.info(f"📊 Evaluation completed: {eval_results}")
            return eval_results

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise


def create_grpo_trainer(
    config: GRPOConfig,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    reward_functions: Optional[List] = None,
    enable_generation_logging: bool = True,
    generation_log_dir: str = None,
    max_generation_samples: int = 5,
    generation_log_every_n_steps: int = 50
) -> UnslothGRPOTrainWorkflow:
    """Create GRPO trainer with given configuration, reward functions, and generation logging options"""
    return UnslothGRPOTrainWorkflow(
        config=config,
        model_init_kwargs=model_init_kwargs,
        reward_functions=reward_functions,
        enable_generation_logging=enable_generation_logging,
        generation_log_dir=generation_log_dir,
        max_generation_samples=max_generation_samples,
        generation_log_every_n_steps=generation_log_every_n_steps)
