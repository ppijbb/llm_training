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
import traceback
# Import TRL components
from trl import GRPOTrainer, GRPOConfig

# Import Unsloth for model loading
from unsloth import FastLanguageModel

# Import transformers for callbacks
from transformers import TrainerCallback, TrainingArguments

# Import custom reward functions for TRL compatibility
from reward.reward_functions import MultiRewardFunction, SingleCustomRewardFunction
from reward.cmd_reward_functions import CommandRewardFunction, ComponentRewardWrapper

logger = logging.getLogger(__name__)


class GenerationLoggingCallback(TrainerCallback):
    """특정 step마다 생성된 텍스트를 콘솔과 파일에 출력하는 콜백 (항상 활성화)"""

    def __init__(
        self, 
        trainer, 
        output_dir: str = "./generation_logs", 
        max_samples: int = 5, 
        log_every_n_steps: int = 3
    ):
        self.trainer = trainer
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.log_every_n_steps = log_every_n_steps
        self.eval_step_count = 0

        # 로그 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"📊 Generation logging callback initialized. Output dir: {output_dir}, Log every {log_every_n_steps} steps")
        print(f"\n{'='*80}")
        print(f"📊 Generation Logging Callback 초기화됨")
        print(f"   출력 디렉토리: {output_dir}")
        print(f"   로깅 주기: 매 {log_every_n_steps} step마다")
        print(f"   최대 샘플 수: {max_samples}")
        print(f"{'='*80}\n")

    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):
        """Step 종료 시 호출 (특정 step마다 반드시 실행)"""
        # 로깅 주기에 맞춰 생성 로그 출력 (무조건 실행)
        if state.global_step > 0 and state.global_step % self.log_every_n_steps == 0:
            self._log_generations(args, state, **kwargs)

    @torch.no_grad()
    def _log_generations(self, args: TrainingArguments, state, **kwargs):
        """실제 생성 로그 작성 (콘솔에 강제 출력)"""
        model = self.trainer.model
        tokenizer = self.trainer.tokenizer

        if not model or not tokenizer:
            logger.warning(f"⚠️ Step {state.global_step}: Model or tokenizer not available. Skipping generation logging.")
            return

        self.eval_step_count += 1
        current_step = state.global_step
        
        print(f"\n{'='*80}")
        print(f"🔄 STEP {current_step} - Generation 테스트 시작 (배치 처리)")
        print(f"{'='*80}")

        was_training = model.training
        model.eval()

        generation_logs = []
        sample_prompts = [
            "Start with 3, 5 4 6, mesial bleeding, middle of suppuration, mobility 2, 3 3 3, 3 2 3, repeat 8, 3 4 4, 3 5 4 furcation grade 2 ",
            "3 2 3, 3 2 3, 3 2 3, repeat, repeat, repeat, bleeding 1 on mesial, bleeding 2 mesial and distal three bleeding all",
            "number 1, 16 impacted, 17 and 32 missing, 5 4 3, 3 3 4, 3 2 3, repeat 9, 4 5 5, bleeding all, mobility class 3",
            "probing 3 3 3, 3 4 3, 4 3 3, 3 2 3, 2 2 3, repeat on 7, 3 2 3, 4 3 4, 3 3 4, 3 2 3, 3, distal suppuration and bleeding",
            "Mark number 18, furcation 2, 19, suppuration distal with proximal bleeding, 20, 4 3 3",
        ]

        try:
            prompts_to_process = sample_prompts[:self.max_samples]
            inputs = tokenizer(
                text=prompts_to_process, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            
            # Decode only the generated part
            input_ids_len = inputs['input_ids'].shape[1]
            generated_ids = outputs[:, input_ids_len:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            for i, (prompt, generated_only) in enumerate(zip(prompts_to_process, generated_texts)):
                log_entry = {
                    "step": current_step,
                    "generation_step": self.eval_step_count,
                    "sample_index": i,
                    "prompt": prompt,
                    "generated": generated_only.strip(),
                }
                generation_logs.append(log_entry)
                print(f"📝 Sample {i+1}/{len(prompts_to_process)}: {generated_only.strip()[:60]}...")
                
        except Exception as e:
            logger.error(f"❌ Error during generation logging: {e}", exc_info=True)

        if generation_logs:
            log_file = os.path.join(self.output_dir, f"generation_log_step_{current_step}.json")
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(generation_logs, f, ensure_ascii=False, indent=2)
                print(f"\n💾 Generation logs saved to {log_file}")
            except Exception as e:
                logger.error(f"❌ Failed to save generation logs: {e}")

        print(f"\n{'='*80}")
        print(f"✅ STEP {current_step} - Generation 테스트 완료")
        print(f"{'='*80}\n")

        if was_training:
            model.train()


class CustomGRPOTrainer(GRPOTrainer):
    """TRL GRPOTrainer를 상속받은 커스텀 트레이너"""

    def __init__(
        self,
        reward_functions: List[MultiRewardFunction|SingleCustomRewardFunction|ComponentRewardWrapper] = None,
        generation_log_dir: str = "./generation_logs",
        max_generation_samples: int = 5,
        generation_log_every_n_steps: int = 5,
        *args,
        **kwargs
    ):
        # model_init_kwargs is not expected by the parent class
        if "model_init_kwargs" in kwargs:
             kwargs.pop("model_init_kwargs",None)
        
        self.custom_reward_functions = reward_functions or []

        super().__init__(reward_funcs=self.custom_reward_functions, *args, **kwargs)
        
        self.add_callback(
            GenerationLoggingCallback(
                trainer=self,
                output_dir=generation_log_dir,
                max_samples=max_generation_samples,
                log_every_n_steps=generation_log_every_n_steps
            ))
        logger.info(f"✅ Generation logging callback added (every {generation_log_every_n_steps} steps)")

    def compute_rewards(
        self,
        completions,
        **kwargs
    ):
        if not self.custom_reward_functions:
            return super().compute_rewards(completions, **kwargs)

        all_rewards = [reward_func(completions, **kwargs) for reward_func in self.custom_reward_functions]

        if all_rewards:
            final_rewards = [sum(rewards) / len(all_rewards) for rewards in zip(*all_rewards)]
            return final_rewards

        return super().compute_rewards(completions, **kwargs)

    def _prepare_inputs(self, inputs):
        if not self.model.training:
            # During evaluation, just move inputs to device
            try:
                device = next(self.model.parameters()).device
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            except StopIteration: # No parameters
                return inputs
        return super()._prepare_inputs(inputs)
        
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        if model.training:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        inputs = self._prepare_inputs(inputs)
        if 'labels' not in inputs:
            return None, None, None

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.get("loss")
            logits = outputs.get("logits")
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, inputs.get('labels'))


class UnslothGRPOTrainWorkflow:
    """GRPO Trainer using TRL's GRPOTrainer with Unsloth optimizations"""

    def __init__(
        self,
        config: GRPOConfig,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        reward_functions: List[MultiRewardFunction|SingleCustomRewardFunction|ComponentRewardWrapper] = None,
        generation_log_dir: str = None,
        max_generation_samples: int = 5,
        generation_log_every_n_steps: int = 10
    ):
        self.config = config
        self.model_init_kwargs = model_init_kwargs or {}
        
        if reward_functions is None:
            logger.info("No reward functions provided, using CommandRewardFunction by default.")
            cmd_reward_func = CommandRewardFunction()
            self.reward_functions = cmd_reward_func.expand_to_individual_rewards()
        else:
            self.reward_functions = reward_functions
            
        self.generation_log_dir = generation_log_dir or os.path.join(config.output_dir, "generation_logs")
        self.max_generation_samples = max_generation_samples
        self.generation_log_every_n_steps = generation_log_every_n_steps
        self.trainer = None
        self._load_model()
        logger.info("✅ Unsloth GRPO Trainer initialized successfully")
    
    def _load_model(self):
        """Load model and tokenizer using Unsloth"""
        model_name = self.model_init_kwargs.get("model_name", "unsloth/Qwen3-0.6B-bnb-4bit")
        logger.info(f"🔄 Loading model: {model_name}")
        
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.config.max_prompt_length,
                dtype=None,
                load_in_4bit=True,
                device_map="auto",
            )
            
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=self.config.seed,
                use_rslora=False,
                loftq_config={},
            )
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}", exc_info=True)
            raise
    
    def create_grpo_trainer(
        self,
        train_dataset,
        eval_dataset=None
    ):
        """Create TRL GRPOTrainer"""
        logger.info("🔄 Creating TRL GRPOTrainer")
        try:
            self.trainer = CustomGRPOTrainer(
                reward_functions=self.reward_functions,
                generation_log_dir=self.generation_log_dir,
                max_generation_samples=self.max_generation_samples,
                generation_log_every_n_steps=self.generation_log_every_n_steps,
                model=self.model,
                args=self.config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
            )
            logger.info("✅ TRL GRPOTrainer created successfully")
            return self.trainer
        except Exception as e:
            logger.error(f"❌ Failed to create GRPOTrainer: {e}", exc_info=True)
            raise
    
    def train(self, train_dataset, eval_dataset=None):
        logger.info("🚀 Starting GRPO training with TRL")
        try:
            trainer = self.create_grpo_trainer(train_dataset, eval_dataset)
            training_result = trainer.train()
            logger.info("✅ GRPO training completed successfully")
            return training_result
        except Exception as e:
            logger.error(f"❌ GRPO training failed: {e}", exc_info=True)
            raise

    def save_model(self, output_dir: Optional[str] = None):
        output_dir = output_dir or self.config.output_dir
        logger.info(f"💾 Saving model to {output_dir}")
        try:
            if self.trainer and hasattr(self.trainer, 'model'):
                self.trainer.save_model(output_dir)
            else:
                self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"✅ Model and tokenizer saved successfully to {output_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}", exc_info=True)
            raise

    def evaluate(self, eval_dataset=None):
        logger.info("📊 Starting evaluation")
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call create_grpo_trainer() first.")
        try:
            eval_results = self.trainer.evaluate(eval_dataset)
            logger.info(f"📊 Evaluation completed: {eval_results}")
            return eval_results
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}", exc_info=True)
            raise

def create_grpo_trainer(
    config: GRPOConfig,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    reward_functions: Optional[List] = None,
    generation_log_dir: str = None,
    max_generation_samples: int = 5,
    generation_log_every_n_steps: int = 50
) -> UnslothGRPOTrainWorkflow:
    """Create GRPO trainer with given configuration, reward functions, and generation logging (항상 활성화)"""
    return UnslothGRPOTrainWorkflow(
        config=config,
        model_init_kwargs=model_init_kwargs,
        reward_functions=reward_functions,
        generation_log_dir=generation_log_dir,
        max_generation_samples=max_generation_samples,
        generation_log_every_n_steps=generation_log_every_n_steps)
