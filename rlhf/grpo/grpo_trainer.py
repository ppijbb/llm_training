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
        if state.global_step % self.log_every_n_steps == 0 and state.global_step > 0:
            self._log_generations(args, state, **kwargs)

    @torch.no_grad()
    def _log_generations(self, args: TrainingArguments, state, **kwargs):
        """실제 생성 로그 작성 (콘솔에 강제 출력)"""
        # model과 tokenizer 가져오기 (trainer에서 직접)
        model = self.trainer.model
        tokenizer = self.trainer.tokenizer

        if not model or not tokenizer:
            # Fallback to kwargs if needed
            model = kwargs.get('model')
            tokenizer = kwargs.get('tokenizer')

        if not model or not tokenizer:
            error_msg = f"⚠️ Step {state.global_step}: Model or tokenizer not available. Skipping generation logging."
            logger.warning(error_msg)
            print(f"\n{error_msg}")
            logger.debug(f"Available kwargs keys: {list(kwargs.keys())}")
            return

        self.eval_step_count += 1
        current_step = state.global_step
        
        logger.info(f"🔍 Step {current_step} - Logging generations (optimized batch mode)...")
        
        # 콘솔에 간소화된 출력
        print(f"\n{'='*80}")
        print(f"🔄 STEP {current_step} - Generation 테스트 시작 (배치 처리)")
        print(f"{'='*80}")

        # 모델 상태 저장 및 evaluation 모드로 전환
        was_training = model.training
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
            # 처리할 프롬프트 선택
            prompts_to_process = sample_prompts[:self.max_samples]
            
            # 배치 처리로 CPU 사용률 감소 및 속도 향상
            try:
                # 배치 토크나이징
                inputs = tokenizer(
                    text=prompts_to_process, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512  # 최대 길이 제한으로 메모리 절약
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # 배치 생성 (한 번에 여러 샘플 처리 - CPU 사용률 감소)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,  # 100 -> 50으로 줄여서 속도 향상
                        num_return_sequences=1,
                        do_sample=False,  # greedy decoding으로 속도 향상 및 CPU 부하 감소
                        pad_token_id=pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,  # 캐시 사용으로 속도 향상
                    )
                
                # 배치 디코딩 (CPU에서 한 번에 처리 - 효율적)
                generated_texts = tokenizer.batch_decode(
                    outputs, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # 각 샘플에 대해 로그 생성
                for i, (prompt, generated_text) in enumerate(zip(prompts_to_process, generated_texts)):
                    # 프롬프트 부분 제거
                    prompt_len = len(prompt)
                    generated_only = generated_text[prompt_len:].strip()[:200] if len(generated_text) > prompt_len else ""
                    
                    log_entry = {
                        "step": current_step,
                        "generation_step": self.eval_step_count,
                        "sample_index": i,
                        "prompt": prompt,
                        "generated": generated_only,
                        "full_response": generated_text[:300]
                    }
                    
                    generation_logs.append(log_entry)
                    
                    # 콘솔에 간소화된 출력
                    print(f"📝 Sample {i+1}/{len(prompts_to_process)}: {generated_only[:60]}..." if generated_only else f"📝 Sample {i+1}/{len(prompts_to_process)}: (empty)")
                    
                    # Logger에도 출력
                    logger.info(f"📝 Sample {i+1}: {generated_only[:100]}")
                
            except Exception as batch_error:
                # 배치 처리 실패 시 개별 처리로 fallback
                logger.warning(f"⚠️ Batch generation failed, falling back to individual: {batch_error}")
                
                for i, prompt in enumerate(prompts_to_process):
                    try:
                        inputs = tokenizer(
                            prompt, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=512
                        )
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}

                        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=50,
                                num_return_sequences=1,
                                do_sample=False,
                                pad_token_id=pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                use_cache=True,
                            )

                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        prompt_len = len(prompt)
                        generated_only = generated_text[prompt_len:].strip()[:200] if len(generated_text) > prompt_len else ""

                        log_entry = {
                            "step": current_step,
                            "generation_step": self.eval_step_count,
                            "sample_index": i,
                            "prompt": prompt,
                            "generated": generated_only,
                            "full_response": generated_text[:300]
                        }

                        generation_logs.append(log_entry)
                        print(f"📝 Sample {i+1}/{len(prompts_to_process)}: {generated_only[:60]}..." if generated_only else f"📝 Sample {i+1}/{len(prompts_to_process)}: (empty)")

                    except Exception as e:
                        error_msg = f"❌ Error generating for sample {i}: {e}"
                        logger.error(error_msg)
                        traceback.print_exc()
                        continue

        except Exception as e:
            error_msg = f"❌ Error during generation logging: {e}"
            logger.error(error_msg)
            print(f"\n{error_msg}")

        # 로그 파일에 저장
        if generation_logs:
            log_file = os.path.join(self.output_dir, f"generation_log_step_{current_step}.json")
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(generation_logs, f, ensure_ascii=False, indent=2)
                save_msg = f"💾 Generation logs saved to {log_file}"
                logger.info(save_msg)
                print(f"\n{save_msg}")
            except Exception as e:
                error_msg = f"❌ Failed to save generation logs: {e}"
                logger.error(error_msg)
                print(f"\n{error_msg}")
        else:
            print(f"\n⚠️ No generation logs to save")

        # 콘솔에 완료 메시지 출력
        print(f"\n{'='*80}")
        print(f"✅ STEP {current_step} - Generation 테스트 완료")
        print(f"{'='*80}\n")

        # 모델을 원래 모드로 복원
        if was_training:
            model.train()


class CustomGRPOTrainer(GRPOTrainer):
    """TRL GRPOTrainer를 상속받은 커스텀 트레이너"""

    def __init__(
        self,
        reward_functions: List[MultiRewardFunction|SingleCustomRewardFunction] = None,
        generation_log_dir: str = "./generation_logs",
        max_generation_samples: int = 5,
        generation_log_every_n_steps: int = 5,
        *args,
        **kwargs
    ):
        kwargs['args'].model_init_kwargs = kwargs["model_init_kwargs"] if "model_init_kwargs" in kwargs else {}
        self.custom_reward_functions = reward_functions or []

        # super().__init__() 먼저 호출 (Trainer 초기화)
        super().__init__(reward_funcs=self.custom_reward_functions, *args, **kwargs)
        
        # 생성 로깅 콜백 설정 (항상 등록, 무조건 실행)
        self.add_callback(
            GenerationLoggingCallback(
                trainer=self,
                output_dir=generation_log_dir,
                max_samples=max_generation_samples,
                log_every_n_steps=generation_log_every_n_steps
            ))
        total_callbacks = len(getattr(self.callback_handler, 'callbacks', [])) if hasattr(self, 'callback_handler') else 0
        logger.info(f"✅ Generation logging callback added to trainer (log_every_n_steps={generation_log_every_n_steps}, total_callbacks={total_callbacks})")
        print(f"✅ Generation logging callback 등록됨 (매 {generation_log_every_n_steps} step마다 실행)")

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

    def _prepare_inputs(self, inputs):
        """
        Override _prepare_inputs to handle evaluation correctly.
        During evaluation, skip the generation and scoring logic that expects
        num_generations completions per prompt.
        """
        # Check if we're in evaluation mode FIRST
        # During evaluation, the model should not be in training mode
        is_eval_mode = False
        if hasattr(self, 'model') and self.model is not None:
            is_eval_mode = not self.model.training
        elif hasattr(self, 'args') and hasattr(self.args, 'do_eval'):
            is_eval_mode = getattr(self.args, 'do_eval', False)
        
        # During training, ALWAYS use parent implementation (don't modify inputs)
        if not is_eval_mode:
            try:
                return super()._prepare_inputs(inputs)
            except RuntimeError as e:
                if "shape" in str(e) and ("num_generations" in str(e) or "invalid for input of size" in str(e)):
                    logger.error(f"❌ Shape mismatch error in _prepare_inputs during training: {e}")
                    raise
                raise
        
        # Only handle evaluation mode here
        # During evaluation, skip generation and return inputs as-is
        # This prevents the shape mismatch error in _generate_and_score_completions
        if isinstance(inputs, dict):
            # Move inputs to the correct device if model is available
            if hasattr(self, 'model') and self.model is not None:
                try:
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                except StopIteration:
                    # Model has no parameters, skip device movement
                    pass
            return inputs
        
        # If inputs is a list during evaluation, handle it
        if isinstance(inputs, list):
            # TRL may pass list of dicts during evaluation
            if len(inputs) > 0 and isinstance(inputs[0], dict):
                # Convert list of dicts to single dict (take first element)
                inputs = inputs[0]
                # Move to device
                if hasattr(self, 'model') and self.model is not None:
                    try:
                        device = next(self.model.parameters()).device
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in inputs.items()}
                    except StopIteration:
                        pass
                return inputs
            else:
                # Unknown list format, pass to parent
                return super()._prepare_inputs(inputs)
        
        # Fallback: return as-is
        return inputs

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Override prediction_step to handle evaluation correctly.
        During evaluation, we need to avoid the generation and scoring logic
        that expects num_generations completions per prompt.
        """
        # During training, ALWAYS use parent implementation
        if model.training:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )
        
        # Only handle evaluation mode here
        try:
            # Prepare inputs (this will handle list conversion if needed)
            inputs = self._prepare_inputs(inputs)
            
            # Ensure inputs is a dict after preparation
            if not isinstance(inputs, dict):
                logger.warning(f"⚠️ Inputs is not a dict after preparation: {type(inputs)}, passing to parent")
                return super().prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys
                )
            
            # For evaluation, compute loss directly if labels are present
            if 'labels' in inputs:
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs.loss if hasattr(outputs, 'loss') else None
                    logits = outputs.logits if hasattr(outputs, 'logits') else None
                    labels = inputs.get('labels')
                    
                    if prediction_loss_only:
                        return (loss, None, None)
                    return (loss, logits, labels)
            
            # Fallback to parent implementation
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )
        except (RuntimeError, TypeError) as e:
            error_str = str(e)
            if "shape" in error_str and ("num_generations" in error_str or "invalid for input of size" in error_str):
                # Handle the shape mismatch error during evaluation
                logger.warning(f"⚠️ Shape mismatch during evaluation, using simplified prediction step: {e}")
                
                # For evaluation, compute loss directly without generation
                if isinstance(inputs, dict) and 'labels' in inputs:
                    with torch.no_grad():
                        outputs = model(**inputs)
                        loss = outputs.loss if hasattr(outputs, 'loss') else None
                        logits = outputs.logits if hasattr(outputs, 'logits') else None
                        labels = inputs.get('labels')
                        
                        if prediction_loss_only:
                            return (loss, None, None)
                        return (loss, logits, labels)
                else:
                    # If no labels, return dummy values
                    logger.warning("⚠️ No labels found in inputs during evaluation, returning None")
                    return (None, None, None)
            elif "list indices must be integers" in error_str or "not str" in error_str or "string indices must be integers" in error_str:
                # Handle TypeError when inputs is list but accessed as dict
                logger.warning(f"⚠️ Inputs type mismatch during evaluation: {e}, attempting to fix")
                # Try to get original inputs and handle properly
                if isinstance(inputs, list) and len(inputs) > 0:
                    if isinstance(inputs[0], dict):
                        inputs = inputs[0]
                        # Retry with dict inputs
                        if 'labels' in inputs:
                            with torch.no_grad():
                                outputs = model(**inputs)
                                loss = outputs.loss if hasattr(outputs, 'loss') else None
                                logits = outputs.logits if hasattr(outputs, 'logits') else None
                                labels = inputs.get('labels')
                                
                                if prediction_loss_only:
                                    return (loss, None, None)
                                return (loss, logits, labels)
                # If we can't fix it, pass to parent
                return super().prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys
                )
            else:
                raise


class UnslothGRPOTrainWorkflow:
    """GRPO Trainer using TRL's GRPOTrainer with Unsloth optimizations"""

    def __init__(
        self,
        config: GRPOConfig,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        reward_functions: List[MultiRewardFunction|SingleCustomRewardFunction] = None,
        generation_log_dir: str = None,
        max_generation_samples: int = 5,
        generation_log_every_n_steps: int = 50
    ):
        self.config = config
        self.model_init_kwargs = model_init_kwargs or {}
        self.model = None
        self.tokenizer = None
        self.reward_functions = reward_functions or []
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
            # Create custom trainer with reward functions and generation logging (항상 활성화)
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

    def save_model(
        self,
        output_dir: Optional[str] = None
    ):
        """Save the trained model with memory optimization"""
        if output_dir is None:
            output_dir = self.config.output_dir

        logger.info(f"💾 Saving model to {output_dir}")
        
        # 메모리 정리 함수
        def clear_gpu_memory():
            """GPU 메모리 정리"""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
        
        # 저장 전 메모리 정리
        logger.info("🧹 Clearing GPU memory before saving...")
        clear_gpu_memory()
        
        # DeepSpeed를 사용하는 경우 DeepSpeed의 save_checkpoint 사용
        try:
            # trainer가 DeepSpeed를 사용하는지 확인
            if (self.trainer is not None and 
                hasattr(self.trainer, 'deepspeed') and 
                self.trainer.deepspeed is not None):
                logger.info("🔧 Using DeepSpeed checkpoint saving...")
                # DeepSpeed checkpoint 저장
                self.trainer.deepspeed.save_checkpoint(output_dir)
                logger.info(f"✅ DeepSpeed checkpoint saved to {output_dir}")
                
                # DeepSpeed checkpoint 후에도 모델 저장 (호환성을 위해)
                try:
                    # 모델을 CPU로 이동하여 저장 (메모리 절약)
                    original_device = next(self.model.parameters()).device
                    logger.info(f"📦 Moving model to CPU for saving (original device: {original_device})...")
                    
                    # 모델을 CPU로 임시 이동
                    self.model = self.model.cpu()
                    clear_gpu_memory()
                    
                    # 모델 저장
                    self.model.save_pretrained(output_dir)
                    logger.info(f"✅ Model saved to {output_dir}")
                    
                    # 모델을 원래 device로 복원
                    self.model = self.model.to(original_device)
                    logger.info(f"✅ Model restored to {original_device}")
                    
                except Exception as cpu_save_error:
                    logger.warning(f"⚠️ CPU save failed, trying GPU save: {cpu_save_error}")
                    # CPU 저장 실패 시 GPU에서 직접 저장
                    self.model.save_pretrained(output_dir)
                    logger.info(f"✅ Model saved to {output_dir} (GPU)")
            else:
                # DeepSpeed를 사용하지 않는 경우 일반 저장
                logger.info("🔧 Using standard model saving...")
                
                # 저장 전 gradient 정리
                if hasattr(self.model, 'zero_grad'):
                    self.model.zero_grad()
                
                # 모델을 CPU로 이동하여 저장 (메모리 절약)
                try:
                    # 모델 파라미터의 device 확인
                    model_params = list(self.model.parameters())
                    if model_params:
                        original_device = model_params[0].device
                        if original_device.type == 'cuda':
                            logger.info(f"📦 Moving model to CPU for saving (original device: {original_device})...")
                            self.model = self.model.cpu()
                            clear_gpu_memory()
                        
                        # 모델 저장
                        self.model.save_pretrained(output_dir)
                        logger.info(f"✅ Model saved to {output_dir}")
                        
                        # 모델을 원래 device로 복원
                        if original_device.type == 'cuda':
                            self.model = self.model.to(original_device)
                            logger.info(f"✅ Model restored to {original_device}")
                    else:
                        # 파라미터가 없는 경우 (PEFT 모델 등)
                        logger.info("⚠️ No model parameters found, saving directly...")
                        self.model.save_pretrained(output_dir)
                        logger.info(f"✅ Model saved to {output_dir}")
                except Exception as cpu_save_error:
                    logger.warning(f"⚠️ CPU save failed, trying GPU save: {cpu_save_error}")
                    # CPU 저장 실패 시 GPU에서 직접 저장
                    self.model.save_pretrained(output_dir)
                    logger.info(f"✅ Model saved to {output_dir} (GPU)")

        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback: 간단한 저장 시도
            try:
                logger.info("🔄 Attempting fallback save...")
                self.model.save_pretrained(output_dir)
                logger.info(f"✅ Fallback save successful to {output_dir}")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback save also failed: {fallback_error}")
                raise
        finally:
            # Tokenizer 저장
            try:
                self.tokenizer.save_pretrained(output_dir)
                logger.info(f"✅ Tokenizer saved to {output_dir}")
            except Exception as tokenizer_error:
                logger.error(f"❌ Failed to save tokenizer: {tokenizer_error}")
            
            # 저장 후 메모리 정리
            logger.info("🧹 Clearing GPU memory after saving...")
            clear_gpu_memory()


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
