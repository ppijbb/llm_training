"""
Unsloth GRPO (Group Relative Policy Optimization) Trainer

This module provides GRPO training functionality using TRL's GRPOTrainer and Unsloth optimizations.
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional

# Import TRL components
from trl import GRPOTrainer, GRPOConfig

# Import Unsloth for model loading
from unsloth import FastLanguageModel

# Import custom reward functions for TRL compatibility
from reward_functions import MultiRewardFunction, SingleCustomRewardFunction

logger = logging.getLogger(__name__)


class CustomGRPOTrainer(GRPOTrainer):
    """TRL GRPOTrainer를 상속받은 커스텀 트레이너"""

    def __init__(
        self,
        reward_functions: List[MultiRewardFunction|SingleCustomRewardFunction] = None,
        *args,
        **kwargs
    ):
        kwargs['args'].model_init_kwargs = kwargs["model_init_kwargs"] if "model_init_kwargs" in kwargs else {}
        self.custom_reward_functions = reward_functions or []
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


class UnslothGRPOTrainer:
    """GRPO Trainer using TRL's GRPOTrainer with Unsloth optimizations"""
    
    def __init__(
        self,
        config: GRPOConfig,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        reward_functions: List[MultiRewardFunction|SingleCustomRewardFunction] = None
    ):
        self.config = config
        self.model_init_kwargs = model_init_kwargs or {}
        self.model = None
        self.tokenizer = None
        self.reward_functions = reward_functions or []
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
                dtype="float16",
                load_in_4bit=True,
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
                loftq_config=None,
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
            # Create custom trainer with reward functions
            self.trainer = CustomGRPOTrainer(
                reward_functions=self.reward_functions,
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
        """Save the trained model"""
        if output_dir is None:
            output_dir = self.config.output_dir
            
        logger.info(f"💾 Saving model to {output_dir}")
        
        try:
            # Save model and tokenizer
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"✅ Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise


def create_grpo_trainer(
    config: GRPOConfig,
    model_init_kwargs: Optional[Dict[str, Any]] = None,
    reward_functions: Optional[List] = None
) -> UnslothGRPOTrainer:
    """Create GRPO trainer with given configuration and reward functions"""
    return UnslothGRPOTrainer(
        config=config,
        model_init_kwargs=model_init_kwargs,
        reward_functions=reward_functions)
