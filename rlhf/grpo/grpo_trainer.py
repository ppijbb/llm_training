"""
Unsloth GRPO (Group Relative Policy Optimization) Trainer

This module provides GRPO training functionality using TRL's GRPOTrainer and Unsloth optimizations.
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable

# Import TRL components
from trl import GRPOTrainer, GRPOConfig

# Import Unsloth for model loading
from unsloth import FastLanguageModel

# Import custom reward functions
from reward_functions import (
    BaseRewardFunction,
    RewardFunctionFactory,
    create_reward_function,
    REWARD_CONFIGS
)

logger = logging.getLogger(__name__)


class UnslothGRPOTrainer:
    """GRPO Trainer using TRL's GRPOTrainer with Unsloth optimizations"""
    
    def __init__(self, config: GRPOConfig, model_init_kwargs: Optional[Dict[str, Any]] = None):
        self.config = config
        self.model_init_kwargs = model_init_kwargs or {}
        self.model = None
        self.tokenizer = None
        self.reward_functions = []
        self.trainer = None
        
        # Initialize components
        self._initialize_reward_functions()
        self._load_model()
        
        logger.info("✅ Unsloth GRPO Trainer initialized successfully")
    
    def _initialize_reward_functions(self):
        """Initialize reward functions for TRL GRPOTrainer"""
        try:
            # Use default systematic reward function
            reward_function = create_reward_function("systematic", "default")

            # Convert to TRL-compatible reward function format
            trl_reward_func = self._convert_to_trl_reward_function(reward_function)
            self.reward_functions = [trl_reward_func]

            logger.info(f"✅ Reward functions initialized: {[f.__name__ for f in self.reward_functions]}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize reward functions: {e}")
            raise
    
    def _convert_to_trl_reward_function(self, reward_function: BaseRewardFunction) -> Callable:
        """Convert custom reward function to TRL-compatible format"""
        def trl_reward_function(completions, **kwargs):
            """TRL-compatible reward function format"""
            # Extract completions and other necessary data
            chosen_completions = kwargs.get("chosen", [])
            rejected_completions = kwargs.get("rejected", [])
            
            # Compute rewards using the original reward function
            rewards = reward_function.compute_batch_rewards(
                chosen_completions=chosen_completions,
                rejected_completions=rejected_completions,
                **kwargs
            )
            
            return rewards
        
        return trl_reward_function
    
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
    
    def create_grpo_trainer(self, train_dataset, eval_dataset=None):
        """Create TRL GRPOTrainer"""
        logger.info("🔄 Creating TRL GRPOTrainer")
        
        try:
            # Create trainer
            self.trainer = GRPOTrainer(
                model=self.model,
                reward_funcs=self.reward_functions,
                args=self._create_training_arguments(),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
            )
            
            logger.info("✅ TRL GRPOTrainer created successfully")
            return self.trainer
            
        except Exception as e:
            logger.error(f"❌ Failed to create GRPOTrainer: {e}")
            raise
    
    def _create_training_arguments(self):
        """Create TrainingArguments for TRL GRPOTrainer"""
        from transformers import TrainingArguments
        
        return TrainingArguments(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            output_dir=self.config.output_dir,
            logging_dir=self.config.logging_dir,
            report_to=self.config.report_to,
            remove_unused_columns=self.config.remove_unused_columns,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            max_grad_norm=self.config.max_grad_norm,
            dataloader_num_workers=self.config.dataloader_num_workers,
            num_train_epochs=self.config.num_train_epochs,
            seed=getattr(self.config, 'seed', 42),
        )
    
    def train(self, train_dataset, eval_dataset=None):
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
    
    def save_model(self, output_dir: Optional[str] = None):
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


def create_grpo_trainer(config: GRPOConfig, model_init_kwargs: Optional[Dict[str, Any]] = None) -> UnslothGRPOTrainer:
    """Create GRPO trainer with given configuration"""
    return UnslothGRPOTrainer(config, model_init_kwargs)
