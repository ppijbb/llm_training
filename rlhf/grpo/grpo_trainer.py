"""
Unsloth GRPO (Group Relative Policy Optimization) Trainer

This module provides GRPO training functionality using Unsloth's optimized training framework.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import json
import os
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np

# Import custom reward functions
from reward_functions import (
    BaseRewardFunction, 
    RewardFunctionFactory, 
    create_reward_function,
    REWARD_CONFIGS
)


logger = logging.getLogger(__name__)

@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    
    # Model configuration
    model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit"
    max_seq_length: int = 2048
    dtype: str = "float16"  # float16, bfloat16, float32
    load_in_4bit: bool = True
    
    # Training configuration
    learning_rate: float = 5e-7
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = -1
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    
    # GRPO specific configuration
    beta: float = 0.1  # KL divergence penalty
    gamma: float = 1.0  # Reward scaling factor
    group_size: int = 4  # Group size for relative ranking
    
    # Reward function configuration
    reward_function_type: str = "systematic"  # systematic, group_relative, multi_objective
    reward_config_name: str = "default"  # default, balanced, aggressive, etc.
    custom_reward_config: Optional[Dict[str, Any]] = None
    
    
    # Output configuration
    output_dir: str = "./grpo_outputs"
    logging_dir: str = "./grpo_logs"
    report_to: str = "wandb"  # wandb, tensorboard, none
    
    # Data configuration
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    max_samples: int = 1000
    test_size: float = 0.1
    
    # Other configuration
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }


class GRPOTrainer:
    """GRPO Trainer using Unsloth's optimized framework"""
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.reward_function = None
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize reward function
        self._initialize_reward_function()
        
        logger.info(f"✅ GRPO Trainer initialized with config: {config.model_name}")
        logger.info(f"🎯 Reward function: {self.reward_function.name if self.reward_function else 'None'}")
    
    
    def _initialize_reward_function(self):
        """Initialize the reward function based on configuration"""
        try:
            if self.config.custom_reward_config:
                # Use custom configuration
                self.reward_function = RewardFunctionFactory.create_reward_function(
                    self.config.reward_function_type,
                    self.config.custom_reward_config
                )
            else:
                # Use predefined configuration
                self.reward_function = create_reward_function(
                    self.config.reward_function_type,
                    self.config.reward_config_name
                )
            
            logger.info(f"✅ Reward function initialized: {self.reward_function.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize reward function: {e}")
            raise
    
    def load_model(self):
        """Load model and tokenizer using Unsloth"""
        logger.info(f"🔄 Loading model: {self.config.model_name}")
        
        try:
            # Load model with Unsloth optimizations
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=self.config.dtype,
                load_in_4bit=self.config.load_in_4bit,
            )
            
            # Set chat template
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=self.config.seed,
                use_rslora=False,
                loftq_config=None,
            )
            
            # Set chat template
            FastLanguageModel.for_training(self.model)
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def prepare_dataset(self, dataset):
        """Prepare dataset for GRPO training"""
        logger.info("🔄 Preparing dataset for GRPO training")
        
        def format_grpo_data(example):
            """Format data for GRPO training"""
            prompt = example.get("prompt", "")
            chosen = example.get("chosen", "")
            rejected = example.get("rejected", "")
            
            # Create chosen and rejected conversations
            chosen_conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ]
            
            rejected_conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}
            ]
            
            # Tokenize conversations
            chosen_tokens = self.tokenizer.apply_chat_template(
                chosen_conversation,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            
            rejected_tokens = self.tokenizer.apply_chat_template(
                rejected_conversation,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            
            return {
                "chosen_input_ids": chosen_tokens.squeeze().tolist(),
                "rejected_input_ids": rejected_tokens.squeeze().tolist(),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }
        
        # Format dataset
        formatted_dataset = dataset.map(
            format_grpo_data,
            remove_columns=dataset.column_names,
            desc="Formatting GRPO data"
        )
        
        logger.info(f"✅ Dataset prepared: {len(formatted_dataset)} samples")
        return formatted_dataset
    
    def create_training_arguments(self):
        """Create training arguments"""
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
            seed=self.config.seed,
        )
    
    def create_grpo_collator(self):
        """Create custom data collator for GRPO"""
        from transformers import DataCollatorWithPadding
        
        class GRPODataCollator(DataCollatorWithPadding):
            def __init__(self, tokenizer, padding=True, max_length=None):
                super().__init__(tokenizer, padding=padding, max_length=max_length)
                self.tokenizer = tokenizer
            
            def __call__(self, features):
                # Separate chosen and rejected data
                chosen_features = []
                rejected_features = []
                
                for feature in features:
                    chosen_features.append({
                        "input_ids": feature["chosen_input_ids"],
                        "attention_mask": [1] * len(feature["chosen_input_ids"])
                    })
                    rejected_features.append({
                        "input_ids": feature["rejected_input_ids"],
                        "attention_mask": [1] * len(feature["rejected_input_ids"])
                    })
                
                # Collate chosen and rejected separately
                chosen_batch = super().__call__(chosen_features)
                rejected_batch = super().__call__(rejected_features)
                
                return {
                    "chosen_input_ids": chosen_batch["input_ids"],
                    "chosen_attention_mask": chosen_batch["attention_mask"],
                    "rejected_input_ids": rejected_batch["input_ids"],
                    "rejected_attention_mask": rejected_batch["attention_mask"],
                    "prompts": [f["prompt"] for f in features],
                    "chosen_responses": [f["chosen"] for f in features],
                    "rejected_responses": [f["rejected"] for f in features]
                }
        
        return GRPODataCollator(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.config.max_seq_length
        )
    
    def create_grpo_trainer(self, train_dataset, eval_dataset=None):
        """Create GRPO trainer"""
        logger.info("🔄 Creating GRPO trainer")
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Create data collator
        data_collator = self.create_grpo_collator()
        
        # Create custom trainer for GRPO
        class GRPOTrainerClass(Trainer):
            def __init__(self, grpo_config, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.grpo_config = grpo_config
            
            def compute_loss(self, model, inputs, return_outputs=False):
                """Compute GRPO loss with custom reward function"""
                chosen_input_ids = inputs["chosen_input_ids"]
                chosen_attention_mask = inputs["chosen_attention_mask"]
                rejected_input_ids = inputs["rejected_input_ids"]
                rejected_attention_mask = inputs["rejected_attention_mask"]
                
                # Get model outputs for chosen and rejected
                chosen_outputs = model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                    labels=chosen_input_ids
                )
                
                rejected_outputs = model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask,
                    labels=rejected_input_ids
                )
                
                # Compute rewards using custom reward function
                if self.grpo_config.reward_function:
                    rewards = self.grpo_config.reward_function.compute_reward(
                        chosen_outputs.logits,
                        rejected_outputs.logits,
                        chosen_attention_mask,
                        rejected_attention_mask,
                        **inputs
                    )
                else:
                    raise ValueError("Reward function not initialized")
                
                # Compute GRPO loss with rewards
                chosen_loss = chosen_outputs.loss
                rejected_loss = rejected_outputs.loss
                
                # Apply reward scaling
                reward_scaled_loss = chosen_loss - rejected_loss + self.grpo_config.gamma * rewards.mean()
                
                # Add KL divergence penalty
                kl_penalty = self.grpo_config.beta * self._compute_kl_divergence(
                    chosen_outputs.logits, rejected_outputs.logits,
                    chosen_attention_mask, rejected_attention_mask
                )
                
                grpo_loss = reward_scaled_loss + kl_penalty
                
                if return_outputs:
                    return grpo_loss, (chosen_outputs, rejected_outputs)
                
                return grpo_loss
            
            def _compute_kl_divergence(self, chosen_logits, rejected_logits, chosen_mask, rejected_mask):
                """Compute KL divergence between chosen and rejected distributions"""
                chosen_probs = torch.softmax(chosen_logits, dim=-1)
                rejected_probs = torch.softmax(rejected_logits, dim=-1)
                
                # Compute KL divergence
                kl_div = torch.sum(
                    chosen_probs * torch.log(chosen_probs / (rejected_probs + 1e-8)),
                    dim=-1
                )
                
                # Average over valid tokens
                chosen_kl = torch.mean(kl_div[chosen_mask.bool()])
                rejected_kl = torch.mean(kl_div[rejected_mask.bool()])
                
                return chosen_kl - rejected_kl
        
        # Create trainer
        self.trainer = GRPOTrainerClass(
            grpo_config=self.config,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("✅ GRPO trainer created successfully")
        return self.trainer
    
    def train(self, train_dataset, eval_dataset=None):
        """Start GRPO training"""
        logger.info("🚀 Starting GRPO training")
        
        try:
            # Prepare datasets
            train_dataset = self.prepare_dataset(train_dataset)
            if eval_dataset:
                eval_dataset = self.prepare_dataset(eval_dataset)
            
            # Create trainer
            self.create_grpo_trainer(train_dataset, eval_dataset)
            
            # Initialize wandb if specified
            if self.config.report_to == "wandb":
                wandb.init(
                    project="grpo-training",
                    config=self.config.to_dict(),
                    name=f"grpo-{self.config.model_name.split('/')[-1]}"
                )
            
            # Start training
            training_result = self.trainer.train()
            
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
            
            # Save training config
            config_path = os.path.join(output_dir, "grpo_config.json")
            with open(config_path, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            logger.info(f"✅ Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def evaluate(self, eval_dataset):
        """Evaluate the model"""
        logger.info("📊 Evaluating model")
        
        try:
            # Prepare eval dataset
            eval_dataset = self.prepare_dataset(eval_dataset)
            
            # Create trainer if not exists
            if self.trainer is None:
                self.create_grpo_trainer(train_dataset=eval_dataset, eval_dataset=eval_dataset)
            
            # Run evaluation
            eval_result = self.trainer.evaluate()
            
            logger.info(f"✅ Evaluation completed: {eval_result}")
            return eval_result
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
    


def create_grpo_trainer(config: GRPOConfig) -> GRPOTrainer:
    """Create GRPO trainer with given configuration"""
    return GRPOTrainer(config)


