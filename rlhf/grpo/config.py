"""
Configuration management for GRPO training
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Import GRPOConfig from grpo_trainer to avoid circular imports
try:
    from grpo_trainer import GRPOConfig
except ImportError:
    # Fallback definition if import fails
    from dataclasses import dataclass
    
    @dataclass
    class GRPOConfig:
        model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit"
        max_seq_length: int = 2048
        dtype: str = "float16"
        load_in_4bit: bool = True
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
        beta: float = 0.1
        gamma: float = 1.0
        group_size: int = 4
        output_dir: str = "./grpo_outputs"
        logging_dir: str = "./grpo_logs"
        report_to: str = "wandb"
        dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
        max_samples: int = 1000
        test_size: float = 0.1
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
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Default configurations for different model sizes
DEFAULT_CONFIGS = {
    "Qwen3-0.6B": {
        "model_name": "unsloth/Qwen3-0.6B-bnb-4bit",
        "max_seq_length": 2048,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-7,
        "num_train_epochs": 3,
        "max_steps": -1,
        "warmup_steps": 5,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "beta": 0.1,
        "gamma": 1.0,
        "group_size": 4,
    },
    
    "llama-3.2-1B": {
        "model_name": "unsloth/llama-3.2-1B-bnb-4bit",
        "max_seq_length": 2048,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-7,
        "num_train_epochs": 2,
        "max_steps": -1,
        "warmup_steps": 10,
        "logging_steps": 20,
        "save_steps": 1000,
        "eval_steps": 1000,
        "beta": 0.1,
        "gamma": 1.0,
        "group_size": 4,
    },
    
    "gemma-3n-E2B": {
        "model_name": "unsloth/gemma-3n-E2B-bnb-4bit",
        "max_seq_length": 2048,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-7,
        "num_train_epochs": 3,
        "max_steps": -1,
        "warmup_steps": 5,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "beta": 0.1,
        "gamma": 1.0,
        "group_size": 4,
    },
    
    "gpt-oss": {
        "model_name": "unsloth/gpt-oss-20b-bnb-4bit",
        "max_seq_length": 2048,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-7,
        "num_train_epochs": 3,
        "max_steps": -1,
        "warmup_steps": 5,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "beta": 0.1,
        "gamma": 1.0,
        "group_size": 4,
        "reward_function_type": "systematic",
        "reward_config_name": "default",
    }
}

# Default dataset configurations
DEFAULT_DATASETS = {
    "ultrafeedback": {
        "dataset_name": "HuggingFaceH4/ultrafeedback_binarized",
        "split": "train_prefs",
        "max_samples": 1000,
        "test_size": 0.1,
    },
    
    "ultrafeedback_large": {
        "dataset_name": "HuggingFaceH4/ultrafeedback_binarized",
        "split": "train_prefs",
        "max_samples": 10000,
        "test_size": 0.05,
    },
    
    "hh_rlhf": {
        "dataset_name": "Anthropic/hh-rlhf",
        "split": "train",
        "max_samples": 1000,
        "test_size": 0.1,
    },
    
    "openai_summarize": {
        "dataset_name": "openai/summarize_from_feedback",
        "split": "train",
        "max_samples": 1000,
        "test_size": 0.1,
    }
}


class ConfigManager:
    """Configuration manager for GRPO training"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
    
    def load_config(self, config_path: str) -> GRPOConfig:
        """Load configuration from file"""
        logger.info(f"📁 Loading config from {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            self.config = GRPOConfig(**config_dict)
            logger.info("✅ Config loaded successfully")
            return self.config
            
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise
    
    def save_config(self, config: GRPOConfig, output_path: str):
        """Save configuration to file"""
        logger.info(f"💾 Saving config to {output_path}")
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            logger.info("✅ Config saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save config: {e}")
            raise
    
    def create_config(
        self, 
        model_name: str = "Qwen3-0.6B",
        dataset_name: str = "ultrafeedback",
        custom_config: Optional[Dict[str, Any]] = None
    ) -> GRPOConfig:
        """Create configuration with defaults"""
        logger.info(f"🔧 Creating config for {model_name} with {dataset_name} dataset")
        
        # Get model config
        if model_name in DEFAULT_CONFIGS:
            model_config = DEFAULT_CONFIGS[model_name].copy()
        else:
            logger.warning(f"⚠️ Unknown model {model_name}, using default config")
            model_config = DEFAULT_CONFIGS["Qwen3-0.6B"].copy()
            model_config["model_name"] = model_name
        
        # Get dataset config
        if dataset_name in DEFAULT_DATASETS:
            dataset_config = DEFAULT_DATASETS[dataset_name].copy()
        else:
            logger.warning(f"⚠️ Unknown dataset {dataset_name}, using default config")
            dataset_config = DEFAULT_DATASETS["ultrafeedback"].copy()
            dataset_config["dataset_name"] = dataset_name
        
        # Merge configs
        config_dict = {**model_config, **dataset_config}
        
        # Apply custom config if provided
        if custom_config:
            config_dict.update(custom_config)
        
        # Create GRPOConfig
        self.config = GRPOConfig(**config_dict)
        
        logger.info("✅ Config created successfully")
        return self.config
    
    def get_available_models(self) -> list:
        """Get list of available model configurations"""
        return list(DEFAULT_CONFIGS.keys())
    
    def get_available_datasets(self) -> list:
        """Get list of available dataset configurations"""
        return list(DEFAULT_DATASETS.keys())
    
    def validate_config(self, config: GRPOConfig) -> bool:
        """Validate configuration"""
        logger.info("🔍 Validating configuration")
        
        try:
            # Check required fields
            required_fields = [
                "model_name", "max_seq_length", "learning_rate",
                "per_device_train_batch_size", "num_train_epochs"
            ]
            
            for field in required_fields:
                if not hasattr(config, field) or getattr(config, field) is None:
                    logger.error(f"❌ Missing required field: {field}")
                    return False
            
            # Check value ranges
            if config.learning_rate <= 0:
                logger.error("❌ Learning rate must be positive")
                return False
            
            if config.per_device_train_batch_size <= 0:
                logger.error("❌ Batch size must be positive")
                return False
            
            if config.num_train_epochs <= 0:
                logger.error("❌ Number of epochs must be positive")
                return False
            
            if config.max_seq_length <= 0:
                logger.error("❌ Max sequence length must be positive")
                return False
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False


def create_default_config(
    model_name: str = "Qwen3-0.6B",
    dataset_name: str = "ultrafeedback",
    output_dir: str = "./grpo_outputs",
    custom_config: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs
):
    """Create default configuration"""
    manager = ConfigManager()
    return manager.create_config(model_name, dataset_name, custom_config)


def load_config_from_file(config_path: str):
    """Load configuration from file"""
    manager = ConfigManager()
    return manager.load_config(config_path)


def save_config_to_file(config, output_path: str):
    """Save configuration to file"""
    manager = ConfigManager()
    manager.save_config(config, output_path)


# Predefined configuration templates
def get_quick_test_config():
    """Get configuration for quick testing"""
    return create_default_config(
        model_name="Qwen3-0.6B",
        dataset_name="ultrafeedback",
        custom_config={
            "max_samples": 100,
            "num_train_epochs": 1,
            "max_steps": 10,
            "logging_steps": 1,
            "save_steps": 5,
            "eval_steps": 5,
            "output_dir": "./test_grpo_outputs"
        }
    )


def get_production_config(model_name: str = "Qwen3-0.6B"):
    """Get configuration for production training"""
    return create_default_config(
        model_name=model_name,
        dataset_name="ultrafeedback_large",
        custom_config={
            "max_samples": 10000,
            "num_train_epochs": 3,
            "max_steps": -1,
            "logging_steps": 50,
            "save_steps": 1000,
            "eval_steps": 1000,
            "output_dir": f"./grpo_outputs_{model_name}",
            "report_to": "wandb"
        }
    )


