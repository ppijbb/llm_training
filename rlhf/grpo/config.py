"""
Configuration management for GRPO training
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Import GRPOConfig from TRL
from trl import GRPOConfig

# Default configurations for different model sizes - mapped to TRL GRPOConfig parameters
DEFAULT_CONFIGS = {
    "Qwen3-0.6B": {
        "model_init_kwargs": {"model_name": "unsloth/Qwen3-0.6B-bnb-4bit"},
        "max_prompt_length": 2048,
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
        "num_generations": 8,
        "max_completion_length": 256,
        "loss_type": "grpo",
    },

    "llama-3.2-1B": {
        "model_init_kwargs": {"model_name": "unsloth/llama-3.2-1B-bnb-4bit"},
        "max_prompt_length": 2048,
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
        "num_generations": 8,
        "max_completion_length": 256,
        "loss_type": "grpo",
    },

    "gemma-3n-E2B": {
        "model_init_kwargs": {"model_name": "unsloth/gemma-3n-E2B-bnb-4bit"},
        "max_prompt_length": 2048,
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
        "num_generations": 8,
        "max_completion_length": 256,
        "loss_type": "grpo",
    },

    "gpt-oss": {
        "model_init_kwargs": {"model_name": "unsloth/gpt-oss-20b-bnb-4bit"},
        "max_prompt_length": 2048,
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
        "num_generations": 8,
        "max_completion_length": 256,
        "loss_type": "grpo",
    }
}

# Default reward configurations
DEFAULT_REWARD_CONFIGS = {
    "Qwen3-0.6B": {
        "reward_function_type": "systematic",
        "reward_config_name": "default",
    },

    "llama-3.2-1B": {
        "reward_function_type": "systematic",
        "reward_config_name": "default",
    },

    "gemma-3n-E2B": {
        "reward_function_type": "systematic",
        "reward_config_name": "default",
    },

    "gpt-oss": {
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
    ) -> tuple[GRPOConfig, Dict[str, Any], Dict[str, Any]]:
        """Create configuration with defaults"""
        logger.info(f"🔧 Creating config for {model_name} with {dataset_name} dataset")
        
        # Get model config
        if model_name in DEFAULT_CONFIGS:
            model_config = DEFAULT_CONFIGS[model_name].copy()
        else:
            logger.warning(f"⚠️ Unknown model {model_name}, using default config")
            model_config = DEFAULT_CONFIGS["Qwen3-0.6B"].copy()
            # Update model name in model_init_kwargs
            if "model_init_kwargs" in model_config:
                model_config["model_init_kwargs"] = {"model_name": model_name}
        
        # Get dataset config (but don't include in GRPOConfig)
        dataset_config = {}
        if dataset_name in DEFAULT_DATASETS:
            dataset_config = DEFAULT_DATASETS[dataset_name].copy()
        else:
            logger.warning(f"⚠️ Unknown dataset {dataset_name}, using default config")
            dataset_config = DEFAULT_DATASETS["ultrafeedback"].copy()
            dataset_config["dataset_name"] = dataset_name

        # Merge configs (exclude dataset-specific fields from GRPOConfig)
        config_dict = model_config.copy()
        dataset_excluded_fields = {"dataset_name", "split", "max_samples", "test_size"}
        config_dict.update({k: v for k, v in dataset_config.items() if k not in dataset_excluded_fields})
        
        # Apply custom config if provided
        if custom_config:
            config_dict.update(custom_config)
        
        # Create GRPOConfig
        self.config = GRPOConfig(**config_dict)

        # Get reward configuration
        reward_config = self.get_reward_config(model_name)

        logger.info("✅ Config created successfully")
        return self.config, reward_config, dataset_config
    
    def get_available_models(self) -> list:
        """Get list of available model configurations"""
        return list(DEFAULT_CONFIGS.keys())
    
    def get_available_datasets(self) -> list:
        """Get list of available dataset configurations"""
        return list(DEFAULT_DATASETS.keys())

    def get_reward_config(self, model_name: str) -> Dict[str, Any]:
        """Get reward configuration for a model"""
        if model_name in DEFAULT_REWARD_CONFIGS:
            return DEFAULT_REWARD_CONFIGS[model_name].copy()
        else:
            return DEFAULT_REWARD_CONFIGS["Qwen3-0.6B"].copy()
    
    def validate_config(self, config: GRPOConfig) -> bool:
        """Validate configuration"""
        logger.info("🔍 Validating configuration")

        try:
            # Check required fields for TRL GRPOConfig
            required_fields = [
                "learning_rate", "per_device_train_batch_size", "num_train_epochs"
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

            # Check TRL GRPOConfig specific fields
            if hasattr(config, 'max_prompt_length') and config.max_prompt_length <= 0:
                logger.error("❌ Max prompt length must be positive")
                return False

            if hasattr(config, 'num_generations') and config.num_generations <= 0:
                logger.error("❌ Number of generations must be positive")
                return False

            if hasattr(config, 'beta') and config.beta < 0:
                logger.error("❌ Beta (KL penalty) must be non-negative")
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
    config, reward_config, dataset_config = create_default_config(
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
    return config, reward_config, dataset_config


def get_production_config(model_name: str = "Qwen3-0.6B"):
    """Get configuration for production training"""
    config, reward_config, dataset_config = create_default_config(
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
    return config, reward_config, dataset_config


