"""
Configuration management for GRPO training using TRL's GRPOConfig
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import GRPOConfig from TRL
from trl import GRPOConfig


def create_grpo_config(model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit",
                      output_dir: str = "./grpo_outputs",
                      **kwargs) -> GRPOConfig:
    """Create GRPOConfig using TRL's standard parameters"""
    logger.info(f"🔧 Creating GRPOConfig for model: {model_name}")

    # TRL GRPOConfig에서 지원하는 설정들만 사용
    default_config = {
        "model_init_kwargs": {"model_name": model_name},
        "output_dir": output_dir,
        "learning_rate": 5e-7,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "max_steps": -1,
        "warmup_steps": 5,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "beta": 0.1,
        "num_generations": 8,
        "max_prompt_length": 2048,
        "max_completion_length": 2048,
        "report_to": "wandb",
        "seed": 42,
        "use_vllm": True,
        "vllm_mode": "chocolate",
        "vllm_enable_sleep_mode": True
    }

    # 사용자 정의 설정으로 업데이트
    default_config.update(kwargs)

    config = GRPOConfig(**default_config)
    logger.info("✅ GRPOConfig created successfully")
    return config


def create_quick_test_config() -> GRPOConfig:
    """Create configuration for quick testing"""
    return create_grpo_config(
        model_name="unsloth/Qwen3-0.6B-bnb-4bit",
        output_dir="./test_grpo_outputs",
        num_train_epochs=1,
        max_steps=10,
        logging_steps=1,
        save_steps=5,
        eval_steps=5,
    )


def create_production_config(model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit") -> GRPOConfig:
    """Create configuration for production training"""
    return create_grpo_config(
        model_name=model_name,
        output_dir=f"./grpo_outputs_{model_name}",
        num_train_epochs=3,
        max_steps=-1,
        logging_steps=50,
        save_steps=1000,
        eval_steps=1000,
    )


