#!/usr/bin/env python3
import unsloth
import weave
from transformers.utils import is_flash_attn_2_available
import inspect

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
"""
Main script for GRPO training with Unsloth

Usage:
    python train_grpo.py --model llama-3.1-8b --dataset ultrafeedback
    python train_grpo.py --config config.json
    python train_grpo.py --custom-data /path/to/data.jsonl
"""
import argparse
import logging
import os
import sys
import json
from typing import List
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from grpo_trainer import create_grpo_trainer
from transformers import TrainingArguments
from trl import GRPOConfig
from data_loader import GRPODataLoader, create_grpo_dataloader
from config import (
    create_grpo_config,
    create_quick_test_config,
    create_production_config,
)
from reward_functions import (
    create_reward_function,
    create_multi_reward_function,
    create_single_reward_function,
    MultiRewardFunction,
    SingleCustomRewardFunction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('grpo_training.log')
    ]
)
logger = logging.getLogger(__name__)


def validate_grpo_config(config) -> bool:
    """Validate GRPOConfig"""
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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="GRPO Training with Unsloth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default settings
  python train_grpo.py --quick-test
  
  # Train with specific model and dataset
  python train_grpo.py --model llama-3.1-8b --dataset ultrafeedback
  
  # Train with custom configuration file
  python train_grpo.py --config my_config.json
  
  # Train with custom data
  python train_grpo.py --custom-data /path/to/data.jsonl --model llama-3.1-8b
  
  # Production training
  python train_grpo.py --production --model llama-3.1-70b
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama-3.1-8b",
        choices=["llama-3.1-8b", "llama-3.1-70b", "gemma-2-9b", "qwen2.5-7b"],
        help="Model to train (default: llama-3.1-8b)"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="ultrafeedback",
        choices=["ultrafeedback", "ultrafeedback_large", "hh_rlhf", "openai_summarize"],
        help="Dataset to use (default: ultrafeedback)"
    )
    
    # Custom data
    parser.add_argument(
        "--custom-data",
        type=str,
        help="Path to custom dataset file (JSON, JSONL, CSV)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/trainer_config.json",
        help="Path to configuration file"
    )
    
    # Training modes
    parser.add_argument(
        "--quick-test",
        action="store_true",
        default=False,
        help="Run quick test with minimal data"
    )
    
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run production training with full dataset"
    )
    
    # Training parameters
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to use"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for trained model"
    )
    
    # Reward function configuration
    parser.add_argument(
        "--reward-function",
        type=str,
        nargs="+",
        default=["single"],
        choices=["single", "multi"],
        help="Reward function types to use: 'single' for unified, 'multi' for multi-component (default: single)"
    )

    parser.add_argument(
        "--reward-config",
        type=str,
        help="Path to reward function configuration file (JSON)"
    )
    
    
    # Other options
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="grpo-training",
        help="Weights & Biases project name"
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires trained model)"
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume training from checkpoint"
    )

    # Generation logging options
    parser.add_argument(
        "--enable-generation-logging",
        action="store_true",
        default=True,
        help="Enable generation logging during evaluation (default: enabled)"
    )

    parser.add_argument(
        "--disable-generation-logging",
        action="store_true",
        help="Disable generation logging during evaluation"
    )

    parser.add_argument(
        "--generation-log-dir",
        type=str,
        default=None,
        help="Directory to save generation logs (default: {output_dir}/generation_logs)"
    )

    parser.add_argument(
        "--max-generation-samples",
        type=int,
        default=5,
        help="Maximum number of samples to generate for logging (default: 5)"
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> GRPOConfig:
    """Create configuration from command line arguments"""
    logger.info("🔧 Creating configuration from arguments")
    
    # Start with base configuration
    if args.config:
        # Load config from file
        logger.info(f"📁 Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
            model_name = config["model_init_kwargs"]["model_name"]
        config = create_grpo_config(model_name=model_name, **config)
        # TODO: 실제 파일에서 설정 로드 구현 필요
    elif args.quick_test:
        config = create_quick_test_config()
        logger.info("⚡ Using quick test configuration")
    elif args.production:
        config = create_production_config(args.model)
        logger.info("🏭 Using production configuration")
    else:
        # 기본 설정 생성
        config = create_grpo_config(
            model_name=f"unsloth/{args.model}-bnb-4bit" if not args.model.startswith("unsloth/") else args.model,
            output_dir="./grpo_outputs"
        )
        logger.info(f"🔧 Using default configuration for {args.model}")
    
    # Override with command line arguments
    custom_config = {}
    
    if args.max_samples:
        custom_config["max_samples"] = args.max_samples
        logger.info(f"📊 Max samples: {args.max_samples}")
    
    if args.epochs:
        custom_config["num_train_epochs"] = args.epochs
        logger.info(f"🔄 Epochs: {args.epochs}")
    
    if args.learning_rate:
        custom_config["learning_rate"] = args.learning_rate
        logger.info(f"📈 Learning rate: {args.learning_rate}")
    
    if args.batch_size:
        custom_config["per_device_train_batch_size"] = args.batch_size
        custom_config["per_device_eval_batch_size"] = args.batch_size
        logger.info(f"📦 Batch size: {args.batch_size}")
    
    if args.output_dir:
        custom_config["output_dir"] = args.output_dir
        logger.info(f"📁 Output directory: {args.output_dir}")
    
    if args.no_wandb:
        custom_config["report_to"] = "none"
        logger.info("🚫 Weights & Biases disabled")
    else:
        custom_config["report_to"] = "wandb"
        logger.info(f"📊 Weights & Biases project: {args.wandb_project}")

    # Note: Reward function configuration is handled by the trainer
    # TRL GRPOTrainer uses built-in reward functions
    logger.info(f"🎯 Using default reward function (handled by TRL GRPOTrainer)")


    # Apply custom configuration
    if custom_config:
        for key, value in custom_config.items():
            setattr(config, key, value)

    return config


def get_generation_logging_settings(args) -> tuple[bool, str, int]:
    """Generation logging 설정을 반환"""
    # enable/disable 플래그 처리
    enable_logging = args.enable_generation_logging and not args.disable_generation_logging

    # 로그 디렉토리 설정
    log_dir = args.generation_log_dir
    if not log_dir and hasattr(args, 'output_dir') and args.output_dir:
        log_dir = os.path.join(args.output_dir, "generation_logs")

    # 최대 샘플 수
    max_samples = args.max_generation_samples

    return enable_logging, log_dir, max_samples


def load_dataset(args, config: GRPOConfig):
    """TRL 표준 데이터셋 로딩"""
    logger.info("📦 Loading dataset with TRL standard")

    # 데이터 로더 생성 (간소화된 버전)
    model_name = config.model_init_kwargs.get("model_name", "unsloth/Qwen3-0.6B-bnb-4bit")
    max_length = getattr(config, 'max_prompt_length', 2048)

    # TRL 표준 데이터 로딩
    if args.custom_data:
        logger.info(f"📁 Loading custom data from {args.custom_data}")
        data_loader = GRPODataLoader(model_name, max_length)
        custom_split = getattr(args, 'split', 'train')  # 기본값 설정
        dataset = data_loader.load_custom_dataset(args.custom_data, split=custom_split)
    else:
        dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
        max_samples = getattr(args, 'max_samples', 1000)
        split = getattr(args, 'split', 'train_prefs')  # 기본값 설정
        logger.info(f"📦 Loading dataset: {dataset_name} (split: {split})")
        data_loader = GRPODataLoader(model_name, max_length)
        dataset = data_loader.load_dataset(dataset_name, split=split, max_samples=max_samples)

    # TRL 표준 형식으로 변환
    dataset = data_loader.prepare_grpo_data(dataset)
    splited = dataset.train_test_split(test_size=0.1)
    train_dataset = splited["train"]
    eval_dataset = splited["test"].shuffle(seed=42).select(range(min(100, len(splited["test"]))))

    logger.info(f"✅ Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
    
    return train_dataset, eval_dataset


def create_reward_functions(args) -> List:
    """통합 보상 함수 생성"""
    logger.info("🎯 Creating unified reward function")

    # 설정 파일에서 보상 함수 설정 로드
    config = {}
    if args.reward_config and os.path.exists(args.reward_config):
        try:
            with open(args.reward_config, 'r') as f:
                config = json.load(f)
            logger.info(f"📁 Loaded reward config from {args.reward_config}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load reward config: {e}")

    # 보상 함수 타입에 따라 생성
    reward_functions = []
    for reward_type in args.reward_function:
        if reward_type == "single":
            reward_func = create_single_reward_function(config)
            logger.info("✅ Created single unified reward function")
        elif reward_type == "multi":
            reward_func = create_multi_reward_function(config)
            logger.info("✅ Created multi-component reward function")
        else:
            logger.warning(f"Unknown reward type: {reward_type}, using single")
            reward_func = create_single_reward_function(config)

        reward_functions.append(reward_func)

    logger.info(f"🎯 Total reward functions: {len(reward_functions)}")
    return reward_functions


def main():
    """Main training function"""
    logger.info("🚀 Starting GRPO Training")
    
    try:
        # Parse arguments
        args = parse_arguments()
        logger.info(f"📋 Arguments: {vars(args)}")
        
        # Create configuration
        config = create_config_from_args(args)
        logger.info(f"⚙️ Configuration: {config.model_init_kwargs.get('model_name')}")
        
        # Validate configuration
        if not validate_grpo_config(config):
            logger.error("❌ Configuration validation failed")
            return 1
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"📁 Output directory: {config.output_dir}")
        
        # Load dataset
        train_dataset, eval_dataset = load_dataset(args, config)

        if len(train_dataset) == 0:
            logger.error("❌ No training data found")
            return 1

        # Create reward functions
        reward_functions = create_reward_functions(args)

        # Get generation logging settings
        enable_logging, log_dir, max_samples = get_generation_logging_settings(args)

        logger.info(f"📊 Generation logging: {'enabled' if enable_logging else 'disabled'}")
        if enable_logging:
            logger.info(f"📁 Generation log directory: {log_dir}")
            logger.info(f"🔢 Max generation samples: {max_samples}")

        # Create trainer with model initialization kwargs, reward functions, and generation logging
        trainer = create_grpo_trainer(
            config=config,
            model_init_kwargs=config.model_init_kwargs,
            reward_functions=reward_functions,
            enable_generation_logging=enable_logging,
            generation_log_dir=log_dir,
            max_generation_samples=max_samples)
        logger.info("✅ GRPO Trainer created")
        
        if args.eval_only:
            # Only run evaluation
            logger.info("📊 Running evaluation only")
            if eval_dataset is None:
                logger.error("❌ No evaluation dataset available")
                return 1
            
            eval_result = trainer.evaluate(eval_dataset)
            logger.info(f"📊 Evaluation results: {eval_result}")
            
        else:
            # Run training
            logger.info("🚀 Starting training")

            # Resume from checkpoint if specified
            if args.resume_from_checkpoint:
                logger.info(f"🔄 Resuming from checkpoint: {args.resume_from_checkpoint}")
                # Note: Actual checkpoint resuming would need to be implemented

            # Start training
            training_result = trainer.train(train_dataset, eval_dataset)

            logger.info("✅ Training completed")

            # Save model
            trainer.save_model()
            logger.info("💾 Model saved")
            
            # Run final evaluation
            if eval_dataset:
                logger.info("📊 Running final evaluation")
                eval_result = trainer.evaluate(eval_dataset)
                logger.info(f"📊 Final evaluation results: {eval_result}")
        
        logger.info("🎉 GRPO training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
