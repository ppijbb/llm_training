#!/usr/bin/env python3
from transformers.utils import is_flash_attn_2_available
import inspect

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

import unsloth
import flash_attn

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
from typing import Optional, Dict, Any
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from grpo_trainer import create_grpo_trainer
from trl import GRPOConfig
from data_loader import GRPODataLoader, create_grpo_dataloader
from config import (
    create_default_config, 
    load_config_from_file, 
    get_quick_test_config,
    get_production_config,
    ConfigManager
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
        help="Path to configuration file"
    )
    
    # Training modes
    parser.add_argument(
        "--quick-test",
        action="store_true",
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
    
    # Reward function options
    parser.add_argument(
        "--reward-function",
        type=str,
        default="systematic",
        choices=["systematic", "group_relative", "multi_objective"],
        help="Reward function type (default: systematic)"
    )
    
    parser.add_argument(
        "--reward-config",
        type=str,
        default="default",
        choices=["default", "balanced", "aggressive"],
        help="Reward function configuration (default: default)"
    )
    
    parser.add_argument(
        "--custom-reward-config",
        type=str,
        help="Path to custom reward function configuration JSON file"
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
    
    return parser.parse_args()


def create_config_from_args(args) -> GRPOConfig:
    """Create configuration from command line arguments"""
    logger.info("🔧 Creating configuration from arguments")
    
    # Start with base configuration
    if args.config:
        config = load_config_from_file(args.config)
        logger.info(f"📁 Loaded config from {args.config}")
    elif args.quick_test:
        config = get_quick_test_config()
        logger.info("⚡ Using quick test configuration")
    elif args.production:
        config = get_production_config(args.model)
        logger.info("🏭 Using production configuration")
    else:
        config = create_default_config(args.model, args.dataset)
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
    
    # Reward function configuration
    custom_config["reward_function_type"] = args.reward_function
    custom_config["reward_config_name"] = args.reward_config
    logger.info(f"🎯 Reward function: {args.reward_function} ({args.reward_config})")
    
    # Load custom reward configuration if provided
    if args.custom_reward_config:
        try:
            with open(args.custom_reward_config, 'r') as f:
                custom_reward_config = json.load(f)
            custom_config["custom_reward_config"] = custom_reward_config
            logger.info(f"📁 Custom reward config loaded: {args.custom_reward_config}")
        except Exception as e:
            logger.error(f"❌ Failed to load custom reward config: {e}")
            return None
    
    
    # Apply custom configuration
    if custom_config:
        for key, value in custom_config.items():
            setattr(config, key, value)
    
    return config


def load_dataset(args, config: GRPOConfig):
    """Load dataset based on arguments"""
    logger.info("📦 Loading dataset")
    
    # Create data loader
    model_name = config.model_init_kwargs.get("model_name", "unsloth/Qwen3-0.6B-bnb-4bit")
    max_length = getattr(config, 'max_prompt_length', 2048)
    data_loader = GRPODataLoader(
        model_name=model_name,
        max_length=max_length,
        batch_size=config.per_device_train_batch_size,
        use_processor=False
    )
    
    # Load dataset
    if args.custom_data:
        logger.info(f"📁 Loading custom data from {args.custom_data}")
        dataset = data_loader.load_custom_dataset(args.custom_data)
    else:
        dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
        max_samples = 1000
        logger.info(f"📦 Loading dataset: {dataset_name}")
        dataset = data_loader.load_dataset(
            dataset_name=dataset_name,
            max_samples=max_samples,
            streaming=False
        )
    
    # Prepare for GRPO
    train_dataset = data_loader.prepare_grpo_data(dataset, "ultrafeedback")
    
    # Split into train/eval if needed
    if len(train_dataset) > 100:  # Only split if we have enough data
        train_size = int(0.9 * len(train_dataset))
        eval_size = len(train_dataset) - train_size
        
        train_dataset = train_dataset.select(range(train_size))
        eval_dataset = train_dataset.select(range(train_size, train_size + eval_size))
        
        logger.info(f"📊 Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
    else:
        eval_dataset = None
        logger.info(f"📊 Using full dataset for training: {len(train_dataset)} samples")
    
    return train_dataset, eval_dataset


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
        manager = ConfigManager()
        if not manager.validate_config(config):
            logger.error("❌ Configuration validation failed")
            return 1
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"📁 Output directory: {config.output_dir}")
        
        # Load dataset
        train_dataset, eval_dataset = load_dataset(args)
        
        if len(train_dataset) == 0:
            logger.error("❌ No training data found")
            return 1
        
        # Create trainer with reward configuration
        trainer = create_grpo_trainer(config)
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
