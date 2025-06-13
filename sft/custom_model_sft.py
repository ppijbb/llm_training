#!/usr/bin/env python3
"""
G3MoE SFT Training Script using Config File
"""

import os
import sys
import json
import torch
import argparse
from typing import Dict, Any
import io
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
import wandb

# Add parent directory to path to import custom modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules  
from models import G3MoEForCausalLM, G3MoEConfig
from data.base_model_sft_dataset import get_dataset, process_vision_info, create_multimodal_collate_fn
from training_utils.utils import format_parameters, load_config, setup_deepspeed_environment
from eval.callbacks import ModelEvalCallback


def setup_model_and_tokenizer(model_config: Dict[str, Any]):
    """Setup G3MoE model and tokenizer"""
    
    # Setup DeepSpeed environment if config is provided
    if model_config.get("deepspeed_config"):
        setup_deepspeed_environment()
    
    # Load tokenizer
    tokenizer_path = model_config.get("tokenizer_name_or_path") or model_config["model_name_or_path"]
    try:
        tokenizer = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_config["trust_remote_code"]
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_config["trust_remote_code"]
        )
    
    with open("/home/conan_jung/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
        chat_template = f.read()
    tokenizer.chat_template = chat_template
    
    # Set padding side for multimodal models
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer.tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "right"
    
    # Ensure tokenizer has pad token
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load and configure G3MoE model configuration
    print("Loading base model configuration...")
    base_config = AutoConfig.from_pretrained(
        model_config["model_name_or_path"],
        trust_remote_code=model_config["trust_remote_code"]
    )
    
    # Convert to dict and update with G3MoE parameters
    base_model_config = base_config.to_dict()
    
    # G3MoE configuration parameters from config file
    g3moe_params = model_config["g3moe_params"]
    g3moe_config = {
        "n_shared_experts": g3moe_params["n_shared_experts"],
        "n_routed_experts": g3moe_params["n_routed_experts"],
        "n_group": g3moe_params["n_group"],
        "topk_group": g3moe_params["topk_group"],
        "num_key_value_heads": base_model_config['text_config']['num_attention_heads'],
        "num_experts_per_tok": g3moe_params["num_experts_per_tok"],
        "first_k_dense_replace": 8,  # Fixed parameter
        "router_aux_loss_coef": 0.001,  # Fixed parameter
        "router_jitter_noise": 0.01,  # Fixed parameter
        "input_jitter_noise": 0.01,  # Fixed parameter
        "model_type": "g3moe_text",
        "rope_scaling": {
            "rope_type": "linear",
            "factor": g3moe_params["rope_scaling_factor"]
        },
        "use_bfloat16": True,
    }
    
    # Update text_config with G3MoE parameters
    base_model_config['text_config'].update(g3moe_config)
    
    # Create G3MoE configuration
    config = G3MoEConfig(**base_model_config)
    print("G3MoE configuration created successfully")
    print(f"  - Shared experts: {g3moe_config['n_shared_experts']}")
    print(f"  - Routed experts: {g3moe_config['n_routed_experts']}")
    print(f"  - Expert groups: {g3moe_config['n_group']}")
    print(f"  - Top-k per group: {g3moe_config['topk_group']}")
    print(f"  - Experts per token: {g3moe_config['num_experts_per_tok']}")
    
    # Load model - use different device_map strategy based on DeepSpeed usage
    device_map = None
    if model_config.get("deepspeed_config"):
        # With DeepSpeed, let DeepSpeed handle device placement
        device_map = None
        print("Using DeepSpeed - letting DeepSpeed handle device placement")
    elif torch.cuda.device_count() > 1:
        # Without DeepSpeed, use auto device mapping for multi-GPU
        device_map = "auto"
        print(f"Using auto device mapping for {torch.cuda.device_count()} GPUs")
    
    # Load G3MoE model with the configured parameters
    print("Loading G3MoE model...")
    try:
        model = G3MoEForCausalLM.from_pretrained(
            model_config["model_name_or_path"],
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=model_config["trust_remote_code"],
            device_map=device_map,
        )
        print("✓ G3MoE model loaded successfully")
        
       
        total_params = model.num_parameters()
        print(f"  - Total parameters: {format_parameters(total_params)}")
        
    except Exception as e:
        print(f"✗ Error loading G3MoE model: {e}")
        print("Falling back to base Gemma model...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name_or_path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=model_config["trust_remote_code"],
            device_map=device_map,
        )
        print("✓ Base Gemma model loaded as fallback")
    
    # Setup LoRA if requested
    if model_config["use_lora"]:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_config["lora_r"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            target_modules=[
                # "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def setup_dataset(data_config: Dict[str, Any], tokenizer):
    """Setup training dataset"""
    
    print(f"Loading dataset: {data_config['dataset_name']}")
    dataset = get_dataset(
        dataset_name=data_config["dataset_name"],
        tokenizer=tokenizer,
        max_length=data_config["max_seq_length"],
        test_size=data_config["test_size"],
        text_only=data_config["text_only"],
        streaming=data_config["streaming"]
    )
    
    print(f"Dataset loaded:")
    for split, data in dataset.items():
        print(f"  {split}: {len(data)} examples")
    
    return dataset


def create_training_args(
    training_config: Dict[str, Any], 
    deepspeed_config: str = None
) -> SFTConfig:
    """Create SFTConfig from training configuration"""
    
    # Create SFTConfig with all parameters
    training_args = SFTConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_ratio=training_config["warmup_ratio"],
        logging_steps=training_config["logging_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        eval_strategy=training_config["eval_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        dataloader_pin_memory=training_config["dataloader_pin_memory"],
        remove_unused_columns=training_config["remove_unused_columns"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        report_to=training_config["report_to"],
        run_name=training_config["run_name"],
        seed=training_config["seed"],
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    
    # Add DeepSpeed config if provided
    if deepspeed_config:
        training_args.deepspeed = deepspeed_config
        print(f"DeepSpeed config set: {deepspeed_config}")
    
    return training_args


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="G3MoE SFT Training with Config File")
    parser.add_argument(
        "--config", 
        type=str, 
        default="sft/config/g3moe_training_config.json",
        help="Path to training configuration JSON file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    model_config = config["model_config"]
    data_config = config["data_config"]
    training_config = config["training_config"]
    
    # Set seed
    set_seed(training_config["seed"])
    
    # Setup logging
    import logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Initialize wandb if needed
    if training_config.get("report_to") and "wandb" in training_config["report_to"]:
        wandb.init(
            project="g3moe-sft",
            name=training_config["run_name"],
            config=config
        )
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_config)
    
    # Setup dataset
    print("Setting up dataset...")
    dataset = setup_dataset(data_config, tokenizer)
    
    # Create training arguments
    training_args = create_training_args(
        training_config, 
        model_config.get("deepspeed_config")
    )
    
    # Create multimodal data collator
    print("Creating multimodal data collator...")
    collate_fn = create_multimodal_collate_fn(tokenizer)
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = SFTTrainer( 
        model=model,
        args=training_args,
        train_dataset=dataset.get("train", None),
        eval_dataset=dataset.get("test", None),
        processing_class=tokenizer,
        data_collator=collate_fn,
    )
    
    # Print training info
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Model: {model_config['model_name_or_path']}")
    print(f"Dataset: {data_config['dataset_name']}")
    print(f"Max sequence length: {data_config['max_seq_length']}")
    print(f"Use LoRA: {model_config['use_lora']}")
    if model_config['use_lora']:
        print(f"LoRA rank: {model_config['lora_r']}")
    print(f"DeepSpeed config: {model_config.get('deepspeed_config', 'None')}")
    print(f"Training epochs: {training_config['num_train_epochs']}")
    print(f"Batch size per device: {training_config['per_device_train_batch_size']}")
    print(f"Gradient accumulation steps: {training_config['gradient_accumulation_steps']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"FP16: {training_config['fp16']}")
    print(f"BF16: {training_config['bf16']}")
    print("="*50)
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
