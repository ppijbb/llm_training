#!/usr/bin/env python3
"""
G3MoE SFT Training Script using TRL SFTTrainer with DeepSpeed Support
"""

import os
import sys
import json
import torch
import argparse
from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
import wandb

# Add parent directory to path to import custom modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules  
from models.g3moe_model import G3MoEForCausalLM, G3MoEConfig
from models.g3moe_config import G3MoETextConfig
from data.base_mode_sft_dataset import get_dataset


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    
    model_name_or_path: str = field(
        default="google/gemma-3-4b-it",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer. If None, uses model_name_or_path"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for efficient fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code when loading model"}
    )
    deepspeed_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed configuration file"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    dataset_name: str = field(
        default="Gunulhona/open_m_3",
        metadata={"help": "The name of the dataset to use"}
    )
    max_seq_length: int = field(
        default=131072,
        metadata={"help": "Maximum sequence length"}
    )
    test_size: float = field(
        default=0.1,
        metadata={"help": "Test size for dataset split"}
    )
    text_only: bool = field(
        default=False,
        metadata={"help": "Whether to use text-only mode"}
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use streaming dataset"}
    )


def load_deepspeed_config(config_path: str) -> dict:
    """Load DeepSpeed configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DeepSpeed config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded DeepSpeed config from: {config_path}")
    return config


def setup_deepspeed_environment():
    """Setup environment variables for DeepSpeed optimization"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Enable DeepSpeed optimizations
    if "DEEPSPEED_ZERO_INIT" not in os.environ:
        os.environ["DEEPSPEED_ZERO_INIT"] = "1"
    
    print("DeepSpeed environment variables set")


def setup_model_and_tokenizer(model_args: ModelArguments):
    """Setup G3MoE model and tokenizer"""
    
    # Setup DeepSpeed environment if config is provided
    if model_args.deepspeed_config:
        setup_deepspeed_environment()
    
    # Load tokenizer
    tokenizer_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    try:
        tokenizer = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_args.trust_remote_code
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_args.trust_remote_code
        )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model configuration
    try:
        # Try to load as G3MoE config first
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code
        )
        base_model_config = config.to_dict()
        base_model_config['text_config'].update(
            {
                "n_shared_experts": 1,
                "n_routed_experts": 2, # 256, 15, 6
                "n_group": 4,
                "topk_group": 8,
                "num_key_value_heads": base_model_config['text_config']['num_attention_heads'],
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 8,
                "router_aux_loss_coef": 0.001,
                "router_jitter_noise": 0.01,
                "input_jitter_noise": 0.01,
                "model_type": "g3moe_text",
                "rope_scaling":{
                    "rope_type": "linear",
                    "factor": 8.0
                },
                # "intermediate_size": base_config['text_config']['hidden_size'],
                "use_bfloat16": True,
            }
        )
        config = G3MoETextConfig(**base_model_config)
        print("Loaded G3MoE config")
    except:
        # Fallback to text config only
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code
        )
        print("Loaded G3MoE text config")
    
    # Load model - use different device_map strategy based on DeepSpeed usage
    device_map = None
    if model_args.deepspeed_config:
        # With DeepSpeed, let DeepSpeed handle device placement
        device_map = None
        print("Using DeepSpeed - letting DeepSpeed handle device placement")
    elif torch.cuda.device_count() > 1:
        # Without DeepSpeed, use auto device mapping for multi-GPU
        device_map = "auto"
        print(f"Using auto device mapping for {torch.cuda.device_count()} GPUs")
    
    try:
        model = G3MoEForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=model_args.trust_remote_code,
            device_map=device_map,
        )
        print("Loaded G3MoE model successfully")
    except Exception as e:
        print(f"Error loading G3MoE model: {e}")
        print("This might be because the model is not yet available. Using base Gemma model...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=model_args.trust_remote_code,
            device_map=device_map,
        )
    
    # Setup LoRA if requested
    if model_args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def setup_dataset(data_args: DataArguments, tokenizer):
    """Setup training dataset"""
    
    print(f"Loading dataset: {data_args.dataset_name}")
    dataset = get_dataset(
        dataset_name=data_args.dataset_name,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        test_size=data_args.test_size,
        text_only=data_args.text_only,
        streaming=data_args.streaming
    )
    
    print(f"Dataset loaded:")
    for split, data in dataset.items():
        print(f"  {split}: {len(data)} examples")
    
    return dataset


def setup_training_args_with_deepspeed(training_args: SFTConfig, model_args: ModelArguments):
    """Setup training arguments with DeepSpeed configuration"""
    
    if model_args.deepspeed_config:
        # Load DeepSpeed configuration
        deepspeed_config = load_deepspeed_config(model_args.deepspeed_config)
        
        # Set DeepSpeed config in training arguments
        training_args.deepspeed = model_args.deepspeed_config
        
        # Override some training arguments based on DeepSpeed config
        if "gradient_accumulation_steps" in deepspeed_config:
            training_args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
            print(f"Set gradient_accumulation_steps to {deepspeed_config['gradient_accumulation_steps']} from DeepSpeed config")
        
        if "train_micro_batch_size_per_gpu" in deepspeed_config:
            training_args.per_device_train_batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            print(f"Set per_device_train_batch_size to {deepspeed_config['train_micro_batch_size_per_gpu']} from DeepSpeed config")
        
        # Enable gradient checkpointing if specified in DeepSpeed config
        if deepspeed_config.get("activation_checkpointing", {}).get("partition_activations", False):
            training_args.gradient_checkpointing = True
            print("Enabled gradient checkpointing from DeepSpeed config")
        
        # Set FP16/BF16 based on DeepSpeed config
        if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
            training_args.fp16 = True
            training_args.bf16 = False
            print("Enabled FP16 from DeepSpeed config")
        elif "bf16" in deepspeed_config and deepspeed_config["bf16"]["enabled"]:
            training_args.bf16 = True
            training_args.fp16 = False
            print("Enabled BF16 from DeepSpeed config")
        
        print(f"DeepSpeed configuration applied: {model_args.deepspeed_config}")
    
    return training_args


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from JSON file
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Parse from command line
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup training arguments with DeepSpeed if specified
    training_args = setup_training_args_with_deepspeed(training_args, model_args)
    
    # Set seed
    set_seed(training_args.seed)
    
    # Setup logging
    import logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Initialize wandb if needed
    if training_args.report_to and "wandb" in training_args.report_to:
        wandb.init(
            project="g3moe-sft",
            name=training_args.run_name,
            config={
                **vars(model_args),
                **vars(data_args),
                **vars(training_args),
            }
        )
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Setup dataset
    print("Setting up dataset...")
    dataset = setup_dataset(data_args, tokenizer)
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        dataset_text_field="input_ids",  # Since we're using pre-tokenized data
        packing=False,  # Don't pack sequences since we have labels
    )
    
    # Print training info
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Dataset: {data_args.dataset_name}")
    print(f"Max sequence length: {data_args.max_seq_length}")
    print(f"Use LoRA: {model_args.use_lora}")
    if model_args.use_lora:
        print(f"LoRA rank: {model_args.lora_r}")
    print(f"DeepSpeed config: {model_args.deepspeed_config or 'None'}")
    print(f"Training steps: {training_args.max_steps if training_args.max_steps > 0 else 'N/A'}")
    print(f"Training epochs: {training_args.num_train_epochs}")
    print(f"Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"FP16: {training_args.fp16}")
    print(f"BF16: {training_args.bf16}")
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
    # Example usage with default arguments
    if len(sys.argv) == 1:
        print("Running with default arguments...")
        sys.argv.extend([
            "--model_name_or_path", "google/gemma-3-4b-it",
            "--output_dir", "./g3moe_sft_output",
            "--num_train_epochs", "3",
            "--per_device_train_batch_size", "2",
            "--per_device_eval_batch_size", "2",
            "--gradient_accumulation_steps", "8",
            "--learning_rate", "2e-5",
            "--weight_decay", "0.01",
            "--lr_scheduler_type", "cosine",
            "--warmup_ratio", "0.1",
            "--logging_steps", "10",
            "--eval_steps", "500",
            "--save_steps", "500",
            "--save_total_limit", "3",
            "--evaluation_strategy", "steps",
            "--load_best_model_at_end", "True",
            "--metric_for_best_model", "eval_loss",
            "--greater_is_better", "False",
            "--fp16", "False",
            "--bf16", "True",
            "--dataloader_pin_memory", "False",
            "--remove_unused_columns", "False",
            "--report_to", "wandb",
            "--run_name", "g3moe-sft-run",
        ])
    
    main()
