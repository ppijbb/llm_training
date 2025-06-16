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
from transformers.utils.import_utils import is_flash_attn_2_available
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
from data.base_model_sft_dataset import get_dataset, create_multimodal_collate_fn
from data.simple_sft_dataset import get_simple_sft_dataset, create_simple_collate_fn, smoltalk_dataset, orca_mini_dataset

from training_utils.utils import format_parameters, load_config, setup_deepspeed_environment
from eval.callbacks import get_model_eval_callback


def load_config(config_path: str):
    """간단한 config 로더"""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_deepspeed_environment():
    """Setup environment variables for DeepSpeed optimization"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Enable DeepSpeed optimizations
    if "DEEPSPEED_ZERO_INIT" not in os.environ:
        os.environ["DEEPSPEED_ZERO_INIT"] = "1"
    
    print("DeepSpeed environment variables set")


def setup_model_and_tokenizer(model_config: Dict[str, Any]):
    """Setup G3MoE model and tokenizer"""
    
    # Setup DeepSpeed environment if config is provided
    if model_config.get("deepspeed_config"):
        setup_deepspeed_environment()
    
    # Load tokenizer - 안정적인 로딩 로직
    tokenizer_path = model_config.get("tokenizer_name_or_path") or model_config["model_name_or_path"]
    print(f"토크나이저 로딩 시도: {tokenizer_path}")
    
    tokenizer = None
    try:
        print("  - AutoProcessor 시도...")
        tokenizer = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_config["trust_remote_code"]
        )
        print("  ✅ AutoProcessor 로드 성공")
    except Exception as e:
        print(f"  ❌ AutoProcessor 실패: {e}")
        try:
            print("  - AutoTokenizer 시도...")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=model_config["trust_remote_code"]
            )
            print("  ✅ AutoTokenizer 로드 성공")
        except Exception as e2:
            print(f"  ❌ AutoTokenizer도 실패: {e2}")
            raise RuntimeError(f"토크나이저 로딩 실패: {e2}")
    
    # Set chat template with error handling
    try:
        with open("/home/conan_jung/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
            chat_template = f.read()
        
        # AutoProcessor인 경우 tokenizer 속성에 설정
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer.tokenizer.chat_template = chat_template
            print("  ✅ 채팅 템플릿을 tokenizer.tokenizer에 설정")
        else:
            tokenizer.chat_template = chat_template
            print("  ✅ 채팅 템플릿을 tokenizer에 설정")
        
        print(f"  - 템플릿 길이: {len(chat_template)}")
    except Exception as e:
        print(f"  ⚠️ 채팅 템플릿 설정 실패: {e}")
        print("  - 기본 템플릿으로 계속 진행")
    
    # Set padding side for multimodal models
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer.tokenizer.padding_side = "right"
        print("  ✅ tokenizer.tokenizer.padding_side = 'right' 설정")
    else:
        tokenizer.padding_side = "right"
        print("  ✅ tokenizer.padding_side = 'right' 설정")
    
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
    
    # Handle different model config structures (Gemma vs others)
    if 'text_config' in base_model_config:
        # Multi-modal model with text_config
        text_config = base_model_config['text_config']
        num_attention_heads = text_config['num_attention_heads']
    else:
        # Direct text model config
        text_config = base_model_config
        num_attention_heads = base_model_config['num_attention_heads']
    
    g3moe_config = {
        "n_shared_experts": g3moe_params["n_shared_experts"],
        "n_routed_experts": g3moe_params["n_routed_experts"],
        "n_group": g3moe_params["n_group"],
        "topk_group": g3moe_params["topk_group"],
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
    base_model_config["text_config"].update(g3moe_config)
    # Create G3MoE configuration
    config = G3MoEConfig(
        text_config=base_model_config["text_config"],
        vision_config=base_model_config["vision_config"],
        boi_token_index=base_model_config["boi_token_index"],
        eoi_token_index=base_model_config["eoi_token_index"],
        image_token_index=base_model_config["image_token_index"],
        initializer_range=base_model_config["initializer_range"],
    )
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
    
    model = G3MoEForCausalLM.from_pretrained(
        model_config["model_name_or_path"],
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_config["trust_remote_code"],
        device_map=device_map,
        low_cpu_mem_usage=True,
        # load_in_4bit=True,
        _attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    )
    print("✓ G3MoE model loaded successfully")
    
    
    total_params = model.num_parameters()
    print(f"  - Total parameters: {format_parameters(total_params)}")

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
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def setup_dataset(data_config: Dict[str, Any], tokenizer):
    """Setup training dataset"""    
    dataset_name = data_config.get("dataset_name", "HuggingFaceTB/smoltalk")
    max_samples = data_config.get("max_samples", 10000)
    max_seq_length = data_config.get("max_seq_length", 131072)
    test_size = data_config.get("test_size", 0.1)
    
    print(f"Loading simple SFT dataset: {dataset_name}")
    print(f"  - Max samples: {max_samples}")
    print(f"  - Max sequence length: {max_seq_length}")
    print(f"  - Test size: {test_size}")
    print(f"  - 토크나이저 타입: {type(tokenizer)}")
    print(f"  - 토크나이저에 chat_template 있음: {hasattr(tokenizer, 'chat_template')}")
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print(f"  - chat_template 길이: {len(str(tokenizer.chat_template))}")
    else:
        print(f"  - ⚠️ chat_template이 설정되지 않음!")
    
    # print(f"Loading dataset: {data_config['dataset_name']}")
    # dataset = get_dataset(
    #     dataset_name=data_config["dataset_name"],
    #     tokenizer=tokenizer,
    #     max_length=data_config["max_seq_length"],
    #     test_size=data_config["test_size"],
    #     text_only=data_config["text_only"],
    #     streaming=data_config["streaming"]
    # )
    try:
        # 간단한 데이터셋 로더 사용
        if "smoltalk" in dataset_name.lower():
            dataset = smoltalk_dataset(tokenizer, max_samples=max_samples)
        elif "orca" in dataset_name.lower():
            dataset = orca_mini_dataset(tokenizer, max_samples=max_samples)
        else:
            # 일반적인 데이터셋 로더 시도
            print(f"일반 데이터셋 로더 시도: {dataset_name}")
            dataset = get_simple_sft_dataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                max_samples=max_samples,
                test_size=test_size
            )
        
        print(f"Dataset loaded:")
        for split, data in dataset.items():
            print(f"  {split}: {len(data)} examples")
        
        # 빈 데이터셋 체크
        if len(dataset.get("train", [])) == 0:
            raise ValueError("훈련 데이터셋이 비어있습니다!")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 데이터셋 로딩 실패: {e}")
        print("🔄 대안 데이터셋으로 재시도 (SmolTalk)")
        try:
            dataset = smoltalk_dataset(tokenizer, max_samples=max_samples)
            print(f"대안 데이터셋 로드 성공:")
            for split, data in dataset.items():
                print(f"  {split}: {len(data)} examples")
            return dataset
        except Exception as e2:
            print(f"❌ 대안 데이터셋도 실패: {e2}")
            raise RuntimeError(f"모든 데이터셋 로딩 시도가 실패했습니다: {e2}")


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
    # collate_fn = create_multimodal_collate_fn(tokenizer)
    print("Creating data collator...")
    collate_fn = create_simple_collate_fn(tokenizer)

    # Setup trainer
    print("Setting up trainer...")
    
    # 데이터셋 검증
    train_dataset = dataset.get("train", None)
    eval_dataset = dataset.get("test", None)
    
    if train_dataset is None or len(train_dataset) == 0:
        raise ValueError(f"훈련 데이터셋이 비어있습니다! 데이터셋 로딩을 확인하세요.")
    
    print(f"✅ 데이터셋 검증 완료:")
    print(f"  - 훈련 데이터: {len(train_dataset)} 샘플")
    if eval_dataset is not None:
        print(f"  - 평가 데이터: {len(eval_dataset)} 샘플")
    else:
        print(f"  - 평가 데이터: 없음")
    
    # SFTTrainer에서 사용할 수 있도록 데이터셋 형태를 한번 더 확인
    print("데이터셋 샘플 확인:")
    print(f"  - 첫 번째 훈련 샘플 키: {list(train_dataset[0].keys())}")
    print(f"  - 첫 번째 샘플 input_ids 길이: {len(train_dataset[0]['input_ids'])}")
    
    trainer = SFTTrainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collate_fn,
    )
    # trainer.add_callback(get_model_eval_callback(trainer=trainer))
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
