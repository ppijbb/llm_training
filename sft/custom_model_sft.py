#!/usr/bin/env python3
"""
G3MoE SFT Training Script using Config File
"""

import os
import sys
import json
import torch
import traceback
import argparse
from typing import Dict, Any
from torchinfo import summary
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM
)
from transformers import logging

from transformers.trainer_utils import set_seed
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import TaskType
import wandb

# Add parent directory to path to import custom modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules  
from models import G3MoEForCausalLM, G3MoEConfig, G3MoEForConditionalGeneration, G3MoETextConfig, G3MoETextModel, G3MoEModel
from data.base_model_sft_dataset import get_dataset, create_multimodal_collate_fn
from data.simple_sft_dataset import get_simple_sft_dataset, create_simple_collate_fn, smoltalk_dataset, orca_mini_dataset

from training_utils.utils import format_parameters, load_config, setup_deepspeed_environment
from optimizers.custom_optimizers import get_custom_optimizer
from optimizers.deepspeed_optimizer_registry import register_custom_optimizers
from eval.callbacks import ModelEvalCallback
from eval.ifeval_callback import IFEvalCallback
from eval.moe_monitoring_callback import create_moe_callback_for_transformers

# Register custom optimizers with DeepSpeed
register_custom_optimizers()
try:
    # AutoConfig.register("g3moe", G3MoEConfig)
    AutoConfig.register("g3moe", G3MoEConfig)
    AutoConfig.register("g3moe_text", G3MoETextConfig)
    AutoModel.register(G3MoEConfig, G3MoEModel)
    AutoModel.register(G3MoETextConfig, G3MoETextModel)
    AutoModelForCausalLM.register(G3MoETextConfig, G3MoEForCausalLM)

    from transformers.modeling_utils import VLMS
    VLMS.append("g3moe")
except Exception as e:
    import traceback
    traceback.format_exc()
    print(f"Failed to register G3MoE model: {e}")
    print("G3MoE cannot train without registering model... exiting...")
    raise e

logging.enable_progress_bar()
logging.set_verbosity_warning()

def load_config(config_path: str):
    """간단한 config 로더"""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_deepspeed_environment():
    """Setup environment variables for DeepSpeed optimization"""
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if "DEEPSPEED_ZERO_INIT" not in os.environ:
        os.environ["DEEPSPEED_ZERO_INIT"] = "1"
    # Ensure global AMP default uses BF16 under CUDA
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch.set_autocast_gpu_dtype(torch.bfloat16)
    except Exception as _:
        pass
    print("DeepSpeed environment variables set")


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def eval_with_memory_optimization(trainer):
    """Memory-optimized evaluation function"""
    print("🔧 Memory-optimized evaluation 시작...")
    
    # GPU 메모리 정리
    clear_gpu_memory()
    
    # 모델을 eval 모드로 설정하고 메모리 최적화
    trainer.model.eval()
    
    # eval 시에는 gradient checkpointing 비활성화
    original_gc = trainer.args.gradient_checkpointing
    trainer.args.gradient_checkpointing = False
    
    try:
        with torch.no_grad():
            # eval 실행
            eval_results = trainer.evaluate()
            
        # 결과 반환
        return eval_results
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("❌ Eval 중 CUDA OOM 발생! 메모리 정리 후 재시도...")
            clear_gpu_memory()
            # 더 작은 배치로 재시도
            original_eval_batch_size = trainer.args.per_device_eval_batch_size
            trainer.args.per_device_eval_batch_size = 1
            try:
                with torch.no_grad():
                    eval_results = trainer.evaluate()
                return eval_results
            finally:
                trainer.args.per_device_eval_batch_size = original_eval_batch_size
        else:
            raise e
    finally:
        # 원래 설정 복원
        trainer.args.gradient_checkpointing = original_gc
        clear_gpu_memory()


def setup_model_and_tokenizer(model_config: Dict[str, Any]):
    """Setup G3MoE model and tokenizer"""
    
    # NOTE: Delay DeepSpeed env setup until AFTER model load to avoid HF ZeRO-3 init slow path
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
        with open("/home/conan/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
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
    if not hasattr(tokenizer, 'pad_token'):
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer.pad_token = tokenizer.tokenizer.pad_token if tokenizer.tokenizer.pad_token is not None else tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if not hasattr(tokenizer, 'convert_tokens_to_ids'):
        tokenizer.convert_tokens_to_ids = tokenizer.tokenizer.convert_tokens_to_ids

    # Prefer config value; default to eager
    attn_from_cfg = (model_config.get("g3moe_params") or {}).get("attn_implementation")
    if attn_from_cfg in {"eager", "sdpa", "flash_attention_2"}:
        attn_implementation = attn_from_cfg
    else:
        attn_implementation = "eager"

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
        "first_k_dense_replace": g3moe_params["first_k_dense_replace"],
        "router_aux_loss_coef": g3moe_params["router_aux_loss_coef"],
        "router_jitter_noise": g3moe_params["router_jitter_noise"],
        "input_jitter_noise": g3moe_params["input_jitter_noise"],
        "model_type": "g3moe_text",
        "rope_scaling": {
            "rope_type": g3moe_params["rope_scaling"]["rope_type"],
            "factor": g3moe_params["rope_scaling"]["factor"]
        },
        "use_bfloat16": True,
        "attn_implementation": attn_implementation
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
        attn_implementation=attn_implementation,
        **{
            k:v for k,v in base_model_config.items() 
            if k not in [
                "text_config", "vision_config", "boi_token_index",
                "eoi_token_index", "image_token_index", "initializer_range",
                "attn_implementation"
            ]
        }
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
    model = G3MoEForConditionalGeneration.from_pretrained(
        model_config["model_name_or_path"],
        config=config,
        torch_dtype=torch.bfloat16, # Using bfloat16
        trust_remote_code=model_config["trust_remote_code"],
        device_map=device_map,
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        use_cache=False,
        gradient_checkpointing=False,
        # load_in_4bit=True,
        attn_implementation=attn_implementation
    )
    print("✓ G3MoE model loaded successfully")
    print(f"  - Attn implementation: {attn_implementation}")
    total_params = model.num_parameters()
    print(f"  - Total parameters: {format_parameters(total_params)}")

    # Setup LoRA if requested
    if model_config["use_lora"]:
        lora_config = LoraConfig(
            r=model_config["lora_r"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            target_modules=[
                # "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "router", "routing_temperature",
                "rnn.weight_ih_l0", "rnn.weight_hh_l0"
            ],
            modules_to_save=["router", "routing_temperature"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,  # 훈련 모드 명시
            fan_in_fan_out=False,  # LoRA 호환성 향상
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()
        
        # LoRA 어댑터 설정
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
            # Ensure router modules remain fully trainable (not LoRA-wrapped)
            if name.endswith('router') or name.split('.')[-1] == 'router':
                for p in module.parameters(recurse=True):
                    p.requires_grad_(True)
        # DDP 정적 그래프 비활성화: MoE 라우팅/LoRA로 스텝마다 활성 파라미터가 달라질 수 있으므로 동적 그래프 허용
        if hasattr(model, '_set_static_graph'):
            model._set_static_graph(True)
        # Ensure all parameters incl. LoRA adapters are bfloat16 for consistency
        try:
            model.to(torch.bfloat16)
            for name, param in model.named_parameters():
                if param.requires_grad and param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)
            print("✓ Parameters cast to bfloat16")
        except Exception as cast_e:
            print(f"⚠️ BF16 cast warning: {cast_e}")
        print("✓ LoRA 적용")
        
    return model, tokenizer


def setup_dataset(data_config: Dict[str, Any], tokenizer):
    """Setup training dataset"""    
    dataset_name = data_config.get("dataset_name", "HuggingFaceTB/smoltalk")
    max_samples = data_config.get("max_samples", 100000)
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
    with open("/home/conan/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
        chat_template = f.read()
        
        # AutoProcessor인 경우 tokenizer 속성에 설정
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer.tokenizer.chat_template = chat_template
            print("  ✅ 채팅 템플릿을 tokenizer.tokenizer에 설정")
        
        tokenizer.chat_template = chat_template
        print("  ✅ 채팅 템플릿을 tokenizer에 설정")
    
    # print(f"Loading dataset: {data_config['dataset_name']}")
    try:
        # 간단한 데이터셋 로더 사용
        if "smoltalk" in dataset_name.lower() or "orca" in dataset_name.lower() or "llava" in dataset_name.lower():
            print(f"일반 데이터셋 로더 시도: {dataset_name}")
            dataset = get_simple_sft_dataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                max_samples=max_samples,
                test_size=test_size,
                use_streaming=True
            )
            collate_fn = None # create_simple_collate_fn(tokenizer)
        else:
            # open_m_3 데이터셋 로더 시도
            dataset = get_dataset(
                tokenizer=tokenizer,
                dataset_name=data_config["dataset_name"],
                max_length=data_config["max_seq_length"],
                test_size=data_config["test_size"],
                text_only=data_config["text_only"],
                streaming=data_config["streaming"]
            )
            collate_fn = create_multimodal_collate_fn(tokenizer)
        
        print(f"Dataset loaded:")
        for split, data in dataset.items():
            print(f"  {split}: {data.info.dataset_size if bool(data_config['streaming']) else len(data)} examples")
        
        # 빈 데이터셋 체크
        if data_config.get("streaming", False):
            if dataset.get("train", []).info.dataset_size == 0:
                raise ValueError("훈련 데이터셋이 비어있습니다!")

        return dataset, collate_fn
        
    except Exception as e:
        print(f"❌ 데이터셋 로딩 실패: {e}")
        assert False, "데이터셋 로딩 실패"
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
    deepspeed_config: str | None = None 
) -> SFTConfig:
    """Create SFTConfig from training configuration"""
    
    # Create SFTConfig with all parameters
    training_args = SFTConfig(
        **training_config,
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    
    # Add DeepSpeed config if provided
    if deepspeed_config:
        import os, json
        ds_cfg_path_abs = os.path.abspath(deepspeed_config)
        training_args.deepspeed = ds_cfg_path_abs
        print(f"DeepSpeed config set: {ds_cfg_path_abs}")
        # Validate that CPU offload is disabled as required
        try:
            with open(ds_cfg_path_abs, "r") as f:
                ds_cfg = json.load(f)
            zero = ds_cfg.get("zero_optimization", {})
            off_opt = (zero.get("offload_optimizer") or {}).get("device", "none").lower()
            off_param = (zero.get("offload_param") or {}).get("device", "none").lower()
            print(f"DeepSpeed zero stage: {zero.get('stage')}")
            print(f"DeepSpeed offload_optimizer.device: {off_opt}")
            print(f"DeepSpeed offload_param.device: {off_param}")
            assert off_opt in {"none", None, ""} and off_param in {"none", None, ""}, (
                "DeepSpeed CPU offload detected in config but must be disabled (device='none')."
            )
            # Workaround: ZeRO-3 + gradient checkpointing can trigger duplicate ds_id assertion
            try:
                zero_stage = int(zero.get("stage", 0) or 0)
            except Exception:
                zero_stage = 0
            if zero_stage == 3 and getattr(training_args, "gradient_checkpointing", False):
                print("⚠️ Detected ZeRO-3 with gradient checkpointing enabled. Disabling to avoid ds_id assertion.")
                training_args.gradient_checkpointing = False
        except Exception as e:
            print(f"⚠️ DeepSpeed config validation warning: {e}")
    
    return training_args


def main(
    model_config: Dict[str, Any], 
    data_config: Dict[str, Any], 
    training_config: Dict[str, Any]
):
    register_custom_optimizers()
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_config)
    
    # Setup dataset
    print("Setting up dataset...")
    dataset, collate_fn = setup_dataset(data_config, tokenizer)
    
    # Create training arguments
    training_args = create_training_args(
        training_config, 
        model_config.get("deepspeed_config")
    )
    
    # Optionally build a custom optimizer (e.g., Muon) prior to DeepSpeed init
    custom_optimizer = None
    try:
        ds_cfg_path = model_config.get("deepspeed_config")
        if ds_cfg_path:
            with open(ds_cfg_path, "r") as f:
                ds_cfg = json.load(f)
            # Prefer explicit custom optimizer block
            custom_opt_section = ds_cfg.get("custom_optimizer")
            from optimizers.deepspeed_optimizer_registry import create_optimizer_from_config
            if custom_opt_section:
                trainable_params = (p for p in model.parameters() if p.requires_grad)
                custom_optimizer = create_optimizer_from_config(custom_opt_section, trainable_params)
                print(f"✓ Using custom optimizer: {custom_opt_section.get('type')}")
            else:
                # Fallback: if optimizer.type is a custom one, build it here
                opt_section = ds_cfg.get("optimizer")
                if opt_section:
                    opt_type = str(opt_section.get("type", "")).lower()
                    if opt_type in {"muon", "muonoptimizer", "lion", "adafactor", "sophia"}:
                        trainable_params = (p for p in model.parameters() if p.requires_grad)
                        custom_optimizer = create_optimizer_from_config(opt_section, trainable_params)
                        print(f"✓ Using custom optimizer from optimizer block: {opt_section.get('type')}")
    except Exception as opt_e:
        print(f"⚠️ Custom optimizer setup skipped: {opt_e}")

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
    # print(f"  - 첫 번째 샘플 input_ids 길이: {len(train_dataset[0]['input_ids'])}")
    
    trainer = SFTTrainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collate_fn,
        optimizers=(custom_optimizer, None) if custom_optimizer is not None else (None, None)
    )
    # Enforce: Disable gradient checkpointing with ZeRO-3 at runtime as an additional safeguard
    try:
        ds_cfg_path = getattr(trainer.args, "deepspeed", None)
        if ds_cfg_path:
            import json
            with open(ds_cfg_path, "r") as f:
                _zero_stage = int((json.load(f).get("zero_optimization", {}) or {}).get("stage", 0) or 0)
            if _zero_stage == 3:
                if hasattr(trainer.model, "gradient_checkpointing_disable"):
                    trainer.model.gradient_checkpointing_disable()
                trainer.args.gradient_checkpointing = False
                print("✓ Disabled gradient checkpointing for DeepSpeed ZeRO-3 compatibility")
    except Exception as _:
        pass
    trainer.add_callback(
        create_moe_callback_for_transformers(
            num_experts=model_config["g3moe_params"]["n_routed_experts"],
            log_every_n_steps=1,           # 50 스텝마다 로그 기록
            logger=wandb,                  # 사용할 로거 지정 (wandb)
            log_to_console=True,           # 콘솔에도 주요 메트릭 출력
                                           # === (선택사항) ===
            log_heatmap_every=5,           # 500 스텝마다 Expert 사용률 히트맵 로깅
            alert_threshold_imbalance=4.0, # 특정 Expert 사용률이 평균의 4배를 초과하면 경고
            unused_expert_threshold=0.25,  # 25% 이상의 Expert가 미사용되면 경고
            entropy_threshold=0.1,         # 라우팅 엔트로피가 0.1 미만이면 경고
            save_detailed_logs=False       # 상세 JSON 로그 저장 여부
        ))
    # trainer.add_callback(
    #     ModelEvalCallback(
    #         trainer=trainer,  # Will be set by Trainer
    #         enable_benchmarks=True,  # Enable benchmark evaluation
    #         benchmarks_to_run=['mmlu', 'hellaswag', 'gsm8k', 'truthfulqa', 'arc', 'piqa'],  # Run multiple benchmarks
    #         benchmark_eval_frequency=training_config["eval_steps"],  # Run benchmarks every 2 epochs
    #         mme_max_samples=10,  # Limit MME samples for faster evaluation
    #     ))
    # trainer.add_callback(
    #     IFEvalCallback(
    #         eval_dataset_name="google/IFEval",
    #         max_samples=100
    #     ))

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
    # summary(
    #     trainer.model,
    #     input_data={
    #         'input_ids': torch.randint(0, tokenizer.tokenizer.vocab_size, (1, 1024), device=trainer.model.device)
    #     }, depth=3)
    # Start training
    print("Starting training...")
    # Guard heavy profiler behind an env flag to avoid OOM from profiler buffers during full training
    enable_profiler = bool(int(os.getenv("PROFILE_TRAINING", "0")))
    if enable_profiler:
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            try:
                trainer.train()
                profiler_table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
                wandb.log({"profiler_table": wandb.Table(data=[profiler_table])})
            except Exception as e:
                traceback.print_exc()
                print(f"⚠️ Profiler error: {e}")
    else:
        try:
            # eval 최적화를 위한 커스텀 eval 함수 설정
            original_eval_fn = getattr(trainer, 'evaluate', None)
            trainer.evaluate = lambda: eval_with_memory_optimization(trainer)
            
            trainer.train()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("❌ CUDA OOM 발생! 메모리 정리 후 재시도...")
                clear_gpu_memory()
                print("GPU 메모리 정리 완료. 다시 시도해주세요.")
                raise e
            else:
                raise e
        finally:
            # 원래 eval 함수 복원
            if original_eval_fn:
                trainer.evaluate = original_eval_fn

    # Save final model
    print("Saving final model...")
    trainer.save_model()
    
    # Save tokenizer``
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    register_custom_optimizers()
    try:
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
        # Initialize wandb if needed
        if training_config.get("report_to") and "wandb" in training_config["report_to"]:
            rank = int(os.getenv("RANK", "0"))
            if rank == 0:
                wandb.init(
                    project="g3moe-sft",
                    name=training_config["run_name"],
                    config=config
                )

        main(model_config, data_config, training_config)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(torch.cuda.memory_summary())
        print(torch.cuda.max_memory_allocated())
        print(torch.cuda.max_memory_reserved())
