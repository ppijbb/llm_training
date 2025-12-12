#!/usr/bin/env python3
"""
SPECTRA SFT Training Script using Config File
"""

import os
import sys
import json
import torch
import traceback
import argparse
import logging
import time
from datetime import datetime
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
from transformers import logging as transformers_logging

from transformers.trainer_utils import set_seed
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import TaskType
import wandb

# Add parent directory to path to import custom modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules  
from models import SPECTRAForCausalLM, SPECTRAConfig, SPECTRAForConditionalGeneration, SPECTRATextConfig, SPECTRATextModel, SPECTRAModel
from data.base_model_sft_dataset import get_dataset, create_multimodal_collate_fn
from data.simple_sft_dataset import get_simple_sft_dataset, create_simple_collate_fn, smoltalk_dataset, orca_mini_dataset, validate_image_data
from data.multi_domain_sft_dataset import get_multi_domain_sft_dataset, create_simple_collate_fn as create_multi_domain_collate_fn, all_domains_dataset

from training_utils.utils import format_parameters, load_config, setup_deepspeed_environment
from optimizers.custom_optimizers import get_custom_optimizer
from optimizers.deepspeed_optimizer_registry import register_custom_optimizers
from eval.callbacks import ModelEvalCallback
# IFEval is now integrated into ModelEvalCallback - no separate callback needed
# from eval.ifeval_callback import IFEvalCallback
from eval.moe_monitoring_callback import create_moe_callback_for_transformers
from eval.router_weight_callback import RouterWeightTrackingCallback

# Register custom optimizers with DeepSpeed
register_custom_optimizers()
try:
    # AutoConfig.register("spectra", SPECTRAConfig)
    AutoConfig.register("spectra", SPECTRAConfig)
    AutoConfig.register("spectra_text", SPECTRATextConfig)
    AutoModel.register(SPECTRAConfig, SPECTRAModel)
    AutoModel.register(SPECTRATextConfig, SPECTRATextModel)
    AutoModelForCausalLM.register(SPECTRAConfig, SPECTRAForConditionalGeneration)

    from transformers.modeling_utils import VLMS
    VLMS.append("spectra")
except Exception as e:
    traceback.format_exc()
    print(f"Failed to register SPECTRA model: {e}")
    print("SPECTRA cannot train without registering model... exiting...")
    raise e

transformers_logging.enable_progress_bar()
transformers_logging.set_verbosity_warning()
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR) 

# Setup comprehensive logging system
def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Setup comprehensive logging system for training monitoring"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"training_detailed_{timestamp}.log"),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # File handler for error logs
    error_handler = logging.FileHandler(
        os.path.join(log_dir, f"training_errors_{timestamp}.log"),
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
logger = setup_logging()

def log_gpu_memory(logger, stage: str, device: int = 0):
    """Log detailed GPU memory information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3    # GB
        
        logger.info(f"🔧 GPU Memory [{stage}] - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        logger.debug(f"🔧 GPU Memory [{stage}] - Max Allocated: {max_allocated:.2f}GB, Max Reserved: {max_reserved:.2f}GB")
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'max_reserved': max_reserved
        }
    return None

def log_training_progress(logger, trainer, step: int = None, epoch: float = None, loss: float = None):
    """Log detailed training progress information"""
    if hasattr(trainer, 'state') and trainer.state is not None:
        state = trainer.state
        current_step = step or state.global_step
        current_epoch = epoch or state.epoch
        current_loss = loss or getattr(state, 'log_history', [{}])[-1].get('train_loss', 'N/A')
        
        logger.info(f"📊 Training Progress - Step: {current_step}, Epoch: {current_epoch:.3f}, Loss: {current_loss}")
        
        # Log learning rate if available
        if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler is not None:
            lr = trainer.lr_scheduler.get_last_lr()[0] if hasattr(trainer.lr_scheduler, 'get_last_lr') else 'N/A'
            logger.debug(f"📊 Learning Rate: {lr}")
        
        # Log gradient norm if available
        if hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
            if hasattr(trainer.accelerator, 'unwrap_model'):
                model = trainer.accelerator.unwrap_model(trainer.model)
                total_norm = 0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                if param_count > 0:
                    total_norm = total_norm ** (1. / 2)
                    logger.debug(f"📊 Gradient Norm: {total_norm:.6f}")

def collect_environment_info():
    """수집 가능한 모든 환경 정보를 수집"""
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'system': {},
        'python': {},
        'pytorch': {},
        'cuda': {},
        'gpu': {},
        'transformers': {},
        'deepspeed': {},
        'environment_variables': {}
    }
    
    try:
        import platform
        import sys
        import subprocess
        
        # 시스템 정보
        env_info['system'] = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'cpu_count': os.cpu_count()
        }
        
        # Python 정보
        env_info['python'] = {
            'version': sys.version,
            'version_info': list(sys.version_info),
            'executable': sys.executable,
            'path': sys.path[:10]  # 처음 10개만
        }
        
        # PyTorch 정보
        env_info['pytorch'] = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() and torch.backends.cudnn.is_available() else None,
            'cudnn_enabled': torch.backends.cudnn.enabled if torch.cuda.is_available() else None,
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
        
        # CUDA 정보
        if torch.cuda.is_available():
            env_info['cuda'] = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'device_capability': torch.cuda.get_device_capability(),
                'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
                'max_memory_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3
            }
            
            # 모든 GPU 정보
            env_info['gpu'] = {}
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    env_info['gpu'][f'device_{i}'] = {
                        'name': props.name,
                        'total_memory_gb': props.total_memory / 1024**3,
                        'major': props.major,
                        'minor': props.minor,
                        'multi_processor_count': props.multi_processor_count
                    }
                except Exception as e:
                    env_info['gpu'][f'device_{i}'] = {'error': str(e)}
        
        # Transformers 정보
        try:
            from transformers import __version__ as transformers_version
            env_info['transformers'] = {
                'version': transformers_version
            }
        except:
            pass
        
        # DeepSpeed 정보
        try:
            import deepspeed
            env_info['deepspeed'] = {
                'version': getattr(deepspeed, '__version__', 'unknown'),
                'available': True
            }
        except:
            env_info['deepspeed'] = {'available': False}
        
        # nvidia-smi 정보 (가능한 경우)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                env_info['gpu']['nvidia_smi'] = result.stdout.strip()
        except:
            pass
        
        # 중요한 환경 변수
        important_env_vars = [
            'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 'DEEPSPEED_ZERO_INIT',
            'TOKENIZERS_PARALLELISM', 'TORCH_NCCL_ASYNC_ERROR_HANDLING',
            'WANDB_PROJECT', 'WANDB_RUN_NAME', 'HF_HOME', 'TRANSFORMERS_CACHE'
        ]
        for var in important_env_vars:
            if var in os.environ:
                env_info['environment_variables'][var] = os.environ[var]
        
    except Exception as e:
        env_info['collection_error'] = str(e)
        env_info['collection_traceback'] = traceback.format_exc()
    
    return env_info

def save_oom_error_info(logger, trainer, error, batch_info=None, output_dir=None):
    """OOM 에러 발생 시 모든 정보를 JSON 파일로 저장"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 출력 디렉토리 결정
        if output_dir is None:
            if trainer is not None:
                if hasattr(trainer, 'args') and hasattr(trainer.args, 'output_dir'):
                    output_dir = trainer.args.output_dir
                elif hasattr(trainer, 'training_args') and hasattr(trainer.training_args, 'output_dir'):
                    output_dir = trainer.training_args.output_dir
                else:
                    output_dir = "logs"
            else:
                output_dir = "logs"
        
        os.makedirs(output_dir, exist_ok=True)
        error_file = os.path.join(output_dir, f"oom_error_info_{timestamp}.json")
        
        # 환경 정보 수집
        env_info = collect_environment_info()
        
        # 에러 정보
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'error_traceback': traceback.format_exc()
        }
        
        # Training state 정보
        training_state = {}
        if trainer is not None and hasattr(trainer, 'state') and trainer.state is not None:
            state = trainer.state
            training_state = {
                'global_step': state.global_step,
                'epoch': state.epoch,
                'current_loss': getattr(state, 'log_history', [{}])[-1].get('train_loss', None),
                'log_history': getattr(state, 'log_history', [])[-10:]  # 최근 10개만
            }
        
        # Model state 정보
        model_state = {}
        if trainer is not None:
            try:
                model = trainer.model
                if hasattr(model, 'module'):  # DeepSpeed 래핑
                    model = model.module
                
                first_param = next(model.parameters())
                model_state = {
                    'device': str(first_param.device),
                    'dtype': str(first_param.dtype),
                    'requires_grad': bool(first_param.requires_grad),
                    'num_parameters': sum(p.numel() for p in model.parameters()),
                    'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
                }
            except Exception as e:
                model_state = {'error': str(e)}
        
        # Batch configuration 정보
        batch_config = {}
        if trainer is not None:
            try:
                batch_config = {
                    'per_device_train_batch_size': getattr(trainer, 'per_device_train_batch_size', None),
                    'gradient_accumulation_steps': getattr(trainer, 'gradient_accumulation_steps', None),
                    'effective_batch_size': getattr(trainer, 'per_device_train_batch_size', 1) * getattr(trainer, 'gradient_accumulation_steps', 1),
                    'max_length': getattr(trainer.args, 'max_length', None) if hasattr(trainer, 'args') else None
                }
            except Exception as e:
                batch_config = {'error': str(e)}
        
        # Dataset 정보
        dataset_info = {}
        if trainer is not None:
            try:
                if hasattr(trainer, 'train_dataset') and trainer.train_dataset is not None:
                    dataset_info = {
                        'dataset_size': len(trainer.train_dataset) if hasattr(trainer.train_dataset, '__len__') else 'unknown',
                        'dataset_type': type(trainer.train_dataset).__name__
                    }
            except Exception as e:
                dataset_info = {'error': str(e)}
        
        # GPU 메모리 정보
        gpu_memory = {}
        if torch.cuda.is_available():
            try:
                gpu_memory = log_gpu_memory(logger, "OOM_ERROR")
                if gpu_memory:
                    gpu_memory['memory_summary'] = torch.cuda.memory_summary(device=None, abbreviated=False)
            except Exception as e:
                gpu_memory = {'error': str(e)}
        
        # 모든 정보 통합
        oom_info = {
            'timestamp': timestamp,
            'environment': env_info,
            'error': error_info,
            'training_state': training_state,
            'model_state': model_state,
            'batch_config': batch_config,
            'batch_info': batch_info,
            'dataset_info': dataset_info,
            'gpu_memory': gpu_memory
        }
        
        # JSON 파일로 저장
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(oom_info, f, indent=2, ensure_ascii=False, default=str)
        
        logger.error(f"💾 OOM 에러 정보가 저장되었습니다: {error_file}")
        return error_file
        
    except Exception as e:
        logger.error(f"❌ OOM 에러 정보 저장 실패: {e}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return None

def log_error_context(logger, error: Exception, context: str = ""):
    """Log detailed error context with system state"""
    logger.error(f"❌ Error in {context}: {str(error)}")
    logger.error(f"❌ Error type: {type(error).__name__}")
    
    # Log traceback
    logger.error(f"❌ Traceback:\n{traceback.format_exc()}")
    
    # Log GPU memory state
    if torch.cuda.is_available():
        memory_info = log_gpu_memory(logger, "ERROR")
        if memory_info:
            logger.error(f"❌ GPU Memory at error - Allocated: {memory_info['allocated']:.2f}GB, Reserved: {memory_info['reserved']:.2f}GB")
    
    # Log system state
    logger.error(f"❌ System state - CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logger.error(f"❌ Current device: {torch.cuda.current_device()}, Device name: {torch.cuda.get_device_name()}")


def handle_cuda_oom(e: torch.OutOfMemoryError, trainer, logger):
    """
    CUDA OOM (GPU 메모리 부족) 전용 처리 함수
    
    Args:
        e: torch.OutOfMemoryError 예외 객체
        trainer: Trainer 객체
        logger: Logger 객체
    """
    error_msg = str(e)
    logger.error("❌ CUDA Out of Memory Error 발생! (GPU 메모리 부족)")
    logger.error(f"   에러 메시지: {error_msg}")
    logger.error("   상세 정보를 수집합니다...")
    
    # 환경 정보 수집 및 로깅
    logger.error("🌍 Collecting environment information...")
    try:
        env_info = collect_environment_info()
        logger.error(f"❌ Environment at CUDA OOM:")
        logger.error(f"  - System: {env_info.get('system', {}).get('platform', 'N/A')}")
        logger.error(f"  - Python: {env_info.get('python', {}).get('version', 'N/A')[:50]}")
        logger.error(f"  - PyTorch: {env_info.get('pytorch', {}).get('version', 'N/A')}")
        logger.error(f"  - CUDA: {env_info.get('pytorch', {}).get('cuda_version', 'N/A')}")
        if 'gpu' in env_info and env_info['gpu']:
            for gpu_key, gpu_info in env_info['gpu'].items():
                if isinstance(gpu_info, dict) and 'name' in gpu_info:
                    logger.error(f"  - GPU {gpu_key}: {gpu_info.get('name', 'N/A')} ({gpu_info.get('total_memory_gb', 'N/A'):.2f}GB)")
    except Exception as env_e:
        logger.error(f"❌ Failed to collect environment info: {env_e}")
    
    # Log detailed memory state at OOM
    log_gpu_memory(logger, "OOM_ERROR")
    
    # Log training state at OOM
    if hasattr(trainer, 'state') and trainer.state is not None:
        state = trainer.state
        logger.error(f"❌ Training state at CUDA OOM:")
        logger.error(f"  - Global step: {state.global_step}")
        logger.error(f"  - Epoch: {state.epoch:.3f}")
        logger.error(f"  - Current loss: {getattr(state, 'log_history', [{}])[-1].get('train_loss', 'N/A')}")
    
    # Log model state
    logger.error(f"❌ Model state at CUDA OOM:")
    try:
        logger.error(f"  - Model device: {next(trainer.model.parameters()).device}")
        logger.error(f"  - Model dtype: {next(trainer.model.parameters()).dtype}")
        logger.error(f"  - Model requires_grad: {next(trainer.model.parameters()).requires_grad}")
    except Exception as model_e:
        logger.error(f"  - Could not get model state: {model_e}")
    
    # Log batch information
    logger.error(f"❌ Batch configuration at CUDA OOM:")
    try:
        # Trainer의 설정에서 배치 정보 가져오기
        batch_size = getattr(trainer, 'per_device_train_batch_size', None)
        grad_accum = getattr(trainer, 'gradient_accumulation_steps', None)
        num_devices = getattr(trainer.args, 'world_size', 1) if hasattr(trainer, 'args') else 1
        
        if batch_size is not None and grad_accum is not None:
            effective_batch = batch_size * grad_accum * num_devices
            logger.error(f"  - Per device batch size: {batch_size}")
            logger.error(f"  - Gradient accumulation: {grad_accum}")
            logger.error(f"  - Number of devices: {num_devices}")
            logger.error(f"  - Effective batch size: {effective_batch}")
        
        # DataLoader에서 실제 배치 크기 확인
        try:
            train_dataloader = trainer.get_train_dataloader()
            if hasattr(train_dataloader, 'batch_size'):
                logger.error(f"  - DataLoader batch_size: {train_dataloader.batch_size}")
            elif hasattr(train_dataloader, 'batch_sampler') and hasattr(train_dataloader.batch_sampler, 'batch_size'):
                logger.error(f"  - BatchSampler batch_size: {train_dataloader.batch_sampler.batch_size}")
        except Exception as dl_e:
            logger.error(f"  - Could not get DataLoader batch size: {dl_e}")
            
    except Exception as batch_e:
        logger.error(f"❌ Could not get batch info: {batch_e}")
    
    # 현재 배치의 데이터 샘플 정보 수집
    logger.error("📊 Collecting data sample information at CUDA OOM...")
    batch_info = None
    try:
        # 배치 추적 callback에서 저장된 정보 사용
        if hasattr(trainer, 'callback_handler') and trainer.callback_handler is not None:
            for callback in trainer.callback_handler.callbacks:
                if hasattr(callback, 'last_batch_info') and callback.last_batch_info is not None:
                    batch_info = callback.last_batch_info
                    logger.error(f"❌ Last processed batch information (step {getattr(callback, 'last_batch_step', 'unknown')}):")
                    break
        
        if batch_info:
            # 배치 크기 정보 (우선 표시)
            if 'actual_batch_size' in batch_info:
                logger.error(f"  - Actual batch size (from tensor): {batch_info['actual_batch_size']}")
            if 'per_device_batch_size' in batch_info:
                logger.error(f"  - Per device batch size: {batch_info['per_device_batch_size']}")
            if 'gradient_accumulation_steps' in batch_info:
                logger.error(f"  - Gradient accumulation steps: {batch_info['gradient_accumulation_steps']}")
            if 'num_devices' in batch_info:
                logger.error(f"  - Number of devices: {batch_info['num_devices']}")
            if 'effective_batch_size' in batch_info:
                logger.error(f"  - Effective batch size: {batch_info['effective_batch_size']}")
            if 'dataloader_batch_size' in batch_info:
                logger.error(f"  - DataLoader batch size: {batch_info['dataloader_batch_size']}")
            
            # Input IDs 정보
            if 'input_ids_shape' in batch_info:
                logger.error(f"  - Input IDs shape: {batch_info['input_ids_shape']}")
                logger.error(f"  - Input IDs total tokens: {batch_info.get('total_tokens', 'N/A')}")
                if 'sample_lengths' in batch_info:
                    logger.error(f"  - Sample lengths: {batch_info['sample_lengths']}")
                    logger.error(f"  - Max sample length: {batch_info.get('max_length', 'N/A')}")
                    logger.error(f"  - Min sample length: {batch_info.get('min_length', 'N/A')}")
                    logger.error(f"  - Avg sample length: {batch_info.get('avg_length', 'N/A'):.2f}")
            
            # Attention mask 정보
            if 'attention_mask_shape' in batch_info:
                logger.error(f"  - Attention mask shape: {batch_info['attention_mask_shape']}")
                logger.error(f"  - Attention mask total elements: {batch_info.get('attention_mask_total', 'N/A')}")
            
            # Pixel values (이미지) 정보
            if 'pixel_values_shape' in batch_info:
                logger.error(f"  - Pixel values shape: {batch_info['pixel_values_shape']}")
                logger.error(f"  - Pixel values dtype: {batch_info.get('pixel_values_dtype', 'N/A')}")
                logger.error(f"  - Pixel values memory (MB): {batch_info.get('pixel_values_memory_mb', 'N/A'):.2f}")
                logger.error(f"  - Number of images in batch: {batch_info.get('num_images', 'N/A')}")
            
            # Image grid 정보
            if 'image_grid_thw' in batch_info:
                logger.error(f"  - Image grid info: {batch_info['image_grid_thw']}")
            
            # Labels 정보
            if 'labels_shape' in batch_info:
                logger.error(f"  - Labels shape: {batch_info['labels_shape']}")
                logger.error(f"  - Non-ignore tokens: {batch_info.get('non_ignore_tokens', 'N/A')}")
                logger.error(f"  - Ignore tokens: {batch_info.get('ignore_tokens', 'N/A')}")
        
        # Trainer의 내부 상태에서 현재 배치 정보 확인 (fallback)
        if not batch_info:
            if hasattr(trainer, '_current_batch') and trainer._current_batch is not None:
                batch = trainer._current_batch
                logger.error(f"❌ Current batch information (from trainer._current_batch):")
                logger.error(f"  - Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'N/A'}")
                
                # Input IDs 정보
                if 'input_ids' in batch and torch.is_tensor(batch['input_ids']):
                    input_ids = batch['input_ids']
                    logger.error(f"  - Input IDs shape: {input_ids.shape}")
                    logger.error(f"  - Input IDs total tokens: {input_ids.numel()}")
                    
                    # 각 샘플의 길이
                    if len(input_ids.shape) > 1:
                        # processing_class에서 tokenizer 가져오기 (deprecated된 tokenizer 대신)
                        processing_class = getattr(trainer, 'processing_class', None)
                        pad_token_id = 0
                        if processing_class is not None:
                            # AutoProcessor인 경우 tokenizer 속성에 접근
                            tokenizer = getattr(processing_class, 'tokenizer', processing_class)
                            pad_token_id = getattr(tokenizer, 'pad_token_id', 0) or getattr(tokenizer, 'eos_token_id', 0)
                        sample_lengths = (input_ids != pad_token_id).sum(dim=1).cpu().tolist()
                        logger.error(f"  - Sample lengths: {sample_lengths}")
                        logger.error(f"  - Max sample length: {max(sample_lengths) if sample_lengths else 'N/A'}")
                        logger.error(f"  - Min sample length: {min(sample_lengths) if sample_lengths else 'N/A'}")
                        logger.error(f"  - Avg sample length: {sum(sample_lengths) / len(sample_lengths) if sample_lengths else 'N/A':.2f}")
                
                # Pixel values (이미지) 정보
                if 'pixel_values' in batch and torch.is_tensor(batch['pixel_values']):
                    pixel_values = batch['pixel_values']
                    logger.error(f"  - Pixel values shape: {pixel_values.shape}")
                    logger.error(f"  - Pixel values memory (MB): {pixel_values.numel() * pixel_values.element_size() / 1024 / 1024:.2f}")
                    logger.error(f"  - Number of images in batch: {pixel_values.shape[0] if len(pixel_values.shape) > 0 else 'N/A'}")
        
        # 최근 처리된 데이터셋 샘플 확인 (가능한 경우)
        if hasattr(trainer, 'train_dataset') and trainer.train_dataset is not None:
            try:
                state = trainer.state
                if state and hasattr(state, 'global_step'):
                    # 현재 step에서 처리 중인 샘플 인덱스 추정
                    dataset_size = len(trainer.train_dataset) if hasattr(trainer.train_dataset, '__len__') else 'unknown'
                    logger.error(f"  - Dataset size: {dataset_size}")
                    
                    # 샘플 몇 개 확인 (메모리 절약을 위해 최소한만)
                    if dataset_size != 'unknown' and dataset_size > 0:
                        sample_indices = []
                        if hasattr(trainer, 'per_device_train_batch_size'):
                            batch_size = trainer.per_device_train_batch_size
                            if hasattr(trainer, 'gradient_accumulation_steps'):
                                batch_size *= trainer.gradient_accumulation_steps
                            
                            # 현재 step에서 처리 중인 샘플 범위 추정
                            start_idx = (state.global_step * batch_size) % dataset_size
                            end_idx = min(start_idx + batch_size, dataset_size)
                            sample_indices = list(range(start_idx, end_idx))[:5]  # 최대 5개만
                        
                        if sample_indices:
                            logger.error(f"  - Estimated sample indices at CUDA OOM: {sample_indices}")
                            for idx in sample_indices[:3]:  # 최대 3개만 상세 확인
                                try:
                                    sample = trainer.train_dataset[idx]
                                    sample_info = {}
                                    
                                    # Messages 정보
                                    if 'messages' in sample:
                                        messages = sample['messages']
                                        if isinstance(messages, list):
                                            total_text_len = 0
                                            for msg in messages:
                                                if isinstance(msg, dict) and 'content' in msg:
                                                    content = msg['content']
                                                    if isinstance(content, list):
                                                        for item in content:
                                                            if isinstance(item, dict) and 'text' in item:
                                                                total_text_len += len(str(item['text']))
                                                    elif isinstance(content, str):
                                                        total_text_len += len(content)
                                            sample_info['messages_text_length'] = total_text_len
                                            sample_info['num_messages'] = len(messages)
                                    
                                    # Images 정보
                                    if 'images' in sample:
                                        images = sample['images']
                                        if isinstance(images, list):
                                            sample_info['num_images'] = len(images)
                                            if images:
                                                try:
                                                    from PIL import Image
                                                    if isinstance(images[0], Image.Image):
                                                        sample_info['image_sizes'] = [img.size for img in images[:3]]
                                                except:
                                                    pass
                                        elif images is not None:
                                            sample_info['has_image'] = True
                                    
                                    logger.error(f"    Sample {idx}: {sample_info}")
                                except Exception as sample_e:
                                    logger.error(f"    Sample {idx}: Could not inspect ({sample_e})")
            except Exception as dataset_e:
                logger.error(f"  - Could not inspect dataset: {dataset_e}")
    
    except Exception as data_collect_e:
        logger.error(f"❌ Failed to collect data sample information: {data_collect_e}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
    
    # 모든 정보를 JSON 파일로 저장
    logger.error("💾 Saving CUDA OOM error information to file...")
    try:
        output_dir = None
        if hasattr(trainer, 'args') and hasattr(trainer.args, 'output_dir'):
            output_dir = trainer.args.output_dir
        elif hasattr(trainer, 'training_args') and hasattr(trainer.training_args, 'output_dir'):
            output_dir = trainer.training_args.output_dir
        
        error_file = save_oom_error_info(logger, trainer, e, batch_info=batch_info, output_dir=output_dir)
        if error_file:
            logger.error(f"✅ CUDA OOM 에러 정보가 저장되었습니다: {error_file}")
            logger.error(f"   파일을 확인하여 상세한 환경 정보와 데이터 정보를 확인할 수 있습니다.")
    except Exception as save_e:
        logger.error(f"❌ CUDA OOM 에러 정보 저장 중 오류 발생: {save_e}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
    
    logger.error("❌ GPU 메모리 정리 중...")
    clear_gpu_memory()
    logger.error("❌ GPU 메모리 정리 완료.")
    logger.error("💡 CUDA OOM 해결 방법 제안:")
    logger.error("   1. per_device_train_batch_size를 더 줄이기 (현재: {})".format(
        trainer.per_device_train_batch_size if hasattr(trainer, 'per_device_train_batch_size') else 'N/A'
    ))
    logger.error("   2. gradient_accumulation_steps를 더 늘리기 (현재: {})".format(
        trainer.gradient_accumulation_steps if hasattr(trainer, 'gradient_accumulation_steps') else 'N/A'
    ))
    logger.error("   3. max_length를 줄이기 (현재: {})".format(
        trainer.args.max_length if hasattr(trainer.args, 'max_length') else 'N/A'
    ))
    logger.error("   4. 다른 프로세스가 GPU를 사용 중인지 확인 (nvidia-smi)")
    logger.error("   5. DeepSpeed ZeRO-3 CPU offload가 제대로 작동하는지 확인")
    logger.error("   6. 이미지가 포함된 샘플이 많으면 이미지 전용 데이터셋으로 분리 고려")
    logger.error("   7. 위의 데이터 샘플 정보를 확인하여 문제가 되는 샘플을 필터링하거나 처리 방식 변경 고려")
    logger.error("   8. 저장된 CUDA OOM 에러 정보 JSON 파일을 확인하여 상세한 환경 정보와 데이터 정보를 분석하세요")
    logger.error("   9. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 환경 변수 설정 고려")


def handle_ram_oom(e: MemoryError, trainer, logger):
    """
    로컬 RAM OOM (시스템 메모리 부족) 전용 처리 함수
    
    Args:
        e: MemoryError 예외 객체
        trainer: Trainer 객체
        logger: Logger 객체
    """
    error_msg = str(e)
    logger.error("❌ MemoryError 발생! (시스템 RAM 메모리 부족)")
    logger.error(f"   에러 메시지: {error_msg}")
    logger.error("   상세 정보를 수집합니다...")
    
    # 시스템 메모리 정보 수집
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.error(f"❌ 시스템 메모리 상태:")
        logger.error(f"  - Total RAM: {memory.total / 1024**3:.2f} GB")
        logger.error(f"  - Available RAM: {memory.available / 1024**3:.2f} GB")
        logger.error(f"  - Used RAM: {memory.used / 1024**3:.2f} GB")
        logger.error(f"  - RAM Usage: {memory.percent:.1f}%")
        
        # 프로세스별 메모리 사용량
        process = psutil.Process()
        process_memory = process.memory_info()
        logger.error(f"  - Current process RSS: {process_memory.rss / 1024**3:.2f} GB")
        logger.error(f"  - Current process VMS: {process_memory.vms / 1024**3:.2f} GB")
    except ImportError:
        logger.error("  - psutil이 설치되지 않아 상세 메모리 정보를 수집할 수 없습니다.")
    except Exception as mem_e:
        logger.error(f"  - 메모리 정보 수집 실패: {mem_e}")
    
    # 환경 정보 수집
    try:
        env_info = collect_environment_info()
        logger.error(f"❌ Environment at RAM OOM:")
        logger.error(f"  - System: {env_info.get('system', {}).get('platform', 'N/A')}")
        logger.error(f"  - CPU count: {env_info.get('system', {}).get('cpu_count', 'N/A')}")
    except Exception as env_e:
        logger.error(f"❌ Failed to collect environment info: {env_e}")
    
    # Training state 정보
    if hasattr(trainer, 'state') and trainer.state is not None:
        state = trainer.state
        logger.error(f"❌ Training state at RAM OOM:")
        logger.error(f"  - Global step: {state.global_step}")
        logger.error(f"  - Epoch: {state.epoch:.3f}")
    
    # 모든 정보를 JSON 파일로 저장
    logger.error("💾 Saving RAM OOM error information to file...")
    try:
        output_dir = None
        if hasattr(trainer, 'args') and hasattr(trainer.args, 'output_dir'):
            output_dir = trainer.args.output_dir
        elif hasattr(trainer, 'training_args') and hasattr(trainer.training_args, 'output_dir'):
            output_dir = trainer.training_args.output_dir
        
        error_file = save_oom_error_info(logger, trainer, e, batch_info=None, output_dir=output_dir)
        if error_file:
            logger.error(f"✅ RAM OOM 에러 정보가 저장되었습니다: {error_file}")
    except Exception as save_e:
        logger.error(f"❌ RAM OOM 에러 정보 저장 중 오류 발생: {save_e}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
    
    logger.error("💡 RAM OOM 해결 방법 제안:")
    logger.error("   1. 데이터셋을 스트리밍 모드로 변경 (streaming=True)")
    logger.error("   2. 데이터 로딩 방식을 변경하여 메모리 사용량 감소")
    logger.error("   3. DeepSpeed ZeRO-3 CPU offload 활성화 (이미 활성화되어 있다면 설정 확인)")
    logger.error("   4. 배치 크기를 줄이고 gradient accumulation을 늘리기")
    logger.error("   5. 데이터셋 전처리를 더 가볍게 만들기")
    logger.error("   6. 시스템의 다른 프로세스가 메모리를 많이 사용하는지 확인")
    logger.error("   7. 스왑 메모리(swap) 사용 고려 (성능 저하 가능)")
    logger.error("   8. 더 많은 RAM이 있는 시스템으로 이동 고려")


def handle_training_exception(e: Exception, trainer, logger, context: str = "training"):
    """
    학습 중 발생하는 일반 exception을 통합 처리하는 함수
    
    Args:
        e: Exception 객체
        trainer: Trainer 객체
        logger: Logger 객체
        context: 에러 발생 컨텍스트 (예: "training", "training_keyboard_interrupt", "training_runtime_error")
    """
    error_msg = str(e)
    error_type = type(e).__name__
    
    logger.error(f"❌ {error_type} during {context}: {error_msg}")
    log_error_context(logger, e, context)
    
    # 특정 에러 타입별 추가 처리
    if isinstance(e, KeyboardInterrupt):
        logger.error("❌ 학습이 사용자에 의해 중단되었습니다.")
    elif isinstance(e, RuntimeError):
        # CUBLAS 메모리 할당 실패 등 RuntimeError 처리
        if "CUBLAS_STATUS_ALLOC_FAILED" in error_msg or "cublasCreate" in error_msg:
            logger.error("❌ CUBLAS 메모리 할당 실패 - GPU 메모리 문제일 수 있습니다.")
            log_gpu_memory(logger, "CUBLAS_ERROR")
        else:
            logger.error(f"❌ RuntimeError: {error_msg}")
    else:
        logger.error(f"❌ Unexpected {error_type}: {error_msg}")


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


def ensure_router_parameters_trainable(model, logger, context: str = ""):
    """
    Router 파라미터를 trainable로 설정하는 통합 함수 (중복 코드 제거)
    
    Args:
        model: 모델 인스턴스 (DeepSpeed 래핑 가능)
        logger: 로거 인스턴스
        context: 컨텍스트 문자열 (로그용)
    
    Returns:
        tuple: (router_params_list, router_param_names_list, trainable_count)
    """
    from models.spectra_model import SPECTRARouter
    
    # 실제 모델 추출 (DeepSpeed 래핑 처리)
    actual_model = model
    if hasattr(model, 'module'):  # DeepSpeed 래핑
        actual_model = model.module
    
    router_params = []
    router_param_names = []
    seen_param_ids = set()
    
    for name, module in actual_model.named_modules():
        if isinstance(module, SPECTRARouter):
            if context:
                logger.debug(f"  [{context}] Found router module: {name}")
            
            # Router 모듈의 모든 파라미터
            for param_name, param in module.named_parameters(recurse=True):
                full_name = f"{name}.{param_name}"
                param_id = id(param)
                if param_id not in seen_param_ids:
                    router_params.append(param)
                    router_param_names.append(full_name)
                    seen_param_ids.add(param_id)
                if not param.requires_grad:
                    param.requires_grad_(True)
                    if context:
                        logger.debug(f"    [{context}] Set requires_grad=True: {full_name}")
            
            # Expression projector 파라미터
            if hasattr(module, 'expression_projector'):
                expr_proj = module.expression_projector
                for ep_param_name, ep_param in expr_proj.named_parameters(recurse=True):
                    full_name = f"{name}.expression_projector.{ep_param_name}"
                    ep_param_id = id(ep_param)
                    if ep_param_id not in seen_param_ids:
                        router_params.append(ep_param)
                        router_param_names.append(full_name)
                        seen_param_ids.add(ep_param_id)
                    if not ep_param.requires_grad:
                        ep_param.requires_grad_(True)
                        if context:
                            logger.debug(f"      [{context}] Set requires_grad=True: {full_name}")
            
            # Load balancer 파라미터
            if hasattr(module, 'load_balancer'):
                lb_module = module.load_balancer
                for lb_param_name, lb_param in lb_module.named_parameters(recurse=True):
                    full_name = f"{name}.load_balancer.{lb_param_name}"
                    lb_param_id = id(lb_param)
                    if lb_param_id not in seen_param_ids:
                        router_params.append(lb_param)
                        router_param_names.append(full_name)
                        seen_param_ids.add(lb_param_id)
                    if not lb_param.requires_grad:
                        lb_param.requires_grad_(True)
                        if context:
                            logger.debug(f"      [{context}] Set requires_grad=True: {full_name}")
    
    trainable_count = sum(1 for p in router_params if p.requires_grad)
    
    if context:
        logger.info(f"  [{context}] Router parameters: {trainable_count}/{len(router_params)} trainable")
    
    return router_params, router_param_names, trainable_count


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection with detailed logging"""
    import gc
    logger.info("🧹 Starting GPU memory cleanup...")
    
    # Log memory before cleanup
    memory_before = log_gpu_memory(logger, "BEFORE_CLEANUP")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("🧹 CUDA cache cleared and synchronized")
    
    # Force garbage collection
    collected = gc.collect()
    logger.debug(f"🧹 Garbage collection freed {collected} objects")
    
    # Log memory after cleanup
    memory_after = log_gpu_memory(logger, "AFTER_CLEANUP")
    
    if memory_before and memory_after:
        freed_allocated = memory_before['allocated'] - memory_after['allocated']
        freed_reserved = memory_before['reserved'] - memory_after['reserved']
        logger.info(f"🧹 Memory cleanup completed - Freed: {freed_allocated:.2f}GB allocated, {freed_reserved:.2f}GB reserved")
    else:
        logger.info("🧹 Memory cleanup completed")


def run_post_training_validation(
    model_path: str,
    training_config_path: str,
    output_dir: str,
    device: str = "cuda"
):
    """
    학습 종료 후 모든 validation 스크립트들을 실행하고 결과를 통합 레포트로 생성
    
    Args:
        model_path: 학습된 모델 경로
        training_config_path: 학습 설정 파일 경로
        output_dir: 결과 저장 디렉토리
        device: 사용할 디바이스
    """
    import subprocess
    import json
    import importlib.util
    from pathlib import Path
    
    logger.info("=" * 80)
    logger.info("🚀 Starting Post-Training Validation")
    logger.info("=" * 80)
    
    validation_results = {}
    validation_output_dir = Path(output_dir) / "validation_results"
    validation_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 실행 가능한 스크립트들
    executable_scripts = [
        {
            "name": "benchmark_evaluation",
            "script": "eval/run_benchmark_evaluation.py",
            "args": ["--model_path", model_path] + (["--training_config_path", training_config_path] if training_config_path else []),
            "description": "Benchmark evaluation (MMLU, HellaSwag, MME)",
            "required": False
        },
        {
            "name": "spectra_validation",
            "script": "eval/run_spectra_validation.py",
            "args": ["--task", "comparison", "--spectra_model", model_path, "--eval_dataset", "dummy", "--output_dir", str(validation_output_dir)],
            "description": "SPECTRA validation (requires eval_dataset)",
            "required": False
        },
        {
            "name": "expert_specialization",
            "script": "eval/analyze_expert_specialization.py",
            "args": ["--model_path", model_path, "--output_dir", str(validation_output_dir), "--dataset", "dummy"],
            "description": "Expert specialization analysis (requires dataset)",
            "required": False
        }
    ]
    
    # 모듈로만 사용되는 스크립트들 (Python으로 직접 실행)
    module_scripts = [
        {
            "name": "information_theoretic_analysis",
            "module": "eval.information_theoretic_analysis",
            "description": "Information-theoretic analysis",
            "required": False
        },
        {
            "name": "spectra_analysis",
            "module": "eval.spectra_analysis",
            "description": "SPECTRA MoE analysis",
            "required": False
        },
        {
            "name": "spectra_semantic_validation",
            "module": "eval.spectra_semantic_validation",
            "description": "SPECTRA semantic validation",
            "required": False
        }
    ]
    
    # 실행 가능한 스크립트들 실행
    for val_script in executable_scripts:
        script_name = val_script["name"]
        script_path = val_script["script"]
        script_args = val_script["args"]
        description = val_script["description"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {description}")
        logger.info(f"Script: {script_path}")
        logger.info(f"{'='*60}")
        
        try:
            # 스크립트가 실행 가능한지 확인
            script_full_path = Path(__file__).parent.parent / script_path
            if not script_full_path.exists():
                logger.warning(f"⚠️ Script not found: {script_full_path}")
                validation_results[script_name] = {
                    "status": "skipped",
                    "reason": "script_not_found",
                    "error": None
                }
                continue
            
            # 필수 인자가 누락된 경우 스킵
            if script_name == "spectra_validation" and "--eval_dataset" in script_args and script_args[script_args.index("--eval_dataset") + 1] == "dummy":
                logger.info(f"⚠️ {description} requires eval_dataset - skipping")
                validation_results[script_name] = {
                    "status": "skipped",
                    "reason": "missing_required_args",
                    "error": "eval_dataset required"
                }
                continue
            
            if script_name == "expert_specialization" and "--dataset" in script_args and script_args[script_args.index("--dataset") + 1] == "dummy":
                logger.info(f"⚠️ {description} requires dataset - skipping")
                validation_results[script_name] = {
                    "status": "skipped",
                    "reason": "missing_required_args",
                    "error": "dataset required"
                }
                continue
            
            # Python 스크립트 실행
            cmd = [
                sys.executable,
                str(script_full_path)
            ] + script_args
            
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1시간 타임아웃
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
                validation_results[script_name] = {
                    "status": "success",
                    "stdout": result.stdout[-1000:] if result.stdout else "",  # 마지막 1000자만 저장
                    "stderr": result.stderr[-1000:] if result.stderr else "",
                    "error": None
                }
            else:
                logger.error(f"❌ {description} failed with return code {result.returncode}")
                validation_results[script_name] = {
                    "status": "failed",
                    "returncode": result.returncode,
                    "stdout": result.stdout[-1000:] if result.stdout else "",
                    "stderr": result.stderr[-1000:] if result.stderr else "",
                    "error": result.stderr[-500:] if result.stderr else "Unknown error"
                }
        
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out after 1 hour")
            validation_results[script_name] = {
                "status": "timeout",
                "error": "Execution timed out after 1 hour"
            }
        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {e}")
            validation_results[script_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # 모듈로만 사용되는 스크립트들 실행
    for module_script in module_scripts:
        script_name = module_script["name"]
        module_name = module_script["module"]
        description = module_script["description"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {description}")
        logger.info(f"Module: {module_name}")
        logger.info(f"{'='*60}")
        
        try:
            # 모듈 동적 로드 및 실행
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                logger.warning(f"⚠️ Module not found: {module_name}")
                validation_results[script_name] = {
                    "status": "skipped",
                    "reason": "module_not_found",
                    "error": None
                }
                continue
            
            # 모듈 로드
            module = importlib.import_module(module_name)
            
            # 모듈에 main 함수가 있으면 실행
            if hasattr(module, 'main'):
                logger.info(f"Executing {module_name}.main()...")
                module.main()
                validation_results[script_name] = {
                    "status": "success",
                    "error": None
                }
                logger.info(f"✅ {description} completed successfully")
            else:
                # main 함수가 없으면 분석 클래스만 사용 가능하다는 메시지
                logger.info(f"⚠️ Module {module_name} has no main() function - analysis classes available for import")
                validation_results[script_name] = {
                    "status": "skipped",
                    "reason": "no_main_function",
                    "error": "Module provides analysis classes but no executable main function"
                }
        
        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {e}")
            validation_results[script_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # 결과 통합 레포트 생성
    report_path = validation_output_dir / "validation_report.json"
    report_summary_path = validation_output_dir / "validation_summary.txt"
    
    # JSON 레포트 저장
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_path": model_path,
            "training_config_path": training_config_path,
            "validation_timestamp": datetime.now().isoformat(),
            "results": validation_results
        }, f, indent=2, ensure_ascii=False)
    
    # 텍스트 요약 레포트 생성
    with open(report_summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Post-Training Validation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Training Config: {training_config_path}\n")
        f.write(f"Validation Timestamp: {datetime.now().isoformat()}\n\n")
        
        f.write("Validation Results Summary:\n")
        f.write("-" * 80 + "\n")
        
        success_count = sum(1 for r in validation_results.values() if r.get("status") == "success")
        failed_count = sum(1 for r in validation_results.values() if r.get("status") in ["failed", "error", "timeout"])
        skipped_count = sum(1 for r in validation_results.values() if r.get("status") == "skipped")
        
        f.write(f"Total: {len(validation_results)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Failed: {failed_count}\n")
        f.write(f"Skipped: {skipped_count}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 80 + "\n")
        for script_name, result in validation_results.items():
            status = result.get("status", "unknown")
            status_symbol = "✅" if status == "success" else "❌" if status in ["failed", "error", "timeout"] else "⚠️"
            f.write(f"{status_symbol} {script_name}: {status}\n")
            if result.get("error"):
                f.write(f"   Error: {result['error'][:200]}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Full results saved to: validation_report.json\n")
        f.write("=" * 80 + "\n")
    
    # 통계 계산
    success_count = sum(1 for r in validation_results.values() if r.get("status") == "success")
    failed_count = sum(1 for r in validation_results.values() if r.get("status") in ["failed", "error", "timeout"])
    skipped_count = sum(1 for r in validation_results.values() if r.get("status") == "skipped")
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 Validation Report Summary")
    logger.info("=" * 80)
    logger.info(f"Total validations: {len(validation_results)}")
    logger.info(f"✅ Success: {success_count}")
    logger.info(f"❌ Failed: {failed_count}")
    logger.info(f"⚠️ Skipped: {skipped_count}")
    logger.info(f"\n📄 Full report: {report_path}")
    logger.info(f"📄 Summary report: {report_summary_path}")
    logger.info("=" * 80)
    
    # 레포트 파일 내용 출력
    try:
        with open(report_summary_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        logger.info("\n" + report_content)
    except Exception as e:
        logger.warning(f"⚠️ Could not read report summary: {e}")
    
    return validation_results


def eval_with_memory_optimization(trainer, original_eval_fn, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    """Memory-optimized evaluation function with detailed logging"""
    logger.info("🔧 Starting memory-optimized evaluation...")
    
    # Log evaluation context
    if hasattr(trainer, 'state') and trainer.state is not None:
        logger.info(f"🔧 Evaluation context - Step: {trainer.state.global_step}, Epoch: {trainer.state.epoch:.3f}")
    
    # Log memory before evaluation
    memory_before = log_gpu_memory(logger, "BEFORE_EVAL")
    
    # GPU 메모리 정리
    clear_gpu_memory()
    
    # 모델을 eval 모드로 설정하고 메모리 최적화
    logger.debug("🔧 Setting model to eval mode...")
    trainer.model.eval()
    
    # eval 시에는 gradient checkpointing 비활성화
    original_gc = trainer.args.gradient_checkpointing
    trainer.args.gradient_checkpointing = False
    logger.debug(f"🔧 Disabled gradient checkpointing for evaluation (was: {original_gc})")
    
    try:
        logger.info("🔧 Starting evaluation with torch.no_grad()...")
        start_time = time.time()
        
        with torch.no_grad():
            # 원래 evaluate 함수 호출 (무한 재귀 방지)
            eval_results = original_eval_fn(
                eval_dataset=eval_dataset, 
                ignore_keys=ignore_keys, 
                metric_key_prefix=metric_key_prefix
            )
        
        eval_time = time.time() - start_time
        logger.info(f"🔧 Evaluation completed in {eval_time:.2f} seconds")
        
        # Log evaluation results
        if eval_results:
            logger.info(f"🔧 Evaluation results: {eval_results}")
        
        # Log memory after evaluation
        memory_after = log_gpu_memory(logger, "AFTER_EVAL")
        
        # 결과 반환
        return eval_results
        
    except Exception as e:
        logger.error(f"❌ Error during evaluation: {str(e)}")
        log_error_context(logger, e, "memory_optimized_evaluation")
        raise e
        
    finally:
        # DeepSpeed 타이머 정리 (eval 후 train 모드로 돌아갈 때 타이머 충돌 방지)
        # 이는 "fwd_microstep timer has already been started" 에러를 방지하기 위함
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            try:
                # DeepSpeed 엔진의 타이머 정리
                deepspeed_engine = trainer.deepspeed
                
                # 방법 1: engine_timers를 통한 타이머 정리
                if hasattr(deepspeed_engine, 'engine_timers'):
                    engine_timers = deepspeed_engine.engine_timers
                    
                    # forward_timers 정리
                    if hasattr(engine_timers, 'forward_timers'):
                        forward_timers = engine_timers.forward_timers
                        if isinstance(forward_timers, dict):
                            for timer_name, timer in forward_timers.items():
                                if timer is not None and hasattr(timer, 'started_') and getattr(timer, 'started_', False):
                                    try:
                                        if hasattr(timer, 'stop'):
                                            timer.stop()
                                            logger.debug(f"🔧 Stopped DeepSpeed forward timer: {timer_name}")
                                    except Exception as e:
                                        logger.debug(f"🔧 Timer {timer_name} stop failed (may already be stopped): {e}")
                
                # 방법 2: timers 객체를 통한 타이머 정리 (다른 DeepSpeed 버전 호환)
                if hasattr(deepspeed_engine, 'timers'):
                    timers = deepspeed_engine.timers
                    if hasattr(timers, '_timers'):
                        for timer_name, timer in timers._timers.items():
                            if timer is not None and hasattr(timer, 'started_') and getattr(timer, 'started_', False):
                                try:
                                    if hasattr(timer, 'stop'):
                                        timer.stop()
                                        logger.debug(f"🔧 Stopped DeepSpeed timer: {timer_name}")
                                except Exception as e:
                                    logger.debug(f"🔧 Timer {timer_name} stop failed (may already be stopped): {e}")
                
                # 방법 3: _stop_timers 메서드가 있다면 사용
                if hasattr(deepspeed_engine, '_stop_timers'):
                    try:
                        deepspeed_engine._stop_timers(deepspeed_engine.engine_timers.forward_timers)
                        logger.debug("🔧 Stopped DeepSpeed forward timers via _stop_timers")
                    except Exception as e:
                        logger.debug(f"🔧 _stop_timers failed: {e}")
                
                logger.debug("🔧 DeepSpeed timers reset completed")
            except Exception as e:
                logger.warning(f"⚠️ Failed to reset DeepSpeed timers: {e}")
                # 타이머 정리 실패해도 학습은 계속 진행
        
        # 모델을 train 모드로 전환 (eval 후 필수)
        # 타이머 정리 후에 train 모드로 전환하여 타이머 충돌 방지
        logger.debug("🔧 Setting model back to train mode...")
        trainer.model.train()
        
        # 원래 설정 복원
        logger.debug(f"🔧 Restoring gradient checkpointing to: {original_gc}")
        trainer.args.gradient_checkpointing = original_gc
        clear_gpu_memory()


def setup_model_and_tokenizer(model_config: Dict[str, Any]):
    """모델과 토크나이저 설정. modules_to_save_list를 반환"""
    """Setup SPECTRA model and tokenizer with detailed logging"""
    logger.info("🚀 Starting model and tokenizer setup...")
    
    # NOTE: Delay DeepSpeed env setup until AFTER model load to avoid HF ZeRO-3 init slow path
    logger.info("🔧 Setting up DeepSpeed environment...")
    setup_deepspeed_environment()
    
    # Check if initializing from scratch
    initialize_from_scratch = model_config.get("initialize_from_scratch", False)
    
    # Load tokenizer - 안정적인 로딩 로직
    if initialize_from_scratch:
        # For from-scratch initialization, use a default tokenizer path or create a minimal one
        tokenizer_path = model_config.get("tokenizer_name_or_path") or "google/gemma-2-2b-it"
        logger.info(f"🔤 Initializing from scratch - using default tokenizer: {tokenizer_path}")
    else:
        tokenizer_path = model_config.get("tokenizer_name_or_path") or model_config["model_name_or_path"]
        logger.info(f"🔤 Loading tokenizer from: {tokenizer_path}")
    
    tokenizer = None
    try:
        logger.debug("  - Attempting AutoProcessor...")
        tokenizer = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_config["trust_remote_code"]
        )
        logger.info("  ✅ AutoProcessor loaded successfully")
    except Exception as e:
        logger.warning(f"  ❌ AutoProcessor failed: {e}")
        try:
            logger.debug("  - Attempting AutoTokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=model_config["trust_remote_code"]
            )
            logger.info("  ✅ AutoTokenizer loaded successfully")
        except Exception as e2:
            logger.error(f"  ❌ AutoTokenizer also failed: {e2}")
            log_error_context(logger, e2, "tokenizer_loading")
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
    attn_from_cfg = (model_config.get("spectra_params") or {}).get("attn_implementation")
    if attn_from_cfg in {"eager", "sdpa", "flash_attention_2"}:
        attn_implementation = attn_from_cfg
    else:
        attn_implementation = "eager"

    # SPECTRA configuration parameters from config file
    spectra_params = model_config.get("spectra_params", {})
    
    # Load and configure SPECTRA model configuration
    if initialize_from_scratch:
        print("Initializing model from scratch...")
        # Load base model config from tokenizer path to get actual architecture
        base_model_path = model_config.get("tokenizer_name_or_path") or model_config.get("model_name_or_path")
        if base_model_path:
            logger.info(f"📐 Loading base model architecture from: {base_model_path}")
            try:
                base_config = AutoConfig.from_pretrained(
                    base_model_path,
                    trust_remote_code=model_config["trust_remote_code"]
                )
                base_model_config = base_config.to_dict()
                
                # Handle different model config structures
                if 'text_config' in base_model_config:
                    text_config = base_model_config['text_config']
                else:
                    text_config = base_model_config
                
                # Get architecture parameters from base model config
                hidden_size = text_config.get('hidden_size')
                num_hidden_layers = text_config.get('num_hidden_layers')
                num_attention_heads = text_config.get('num_attention_heads')
                num_key_value_heads = text_config.get('num_key_value_heads', num_attention_heads)
                intermediate_size = text_config.get('intermediate_size')
                vocab_size = text_config.get('vocab_size')
                max_position_embeddings = text_config.get('max_position_embeddings', 2048)
                
                logger.info(f"  ✅ Loaded architecture: layers={num_hidden_layers}, hidden_size={hidden_size}, heads={num_attention_heads}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load base model config: {e}")
                logger.warning(f"  ⚠️ Using fallback defaults (this should not happen!)")
                # Fallback to defaults only if config loading fails
                hidden_size = 512
                num_hidden_layers = 4
                num_attention_heads = 4
                num_key_value_heads = 2
                intermediate_size = 2048
                vocab_size = 256000
                max_position_embeddings = 2048
        else:
            raise ValueError("Cannot initialize from scratch without tokenizer_name_or_path or model_name_or_path")
        
        # Create a minimal config from scratch
        from transformers.models.siglip import SiglipVisionConfig
        
        # Create text config from scratch
        text_config_dict = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": hidden_size // num_attention_heads,
            "n_shared_experts": spectra_params.get("n_shared_experts", 1),
            "n_routed_experts": spectra_params.get("n_routed_experts", 8),
            "n_group": spectra_params.get("n_group", 2),
            "topk_group": spectra_params.get("topk_group", 2),
            "num_experts_per_tok": spectra_params.get("num_experts_per_tok", 2),
            "first_k_dense_replace": spectra_params.get("first_k_dense_replace", 0),
            "router_aux_loss_coef": 0.0,  # [Minimalist] Sinkhorn이 구조적으로 처리
            "router_jitter_noise": spectra_params.get("router_jitter_noise", 0.01),
            "input_jitter_noise": spectra_params.get("input_jitter_noise", 0.0),
            "router_dim": spectra_params.get("router_dim", 128),
            "neftune_noise_alpha": spectra_params.get("neftune_noise_alpha", 0.0),
            "model_type": "spectra_text",
            "rope_scaling": spectra_params.get("rope_scaling", {"rope_type": "default", "factor": 1.0}),
            "attn_implementation": attn_implementation,
            "max_position_embeddings": max_position_embeddings,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
        }
        
        # Create vision config (minimal for from-scratch)
        vision_config = SiglipVisionConfig(
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
        )
        
        # Create SPECTRA config from scratch
        config = SPECTRAConfig(
            text_config=text_config_dict,
            vision_config=vision_config,
            boi_token_index=255999,
            eoi_token_index=256000,
            image_token_index=262144,
            initializer_range=0.02,
            attn_implementation=attn_implementation,
        )
    else:
        print("Loading base model configuration...")
        base_config = AutoConfig.from_pretrained(
            model_config["model_name_or_path"],
            trust_remote_code=model_config["trust_remote_code"]
        )
        
        # Convert to dict and update with SPECTRA parameters
        base_model_config = base_config.to_dict()
        
        # Handle different model config structures (Gemma vs others)
        if 'text_config' in base_model_config:
            # Multi-modal model with text_config
            text_config = base_model_config['text_config']
            num_attention_heads = text_config['num_attention_heads']
        else:
            # Direct text model config
            text_config = base_model_config
            num_attention_heads = base_model_config['num_attention_heads']
        
        spectra_config = {
            "n_shared_experts": spectra_params["n_shared_experts"],
            "n_routed_experts": spectra_params["n_routed_experts"],
            "n_group": spectra_params["n_group"],
            "topk_group": spectra_params["topk_group"],
            "num_experts_per_tok": spectra_params["num_experts_per_tok"],
            "first_k_dense_replace": spectra_params["first_k_dense_replace"],
            "router_dim": spectra_params.get("router_dim", 128),
            "router_aux_loss_coef": 0.0,  # [Minimalist] Sinkhorn이 구조적으로 처리
            "router_jitter_noise": spectra_params["router_jitter_noise"],
            "input_jitter_noise": spectra_params["input_jitter_noise"],
            "router_z_loss_coef": 0.0,  # [Minimalist] 불필요
            "router_entropy_coef": spectra_params.get("router_entropy_coef", 0.1),  # [Sharpening] 확실한 선택 유도
            "usage_uniformity_coef": 0.0,  # [Minimalist] Sinkhorn이 구조적으로 처리
            "ema_alpha": spectra_params.get("ema_alpha", 0.95),
            "balancing_strength": spectra_params.get("balancing_strength", 1e-2),
            "neftune_noise_alpha": spectra_params.get("neftune_noise_alpha", 0.0),
            "no_rope_layer_interval": spectra_params.get("no_rope_layer_interval", 0),
            "use_sliding_window": spectra_params.get("use_sliding_window", False),
            "model_type": "spectra_text",
            "rope_scaling": {
                "rope_type": spectra_params["rope_scaling"]["rope_type"],
                "factor": spectra_params["rope_scaling"]["factor"]
            },
            "use_bfloat16": True,
            "attn_implementation": attn_implementation,
            # Expression Projector parameters
            "expression_ortho_strength": spectra_params.get("expression_ortho_strength", 0.1),
            "expression_init_scale": spectra_params.get("expression_init_scale", 0.1),
            # Router parameters
            "router_gru_num_layers": spectra_params.get("router_gru_num_layers", 3),
            "router_logit_scale_init": spectra_params.get("router_logit_scale_init", 2.302585092994046),
            "router_logit_scale_max": spectra_params.get("router_logit_scale_max", 100.0),
            "router_layernorm_eps": spectra_params.get("router_layernorm_eps", 1e-5),
            # Sinkhorn parameters
            "sinkhorn_ortho_penalty_alpha": spectra_params.get("sinkhorn_ortho_penalty_alpha", 0.5),
            "spechorn_sinkhorn_eps": spectra_params.get("spechorn_sinkhorn_eps", 0.05),
            "spechorn_sinkhorn_iter": spectra_params.get("spechorn_sinkhorn_iter", 4),
            "spechorn_bias_scale": spectra_params.get("spechorn_bias_scale", 8.0),
            "spechorn_cap_penalty_scale": spectra_params.get("spechorn_cap_penalty_scale", 15.0),
            "spechorn_ortho_scale": spectra_params.get("spechorn_ortho_scale", 0.4),
            "spechorn_spec_update_every": spectra_params.get("spechorn_spec_update_every", 16),
            "spechorn_target_cv_min": spectra_params.get("spechorn_target_cv_min", 0.03),
            "spechorn_target_cv_max": spectra_params.get("spechorn_target_cv_max", 0.08),
            "spechorn_cap_penalty_min": spectra_params.get("spechorn_cap_penalty_min", 5.0),
            "spechorn_cap_penalty_max": spectra_params.get("spechorn_cap_penalty_max", 30.0),
            "spechorn_cap_penalty_step": spectra_params.get("spechorn_cap_penalty_step", 1.0),
            "spechorn_bias_scale_min": spectra_params.get("spechorn_bias_scale_min", 4.0),
            "spechorn_bias_scale_max": spectra_params.get("spechorn_bias_scale_max", 12.0),
            "spechorn_ortho_scale_min": spectra_params.get("spechorn_ortho_scale_min", 0.1),
            "spechorn_ortho_scale_max": spectra_params.get("spechorn_ortho_scale_max", 0.6),
            # Loss coefficients
            "speciality_loss_coef": spectra_params.get("speciality_loss_coef", 0.0005),
            "contrastive_loss_coef": spectra_params.get("contrastive_loss_coef", 0.0005),
            "expression_reg_loss_coef": spectra_params.get("expression_reg_loss_coef", 1.0),
            "cosine_similarities_loss_coef": spectra_params.get("cosine_similarities_loss_coef", 0.001),
            "ortho_loss_coef": spectra_params.get("ortho_loss_coef", 0.003),
            "sinkhorn_distillation_coef": spectra_params.get("sinkhorn_distillation_coef", 0.05),
            "sinkhorn_teacher_epsilon": spectra_params.get("sinkhorn_teacher_epsilon", 0.05),
            # Load balancing parameters
            "lb_bias_to_hn": spectra_params.get("lb_bias_to_hn", True),
            "lb_bias_scale": spectra_params.get("lb_bias_scale", 0.1),
            "bias_lr": spectra_params.get("bias_lr", 1e-3),
            "bias_decay": spectra_params.get("bias_decay", 0.95),
            "bias_max": spectra_params.get("bias_max", 3.0),
            "lb_bias_coef": spectra_params.get("lb_bias_coef", 1.2),
            "gslb_coef": spectra_params.get("gslb_coef", 5e-2),
            "lb_l2_coef": spectra_params.get("lb_l2_coef", 5e-3),
            "lb_cv_coef": spectra_params.get("lb_cv_coef", 2e-2),
            "lb_entropy_floor_coef": spectra_params.get("lb_entropy_floor_coef", 2e-4),
            "lb_topk_l2_coef": spectra_params.get("lb_topk_l2_coef", 1.0),
            "lb_topk_cv_coef": spectra_params.get("lb_topk_cv_coef", 0.9),
            "routed_scaling_factor": spectra_params.get("routed_scaling_factor", 1.0),
        }
        base_model_config["text_config"].update(spectra_config)
        # Create SPECTRA configuration
        config = SPECTRAConfig(
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
    print("SPECTRA configuration created successfully")
    if initialize_from_scratch:
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Num layers: {num_hidden_layers}")
        print(f"  - Num attention heads: {num_attention_heads}")
    print(f"  - Shared experts: {config.text_config.n_shared_experts}")
    print(f"  - Routed experts: {config.text_config.n_routed_experts}")
    print(f"  - Expert groups: {config.text_config.n_group}")
    print(f"  - Top-k per group: {config.text_config.topk_group}")
    print(f"  - Experts per token: {config.text_config.num_experts_per_tok}")
    
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
    
    # Load SPECTRA model with the configured parameters
    logger.info("🤖 Loading SPECTRA model...")
    if initialize_from_scratch:
        logger.info(f"🤖 Initializing from scratch (no pretrained model)")
    else:
        logger.info(f"🤖 Model path: {model_config.get('model_name_or_path', 'N/A')}")
    logger.info(f"🤖 Device map: {device_map}")
    logger.info(f"🤖 Attention implementation: {attn_implementation}")
    
    # Log memory before model loading
    memory_before = log_gpu_memory(logger, "BEFORE_MODEL_LOAD")
    
    try:
        start_time = time.time()
        if initialize_from_scratch:
            # Initialize model from scratch with random weights
            logger.info("🔨 Initializing model from scratch (random weights)...")
            model = SPECTRAForConditionalGeneration(config=config)
            # Defer device/dtype placement to PEFT/Trainer to avoid multi-GPU OOM
        else:
            model = SPECTRAForConditionalGeneration.from_pretrained(
                model_config["model_name_or_path"],
                config=config,
                torch_dtype=torch.bfloat16, # Using bfloat16
                trust_remote_code=model_config["trust_remote_code"],
                device_map=device_map,
                low_cpu_mem_usage=True,
                offload_state_dict=True,
                use_cache=True,
                gradient_checkpointing=True,
                # load_in_4bit=True,
                attn_implementation=attn_implementation
            )
        load_time = time.time() - start_time
        logger.info(f"✅ SPECTRA model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"  - Attn implementation: {attn_implementation}")
        
        # Log memory after model loading
        memory_after = log_gpu_memory(logger, "AFTER_MODEL_LOAD")
        
        total_params = model.num_parameters()
        logger.info(f"  - Total parameters: {format_parameters(total_params)}")
        logger.info(f"  - Model Memory consumption: {memory_before['allocated']:.2f}GB → {memory_after['allocated']:.2f}GB")
        # Log model device placement
        if hasattr(model, 'device'):
            logger.info(f"  - Model device: {model.device}")
        elif hasattr(model, 'hf_device_map'):
            logger.info(f"  - Model device map: {model.hf_device_map}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load SPECTRA model: {str(e)}")
        log_error_context(logger, e, "model_loading")
        raise e

    modules_to_save_list = None
    # Setup LoRA if requested
    if model_config["use_lora"]:
        logger.info("🔍 Enabling LoRA for router components (router/balancer/projector)")
        from models.spectra_model import SPECTRARouter

        # LoRA only on supported Linear submodules (avoid wrapping custom modules)
        lora_target_modules = [
            # experts FFN
            "gate_proj",
            "up_proj",
            "down_proj",
            # ManualGRUCell linears (router balancer)
            "weight_ih_gates",
            "weight_hh_gates",
            "weight_ih_cand",
            "weight_hh_cand",
            # Dual solver projections
            "u_proj",
            "v_proj",
            # Bias predictor linears (unwrapped)
            "bias_pred_fc1",
            "bias_pred_fc2",
            # expression projector linear head
            "linear_projection",
        ]

        lora_config = LoraConfig(
            r=model_config["lora_r"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            target_modules=lora_target_modules,
            modules_to_save=None,
            ensure_weight_tying=True,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            fan_in_fan_out=False,
        )

        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()

        # Keep router params trainable with a single pass
        ensure_router_parameters_trainable(model, logger, context="PEFT_setup")
        logger.info("✅ LoRA ready.")
        
    # CRITICAL: LoRA 비활성화 시에도 router 파라미터를 항상 학습 가능하도록 설정
    # DeepSpeed ZeRO-3 + CPU offload 환경에서도 router가 학습되도록 보장
    logger.info("=" * 80)
    logger.info("🔧 Ensuring router parameters are trainable (regardless of LoRA setting)...")
    logger.info("=" * 80)
    
    # 통합 함수 사용 (중복 코드 제거)
    router_params, router_param_names, trainable_count = ensure_router_parameters_trainable(
        model, logger, context="LoRA_setup"
    )
    
    if router_params:
        logger.info(f"✅ Router parameters trainable: {trainable_count}/{len(router_params)}")
    else:
        logger.warning("⚠️ No router modules found in model!")
    
    logger.info("=" * 80)
        
    # modules_to_save_list 반환 (Trainer 생성 후 optimizer에 추가하기 위해)
    modules_to_save_list_to_return = modules_to_save_list if 'modules_to_save_list' in locals() else None
    return model, tokenizer, modules_to_save_list_to_return


def setup_dataset(data_config: Dict[str, Any], tokenizer):
    """Setup training dataset"""    
    use_multi_domain = data_config.get("use_multi_domain", False)
    dataset_name = data_config.get("dataset_name", "HuggingFaceTB/smoltalk")
    max_samples = data_config.get("max_samples", 100000)
    max_samples_per_domain = data_config.get("max_samples_per_domain", None)  # multi-domain용
    max_seq_length = data_config.get("max_seq_length", 131072) or 131072
    test_size = data_config.get("test_size", 0.1)
    use_streaming = data_config.get("streaming", False)
    max_workers = data_config.get("max_workers", 4)  # multi-domain 병렬 처리용
    
    print(f"Loading dataset: {dataset_name}")
    print(f"  - Use multi-domain: {use_multi_domain}")
    print(f"  - Max samples: {max_samples}")
    if max_samples_per_domain:
        print(f"  - Max samples per domain: {max_samples_per_domain}")
    print(f"  - Max sequence length: {max_seq_length}")
    print(f"  - Test size: {test_size}")
    print(f"  - Streaming: {use_streaming}")
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
    
    try:
        # Multi-domain 데이터셋 사용
        if use_multi_domain:
            print(f"🔄 Multi-domain 데이터셋 로더 사용")
            # domain_configs가 지정되어 있으면 사용, 없으면 모든 도메인 사용
            domain_configs = data_config.get("domain_configs", None)
            
            if max_samples_per_domain is None:
                # max_samples_per_domain이 없으면 max_samples를 도메인 수로 나눔
                if domain_configs:
                    num_domains = len(domain_configs)
                else:
                    from data.multi_domain_sft_dataset import DOMAIN_DATASETS
                    num_domains = len(DOMAIN_DATASETS)
                max_samples_per_domain = max(1, max_samples // num_domains)
                print(f"  - 자동 계산된 도메인당 샘플 수: {max_samples_per_domain}")
            
            dataset = get_multi_domain_sft_dataset(
                domain_configs=domain_configs,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                max_samples_per_domain=max_samples_per_domain,
                test_size=test_size,
                use_streaming=use_streaming,
                max_workers=max_workers
            )
            # Multi-domain용 collate 함수 사용 (allow_text_only=True)
            # processor 생성 (AutoProcessor 또는 tokenizer)
            # tokenizer가 이미 AutoProcessor인 경우 그대로 사용
            if hasattr(tokenizer, 'tokenizer'):
                # AutoProcessor인 경우
                processor = tokenizer
            else:
                # AutoTokenizer인 경우, tokenizer를 processor로 사용
                # (multi_domain_collate_fn이 tokenizer도 처리할 수 있음)
                processor = tokenizer
            
            collate_fn = create_multi_domain_collate_fn(processor, max_length=max_seq_length, allow_text_only=True)
        
        # 간단한 데이터셋 로더 사용
        elif "smoltalk" in dataset_name.lower() or "orca" in dataset_name.lower() or "llava" in dataset_name.lower():
            print(f"일반 데이터셋 로더 시도: {dataset_name}")
            dataset = get_simple_sft_dataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                max_samples=max_samples,
                test_size=test_size,
                use_streaming=use_streaming
            )
            # 이미지 중첩 리스트 문제 해결을 위한 커스텀 data collator 사용
            collate_fn = create_simple_collate_fn(tokenizer, max_length=max_seq_length)
        else:
            # open_m_3 데이터셋 로더 시도
            dataset = get_dataset(
                tokenizer=tokenizer,
                dataset_name=data_config["dataset_name"],
                max_length=data_config["max_seq_length"],
                test_size=data_config["test_size"],
                text_only=data_config.get("text_only", False),
                streaming=data_config["streaming"]
            )
            collate_fn = create_multimodal_collate_fn(tokenizer)
        
        print(f"Dataset loaded:")
        for split, data in dataset.items():
            try:
                if use_streaming and hasattr(data, 'info') and hasattr(data.info, 'dataset_size'):
                    size = data.info.dataset_size
                else:
                    size = len(data) if hasattr(data, '__len__') else "unknown"
                print(f"  {split}: {size} examples")
            except Exception as e:
                print(f"  {split}: size unknown ({e})")
        
        # 빈 데이터셋 체크
        train_dataset = dataset.get("train", None)
        if train_dataset is None:
            raise ValueError("훈련 데이터셋이 없습니다!")
        
        if use_streaming:
            if hasattr(train_dataset, 'info') and hasattr(train_dataset.info, 'dataset_size'):
                if train_dataset.info.dataset_size == 0:
                    raise ValueError("훈련 데이터셋이 비어있습니다!")
        else:
            if hasattr(train_dataset, '__len__') and len(train_dataset) == 0:
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
    
    # Force save_safetensors=False to handle shared router parameters in MoE
    # This avoids RuntimeError when saving models with global_router shared across layers
    training_config["save_safetensors"] = False
    
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
        # NOTE: Router learning issues with ZeRO-3 + CPU offload
        # If router weights are not learning, try:
        # 1. Reduce ZeRO stage from 3 to 2 (change "stage": 3 to "stage": 2)
        # 2. Disable CPU offload (set "device": "none" for offload_optimizer and offload_param)
        # 3. These changes help isolate whether the issue is due to parameter partitioning or offloading
        try:
            with open(ds_cfg_path_abs, "r") as f:
                ds_cfg = json.load(f)
            zero = ds_cfg.get("zero_optimization", {})
            off_opt = (zero.get("offload_optimizer") or {}).get("device", "none").lower()
            off_param = (zero.get("offload_param") or {}).get("device", "none").lower()
            zero_stage = zero.get("stage", 0)
            print(f"DeepSpeed zero stage: {zero_stage}")
            print(f"DeepSpeed offload_optimizer.device: {off_opt}")
            print(f"DeepSpeed offload_param.device: {off_param}")
            
            # Warn if using ZeRO-3 with CPU offload (may cause router learning issues)
            if zero_stage == 3 and (off_opt != "none" or off_param != "none"):
                print("⚠️ WARNING: Using ZeRO-3 with CPU offload may cause router learning issues!")
                print("   If router weights are not learning, try:")
                print("   1. Reduce ZeRO stage to 2 (change 'stage': 3 to 'stage': 2)")
                print("   2. Disable CPU offload (set 'device': 'none' for offload_optimizer and offload_param)")
            # assert off_opt in {"none", None, ""} and off_param in {"none", None, ""}, (
            #     "DeepSpeed CPU offload detected in config but must be disabled (device='none')."
            # )
            # Workaround: ZeRO-3 + gradient checkpointing can trigger duplicate ds_id assertion
            try:
                zero_stage = int(zero.get("stage", 0) or 0)
            except Exception:
                zero_stage = 0
            # if zero_stage == 3 and getattr(training_args, "gradient_checkpointing", False):
            #     print("⚠️ Detected ZeRO-3 with gradient checkpointing enabled. Disabling to avoid ds_id assertion.")
            #     training_args.gradient_checkpointing = False
        except Exception as e:
            print(f"⚠️ DeepSpeed config validation warning: {e}")
    
    return training_args


def main(
    model_config: Dict[str, Any], 
    data_config: Dict[str, Any], 
    training_config: Dict[str, Any],
    config_path: str = None
):
    register_custom_optimizers()
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    setup_result = setup_model_and_tokenizer(model_config)
    if len(setup_result) == 3:
        model, tokenizer, modules_to_save_list = setup_result
    else:
        model, tokenizer = setup_result
        modules_to_save_list = None
    
    # Verify SPECTRAMoE class is accessible for DeepSpeed
    from models.spectra_model import SPECTRAMoE
    moe_layers_found = []
    for name, module in model.named_modules():
        if isinstance(module, SPECTRAMoE):
            moe_layers_found.append(name)
    logger.info(f"✅ Found {len(moe_layers_found)} SPECTRAMoE layers in model")
    if moe_layers_found:
        logger.info(f"   All MoE layers ({len(moe_layers_found)}):")
        for i, layer_name in enumerate(moe_layers_found):
            logger.debug(f"     [{i}] {layer_name}")
    else:
        logger.warning("⚠️ No SPECTRAMoE layers found! DeepSpeed may fail to find MoE classes.")
    
    # Setup dataset
    print("Setting up dataset...")
    dataset, collate_fn = setup_dataset(data_config, tokenizer)
    
    # 모델 및 데이터셋 로드 후 메모리 정리
    logger.info("🧹 모델 및 데이터셋 로드 후 GPU 메모리 정리...")
    clear_gpu_memory()
    
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
    if eval_dataset is None:
        splited = train_dataset.train_test_split(test_size=0.1)
        train_dataset = splited["train"]
        eval_dataset = splited["test"]
    
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
    print(f"  - 첫 번째 샘플 messages: {train_dataset[0]['messages'][:100]}")
    
    # 이미지가 있는 경우에만 출력 (multi-domain에서는 텍스트 전용 샘플이 있을 수 있음)
    first_sample_images = train_dataset[0].get('images', [])
    if first_sample_images and len(first_sample_images) > 0:
        if hasattr(first_sample_images[0], 'size'):
            print(f"  - 첫 번째 샘플 images: {first_sample_images[0].size}")
        else:
            print(f"  - 첫 번째 샘플 images: {type(first_sample_images[0])} (이미지 객체)")
    else:
        print(f"  - 첫 번째 샘플 images: 없음 (텍스트 전용 샘플)")
    
    trainer = SFTTrainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collate_fn,
        optimizers=(custom_optimizer, None) if custom_optimizer is not None else (None, None)
    )
    
    # CRITICAL: Trainer 생성 직후에 router 파라미터의 requires_grad를 설정하고 optimizer에 추가
    # Trainer 생성 시 optimizer가 초기화되면서 requires_grad=False인 파라미터는 제외됨
    # 따라서 Trainer 생성 직후에 requires_grad를 True로 설정하고 optimizer에 명시적으로 추가해야 함
    logger.info("=" * 80)
    logger.info("🔧 CRITICAL: Setting requires_grad=True for router parameters AFTER Trainer creation...")
    logger.info("   (Trainer has initialized optimizer - need to add router params explicitly)")
    logger.info("=" * 80)
    
    # 통합 함수 사용
    router_params_after_trainer, router_param_names_after_trainer, trainable_count = ensure_router_parameters_trainable(
        trainer.model, logger, context="Trainer_creation"
    )
    
    logger.info(f"✅ Router parameters trainable status: {trainable_count}/{len(router_params_after_trainer)}")
    if trainable_count < len(router_params_after_trainer):
        logger.error(f"❌ CRITICAL: {len(router_params_after_trainer) - trainable_count} router params still not trainable!")
        for i, (full_name, param) in enumerate(zip(router_param_names_after_trainer, router_params_after_trainer)):
            if not param.requires_grad:
                logger.error(f"   {full_name}: requires_grad={param.requires_grad}")
    
    # Optimizer에 명시적으로 추가 (일반 optimizer 케이스)
    if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
        optimizer_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
        router_param_ids = {id(p) for p in router_params_after_trainer}
        missing_params = [p for p in router_params_after_trainer if id(p) not in optimizer_param_ids]
        
        if missing_params:
            logger.info(f"🔧 Adding {len(missing_params)} router parameters to optimizer...")
            if len(trainer.optimizer.param_groups) > 0:
                trainer.optimizer.param_groups[0]['params'].extend(missing_params)
                logger.debug(f"  ✓ Added {len(missing_params)} parameters to optimizer param_groups[0]")
                
                # 재확인
                optimizer_param_ids_after = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
                in_optimizer_after = router_param_ids & optimizer_param_ids_after
                logger.debug(f"✅ Router params in optimizer (after fix): {len(in_optimizer_after)}/{len(router_params_after_trainer)}")
            else:
                logger.error("❌ No param_groups found in optimizer - cannot add parameters")
        else:
            logger.info(f"✅ All {len(router_params_after_trainer)} router parameters already in optimizer")
    elif hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
        logger.info("✅ DeepSpeed detected - router params with requires_grad=True will be included automatically")
        logger.info(f"   Router params with requires_grad=True: {trainable_count}/{len(router_params_after_trainer)}")
    else:
        logger.warning("⚠️ Optimizer not yet initialized - will be checked in ensure_router_in_optimizer")
    
    logger.info("=" * 80)
    
    # Trainer 생성 후 wandb가 초기화되었는지 확인하고, 필요시 초기화
    # DeepSpeed가 Trainer를 초기화할 때 wandb를 자동으로 초기화하지만,
    # callback이 wandb를 사용하기 전에 확실히 초기화되어 있는지 보장
    if training_config.get("report_to", None) and "wandb" in training_config["report_to"]:
        import wandb
        rank = int(os.getenv("RANK", "0"))
        if rank == 0 and (wandb.run is None or not wandb.run):
            # Trainer가 아직 wandb를 초기화하지 않았다면 여기서 초기화
            run = wandb.init(
                project="spectra-sft",
                name=training_config["run_name"],
                config=config,
                mode="online"  # 항상 online으로 wandb에 기록
            )
            run.define_metric("train/*", step_metric="train/global_step")
            run.define_metric("validation/*", step_metric="validation/step")
            run.define_metric("eval/*", step_metric="eval/step")
            run.define_metric("moe/*", step_metric="train/global_step")
            run.define_metric("multi_modality/*", step_metric="train/global_step")
            run.define_metric("router/*", step_metric="train/global_step")
            run.define_metric("other/*", step_metric="train/global_step")

            logger.info("✅ wandb initialized after Trainer creation")
        elif wandb.run is not None:
            logger.info("✅ wandb already initialized by Trainer")
    # ZeRO-3에서도 gradient checkpointing 사용 가능 (DeepSpeed activation checkpointing과 함께 사용)
    # 단, DeepSpeed config에 activation_checkpointing이 활성화되어 있으면 그것을 우선 사용
    try:
        ds_cfg_path = getattr(trainer.args, "deepspeed", None)
        if ds_cfg_path:
            import json
            with open(ds_cfg_path, "r") as f:
                ds_cfg = json.load(f)
            _zero_stage = int((ds_cfg.get("zero_optimization", {}) or {}).get("stage", 0) or 0)
            # DeepSpeed activation checkpointing이 활성화되어 있으면 그대로 사용
            ds_activation_checkpointing = ds_cfg.get("activation_checkpointing", {})
            if ds_activation_checkpointing and ds_activation_checkpointing.get("partition_activations", False):
                print("✓ DeepSpeed activation checkpointing 활성화됨 - PyTorch gradient checkpointing과 함께 사용")
                # PyTorch gradient checkpointing도 활성화 (DeepSpeed와 함께 사용 가능)
                trainer.args.gradient_checkpointing = True
            elif _zero_stage == 3:
                # ZeRO-3이고 DeepSpeed activation checkpointing이 없으면 PyTorch gradient checkpointing 사용
                print("✓ ZeRO-3에서 PyTorch gradient checkpointing 활성화 (메모리 절약)")
                trainer.args.gradient_checkpointing = True
    except Exception as e:
        print(f"⚠️ Gradient checkpointing 설정 확인 실패: {e}, 기본값 사용")
        pass
    
    # Add MoE monitoring callback
    trainer.add_callback(
        create_moe_callback_for_transformers(
            num_experts=model_config.get("spectra_params", {}).get("n_routed_experts", 8),
            log_every_n_steps=1,             # 매 스텝마다 로그 기록
            logger=wandb,                    # 사용할 로거 지정 (wandb)
            log_to_console=False,            # 콘솔에도 주요 메트릭 출력 (디버깅용)
            debug_logging=True,              # 디버그 로깅 활성화
            tokenizer=tokenizer,             # ✅ tokenizer 직접 전달
                       #  === (선택사항) ===  #
            log_heatmap_every=5,             # 500 스텝마다 Expert 사용률 히트맵 로깅
            alert_threshold_imbalance=4.0,   # 특정 Expert 사용률이 평균의 4배를 초과하면 경고
            unused_expert_threshold=0.25,    # 25% 이상의 Expert가 미사용되면 경고
            entropy_threshold=0.1,           # 라우팅 엔트로피가 0.1 미만이면 경고
            save_detailed_logs=False,        # 상세 JSON 로그 저장 여부
            enable_generation_logging=True,  # 생성 로깅 활성화
        ))
    
    # Add Router Weight Tracking callback (weight 변화 체크 및 학습 중단)
    router_weight_callback = RouterWeightTrackingCallback(
        save_dir=os.path.join(training_args.output_dir, "router_weight_logs"),
        log_every_n_steps=1,  # 매 step마다 체크
        check_weight_change=True,  # weight 변화 체크 활성화
        min_change_threshold=1e-8,  # 최소 변화 임계값
        check_after_steps=2,  # step 후부터 체크 시작 (step 부터 변화 데이터 있음)
        verbose=True,
    )
    trainer.add_callback(router_weight_callback)
    logger.info("✅ RouterWeightTrackingCallback added (will stop training if router weights don't change)")
    
    # CRITICAL: PEFT modules_to_save 동기화 callback 추가
    # original_module.*가 forward에서 사용되어 학습되지만, 저장은 modules_to_save.default.*에만 됨
    # 따라서 학습 중에 original_module.*의 값을 modules_to_save.default.*로 주기적으로 동기화해야 함
    from transformers import TrainerCallback
    
    class ModulesToSaveSyncCallback(TrainerCallback):
        """original_module.*의 값을 modules_to_save.default.*로 동기화"""
        def __init__(self, sync_every_n_steps=10):
            self.sync_every_n_steps = sync_every_n_steps
            self.last_sync_step = -1
        
        def on_step_end(self, args, state, control, model=None, **kwargs):
            """각 step 끝에서 동기화 (주기적으로)"""
            if state.global_step % self.sync_every_n_steps == 0 and state.global_step > self.last_sync_step:
                try:
                    actual_model = model
                    if hasattr(model, 'module'):  # DeepSpeed 래핑
                        actual_model = model.module
                    
                    if actual_model is None:
                        return control
                    
                    from models.spectra_model import SPECTRARouter
                    sync_count = 0
                    
                    for name, module in actual_model.named_modules():
                        if isinstance(module, SPECTRARouter):
                            # expression_projector의 linear_projection 동기화
                            if hasattr(module, 'expression_projector'):
                                expr_proj = module.expression_projector
                                if hasattr(expr_proj, 'linear_projection'):
                                    lin_proj = expr_proj.linear_projection
                                    
                                    # PEFT ModulesToSaveWrapper 확인
                                    if hasattr(lin_proj, 'original_module') and hasattr(lin_proj, 'modules_to_save'):
                                        if hasattr(lin_proj.modules_to_save, 'default'):
                                            default_module = lin_proj.modules_to_save.default
                                            
                                            # original_module의 파라미터를 modules_to_save.default로 복사
                                            for orig_param_name, orig_param in lin_proj.original_module.named_parameters(recurse=True):
                                                if hasattr(default_module, orig_param_name):
                                                    default_param = getattr(default_module, orig_param_name)
                                                    if orig_param.shape == default_param.shape:
                                                        with torch.no_grad():
                                                            default_param.data.copy_(orig_param.data)
                                                        sync_count += 1
                                    
                                    # 또는 modules_to_save.default가 직접 있는 경우
                                    elif hasattr(lin_proj, 'modules_to_save') and hasattr(lin_proj.modules_to_save, 'default'):
                                        # 이 경우는 이미 modules_to_save.default가 forward에서 사용되는 경우
                                        pass
                    
                    if sync_count > 0:
                        logger.debug(f"✅ Synced {sync_count} router parameters from original_module to modules_to_save.default at step {state.global_step}")
                    
                    self.last_sync_step = state.global_step
                except Exception as e:
                    logger.warning(f"⚠️ Failed to sync modules_to_save at step {state.global_step}: {e}")
            
            return control
        
        def on_save(self, args, state, control, model=None, **kwargs):
            """Checkpoint 저장 전에 동기화 (저장 직전)"""
            try:
                actual_model = model
                if hasattr(model, 'module'):  # DeepSpeed 래핑
                    actual_model = model.module
                
                if actual_model is None:
                    return control
                
                from models.spectra_model import SPECTRARouter
                sync_count = 0
                
                for name, module in actual_model.named_modules():
                    if isinstance(module, SPECTRARouter):
                        # expression_projector의 linear_projection 동기화
                        if hasattr(module, 'expression_projector'):
                            expr_proj = module.expression_projector
                            if hasattr(expr_proj, 'linear_projection'):
                                lin_proj = expr_proj.linear_projection
                                
                                # PEFT ModulesToSaveWrapper 확인
                                if hasattr(lin_proj, 'original_module') and hasattr(lin_proj, 'modules_to_save'):
                                    if hasattr(lin_proj.modules_to_save, 'default'):
                                        default_module = lin_proj.modules_to_save.default
                                        
                                        # original_module의 파라미터를 modules_to_save.default로 복사
                                        for orig_param_name, orig_param in lin_proj.original_module.named_parameters(recurse=True):
                                            if hasattr(default_module, orig_param_name):
                                                default_param = getattr(default_module, orig_param_name)
                                                if orig_param.shape == default_param.shape:
                                                    with torch.no_grad():
                                                        default_param.data.copy_(orig_param.data)
                                                    sync_count += 1
                
                if sync_count > 0:
                    logger.info(f"✅ Synced {sync_count} router parameters from original_module to modules_to_save.default before save at step {state.global_step}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to sync modules_to_save before save at step {state.global_step}: {e}")
            
            return control
    
    modules_to_save_sync_callback = ModulesToSaveSyncCallback(sync_every_n_steps=10)
    trainer.add_callback(modules_to_save_sync_callback)
    logger.info("✅ ModulesToSaveSyncCallback added (will sync original_module.* to modules_to_save.default.* every 10 steps and before save)")
    
    # ✅ Router 파라미터 검증 및 학습 가능하도록 강제 설정 (중복 코드 제거)
    def ensure_router_in_optimizer(trainer, model, modules_to_save_list=None):
        """Router 파라미터가 올바르게 학습 가능한지 검증하고 필요시 수정"""
        try:
            # 통합 함수 사용 (중복 코드 제거)
            router_params, router_param_names, trainable_count = ensure_router_parameters_trainable(
                model, logger, context="optimizer_validation"
            )
            
            if not router_params:
                logger.error("❌ CRITICAL: No router parameters found in model!")
                return
            
            logger.debug(f"✅ Found {len(router_params)} router parameters")
            logger.debug(f"✅ Router parameters trainable: {trainable_count}/{len(router_params)}")
            
            # Optimizer 포함 여부 확인 및 추가
            if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
                logger.info("✅ DeepSpeed detected - router params with requires_grad=True will be included automatically")
                if hasattr(trainer.deepspeed, 'optimizer') and trainer.deepspeed.optimizer is not None:
                    ds_optimizer = trainer.deepspeed.optimizer
                    if hasattr(ds_optimizer, 'param_groups'):
                        ds_param_ids = {id(p) for group in ds_optimizer.param_groups for p in group['params']}
                        router_param_ids = {id(p) for p in router_params}
                        in_ds_optimizer = router_param_ids & ds_param_ids
                        logger.debug(f"   Router params in DeepSpeed optimizer: {len(in_ds_optimizer)}/{len(router_params)}")
            
            elif hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                optimizer_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
                router_param_ids = {id(p) for p in router_params}
                in_optimizer = router_param_ids & optimizer_param_ids
                logger.debug(f"✅ Router params in optimizer: {len(in_optimizer)}/{len(router_params)}")
                
                if len(in_optimizer) < len(router_params):
                    missing_params = [p for p in router_params if id(p) not in optimizer_param_ids]
                    if len(trainer.optimizer.param_groups) > 0:
                        trainer.optimizer.param_groups[0]['params'].extend(missing_params)
                        logger.debug(f"  ✓ Added {len(missing_params)} parameters to optimizer")
            else:
                logger.warning("⚠️ Optimizer not yet initialized - will be checked after training starts")
        
        except Exception as e:
            logger.error(f"❌ Error validating router weights: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
    
    # Trainer 생성 후 router 파라미터를 optimizer에 추가
    # DeepSpeed의 경우 trainer.train() 호출 전에 해야 함
    logger.info("=" * 80)
    logger.info("🔍 FINAL CHECK: Ensuring router parameters are trainable and in optimizer...")
    logger.info("=" * 80)
    actual_model = trainer.model
    if hasattr(trainer.model, 'module'):  # DeepSpeed 래핑
        actual_model = trainer.model.module
    
    # modules_to_save_list 전달 (setup_model_and_tokenizer에서 생성됨)
    # main 함수에서 modules_to_save_list가 정의되어 있음
    ensure_router_in_optimizer(trainer, actual_model, modules_to_save_list)
    logger.info("=" * 80)
    
    # Add custom training progress callback
    from transformers import TrainerCallback
    
    # 배치 정보를 저장하는 callback (OOM 디버깅용)
    class BatchTrackingCallback(TrainerCallback):
        """배치 정보를 추적하여 OOM 발생 시 디버깅 정보 제공"""
        def __init__(self, trainer_ref):
            self.last_batch_info = None
            self.last_batch_step = -1
            self.trainer_ref = trainer_ref  # Trainer 참조
        
        def on_train_batch_begin(self, args, state, control, model=None, inputs=None, **kwargs):
            """배치 시작 시 배치 정보 저장 - 가장 확실한 방법"""
            try:
                if inputs is not None:
                    trainer = kwargs.get('trainer') or self.trainer_ref
                    self._save_batch_info(inputs, state.global_step, trainer)
            except Exception:
                pass  # 배치 정보 저장 실패해도 학습은 계속
        
        def on_step_begin(self, args, state, control, **kwargs):
            """Step 시작 시 배치 정보 저장 시도 (fallback)"""
            try:
                # Trainer의 내부 상태에서 배치 확인
                trainer = kwargs.get('trainer') or self.trainer_ref
                if trainer is not None:
                    # Trainer의 _current_batch 또는 최근 배치 확인
                    if hasattr(trainer, '_current_batch') and trainer._current_batch is not None:
                        self._save_batch_info(trainer._current_batch, state.global_step, trainer)
            except Exception:
                pass  # 배치 정보 저장 실패해도 학습은 계속
        
        def on_step_end(self, args, state, control, **kwargs):
            """Step 종료 시 배치 정보 저장 시도 (fallback)"""
            try:
                trainer = kwargs.get('trainer') or self.trainer_ref
                if trainer is not None:
                    # Trainer의 내부 상태에서 배치 확인
                    if hasattr(trainer, '_current_batch') and trainer._current_batch is not None:
                        self._save_batch_info(trainer._current_batch, state.global_step, trainer)
            except Exception:
                pass
        
        def _save_batch_info(self, batch, step, trainer):
            """배치 정보를 메모리 효율적으로 저장"""
            try:
                batch_info = {}
                
                # Trainer 설정에서 배치 정보 가져오기
                if trainer is not None:
                    try:
                        batch_info['per_device_batch_size'] = getattr(trainer, 'per_device_train_batch_size', None)
                        batch_info['gradient_accumulation_steps'] = getattr(trainer, 'gradient_accumulation_steps', None)
                        batch_info['num_devices'] = getattr(trainer.args, 'world_size', 1) if hasattr(trainer, 'args') else 1
                        if batch_info['per_device_batch_size'] and batch_info['gradient_accumulation_steps']:
                            batch_info['effective_batch_size'] = batch_info['per_device_batch_size'] * batch_info['gradient_accumulation_steps'] * batch_info['num_devices']
                        
                        # DataLoader에서 실제 배치 크기 확인
                        try:
                            train_dataloader = trainer.get_train_dataloader()
                            if hasattr(train_dataloader, 'batch_size'):
                                batch_info['dataloader_batch_size'] = train_dataloader.batch_size
                            elif hasattr(train_dataloader, 'batch_sampler') and hasattr(train_dataloader.batch_sampler, 'batch_size'):
                                batch_info['dataloader_batch_size'] = train_dataloader.batch_sampler.batch_size
                        except Exception:
                            pass
                    except Exception:
                        pass
                
                # Input IDs 정보
                if 'input_ids' in batch and torch.is_tensor(batch['input_ids']):
                    input_ids = batch['input_ids']
                    batch_info['input_ids_shape'] = list(input_ids.shape)
                    if len(input_ids.shape) > 1:
                        # 각 샘플의 실제 길이 (pad 제외)
                        pad_token_id = 0
                        # processing_class에서 tokenizer 가져오기 (deprecated된 tokenizer 대신)
                        processing_class = getattr(trainer, 'processing_class', None)
                        if processing_class is not None:
                            # AutoProcessor인 경우 tokenizer 속성에 접근
                            tokenizer = getattr(processing_class, 'tokenizer', processing_class)
                            pad_token_id = getattr(tokenizer, 'pad_token_id', 0) or getattr(tokenizer, 'eos_token_id', 0)
                        
                        sample_lengths = (input_ids != pad_token_id).sum(dim=1).cpu().tolist()
                        batch_info['sample_lengths'] = sample_lengths[:10]  # 최대 10개만
                        batch_info['max_length'] = max(sample_lengths) if sample_lengths else 0
                        batch_info['min_length'] = min(sample_lengths) if sample_lengths else 0
                        batch_info['avg_length'] = sum(sample_lengths) / len(sample_lengths) if sample_lengths else 0
                        batch_info['total_tokens'] = input_ids.numel()
                
                # Attention mask 정보
                if 'attention_mask' in batch and torch.is_tensor(batch['attention_mask']):
                    attn_mask = batch['attention_mask']
                    batch_info['attention_mask_shape'] = list(attn_mask.shape)
                    batch_info['attention_mask_total'] = attn_mask.numel()
                
                # Pixel values (이미지) 정보
                if 'pixel_values' in batch and torch.is_tensor(batch['pixel_values']):
                    pixel_values = batch['pixel_values']
                    batch_info['pixel_values_shape'] = list(pixel_values.shape)
                    batch_info['pixel_values_dtype'] = str(pixel_values.dtype)
                    batch_info['pixel_values_memory_mb'] = pixel_values.numel() * pixel_values.element_size() / 1024 / 1024
                    batch_info['num_images'] = pixel_values.shape[0] if len(pixel_values.shape) > 0 else 0
                
                # Image grid 정보
                if 'image_grid_thw' in batch:
                    batch_info['image_grid_thw'] = batch['image_grid_thw']
                
                # Labels 정보
                if 'labels' in batch and torch.is_tensor(batch['labels']):
                    labels = batch['labels']
                    batch_info['labels_shape'] = list(labels.shape)
                    if labels.numel() > 0:
                        non_ignore = (labels != -100).sum().item()
                        batch_info['non_ignore_tokens'] = non_ignore
                        batch_info['ignore_tokens'] = (labels == -100).sum().item()
                
                # 실제 배치 크기 (텐서에서 직접 확인)
                if 'input_ids' in batch and torch.is_tensor(batch['input_ids']):
                    batch_info['actual_batch_size'] = batch['input_ids'].shape[0] if len(batch['input_ids'].shape) > 0 else 1
                    # 기존 batch_size 필드도 유지 (하위 호환성)
                    batch_info['batch_size'] = batch_info['actual_batch_size']
                
                self.last_batch_info = batch_info
                self.last_batch_step = step
            except Exception as e:
                pass  # 배치 정보 저장 실패해도 학습은 계속
    
    # 배치 추적 callback 추가
    batch_tracker = BatchTrackingCallback(trainer)
    trainer.add_callback(batch_tracker)
    
    # training_step 래핑 제거 - DeepSpeed timer 충돌 방지
    # 대신 callback의 on_step_begin에서 배치 정보 저장
    
    class DetailedTrainingCallback(TrainerCallback):
        def __init__(self, logger):
            self.logger = logger
            self.last_log_time = time.time()
            self.log_interval = 10  # Log every 10 seconds during training
            
        def on_step_begin(self, args, state, control, **kwargs):
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                log_training_progress(
                    self.logger, 
                    kwargs.get('trainer'), 
                    step=state.global_step, 
                    epoch=state.epoch)
                log_gpu_memory(self.logger, f"STEP_{state.global_step}")
                self.last_log_time = current_time
                
        def on_step_end(self, args, state, control, **kwargs):
            # Log every 10 steps for detailed monitoring
            if state.global_step % 10 == 0:
                self.logger.debug(f"📊 Step {state.global_step} completed")
                
        def on_epoch_begin(self, args, state, control, **kwargs):
            self.logger.info(f"📅 Starting epoch {int(state.epoch)}")
            log_gpu_memory(self.logger, f"EPOCH_{int(state.epoch)}_START")
            
        def on_epoch_end(self, args, state, control, **kwargs):
            self.logger.info(f"📅 Completed epoch {int(state.epoch)}")
            log_gpu_memory(self.logger, f"EPOCH_{int(state.epoch)}_END")
            
        def on_train_begin(self, args, state, control, **kwargs):
            self.logger.info("🚀 Training started")
            log_gpu_memory(self.logger, "TRAINING_BEGIN")
            
        def on_train_end(self, args, state, control, **kwargs):
            self.logger.info("✅ Training ended")
            log_gpu_memory(self.logger, "TRAINING_END")
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                # Log important metrics
                if 'train_loss' in logs:
                    self.logger.info(f"📊 Train Loss: {logs['train_loss']:.6f}")
                if 'learning_rate' in logs:
                    self.logger.debug(f"📊 Learning Rate: {logs['learning_rate']:.2e}")
                if 'grad_norm' in logs:
                    self.logger.debug(f"📊 Gradient Norm: {logs['grad_norm']:.6f}")
    
    # trainer.add_callback(DetailedTrainingCallback(logger))

    # ===== Benchmark evaluation callback (lightweight by default) =====
    benchmark_eval_enabled = training_config.get("enable_benchmark_eval", True)
    benchmark_eval_tasks = training_config.get(
        "benchmark_eval_tasks",
        ['mmlu', 'hellaswag', 'gsm8k', 'truthfulqa', 'arc', 'ifeval'],  # IFEval integrated
    )
    benchmark_eval_mode = training_config.get("benchmark_eval_mode", "step")
    if benchmark_eval_mode not in {"step", "epoch"}:
        benchmark_eval_mode = "step"

    default_benchmark_freq = 1000 # training_config.get("eval_steps", 1000)
    benchmark_eval_frequency = int(training_config.get("benchmark_eval_frequency", default_benchmark_freq) or default_benchmark_freq)

    if benchmark_eval_enabled:
        logger.info(
            f"✅ Enabling benchmark callback (mode={benchmark_eval_mode}, freq={benchmark_eval_frequency}, tasks={benchmark_eval_tasks})"
        )
        trainer.add_callback(
            ModelEvalCallback(
                trainer=trainer,
                enable_benchmarks=True,
                benchmarks_to_run=benchmark_eval_tasks,
                benchmark_eval_frequency=benchmark_eval_frequency,
                eval_mode=benchmark_eval_mode,
                mme_max_samples=training_config.get("benchmark_mme_max_samples", 5),
                benchmark_max_samples_per_task=training_config.get("benchmark_max_samples_per_task", 3),
                benchmark_gsm8k_max_samples=training_config.get("benchmark_gsm8k_max_samples", 3),
                benchmark_max_tasks=training_config.get("benchmark_max_tasks"),
                benchmark_max_new_tokens=training_config.get("benchmark_max_new_tokens", 64),
                benchmark_disable_cot=training_config.get("benchmark_disable_cot", True),
                benchmark_ifeval_max_samples=training_config.get("benchmark_ifeval_max_samples", 5),
            )
        )
    else:
        logger.info("ℹ️ Benchmark callback disabled (enable_benchmark_eval=False)")

    # Print training info
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    if model_config.get("initialize_from_scratch", False):
        print(f"Model: Initializing from scratch (small model)")
    else:
        print(f"Model: {model_config.get('model_name_or_path', 'N/A')}")
    
    # Calculate and print model parameters
    try:
        # Get actual model (handle DeepSpeed wrapping)
        model_to_count = trainer.model
        if hasattr(model_to_count, 'module'):
            model_to_count = model_to_count.module
        
        # Count total parameters
        total_params = sum(p.numel() for p in model_to_count.parameters())
        trainable_params = sum(p.numel() for p in model_to_count.parameters() if p.requires_grad)
        
        # Format in B (billion) or M (million) units
        if total_params >= 1e9:
            total_str = f"{total_params / 1e9:.2f}B"
        elif total_params >= 1e6:
            total_str = f"{total_params / 1e6:.2f}M"
        else:
            total_str = f"{total_params / 1e3:.2f}K"
        
        if trainable_params >= 1e9:
            trainable_str = f"{trainable_params / 1e9:.2f}B"
        elif trainable_params >= 1e6:
            trainable_str = f"{trainable_params / 1e6:.2f}M"
        else:
            trainable_str = f"{trainable_params / 1e3:.2f}K"
        
        print(f"Model parameters: {total_str} total, {trainable_str} trainable")
    except Exception as e:
        print(f"Model parameters: Unable to calculate ({e})")
    
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
    try:
        # Log training start
        logger.info(f"🚀 Starting training...")
        logger.info(f"🔧 Training configuration:")
        logger.info(f"  - Epochs: {training_config['num_train_epochs']}")
        logger.info(f"  - Batch size per device: {training_config['per_device_train_batch_size']}")
        logger.info(f"  - Gradient accumulation steps: {training_config['gradient_accumulation_steps']}")
        logger.info(f"  - Learning rate: {training_config['learning_rate']}")
        logger.info(f"  - Max sequence length: {data_config['max_seq_length']}")
        
        # eval 최적화를 위한 커스텀 eval 함수 설정
        logger.info("🔧 Setting up memory-optimized evaluation...")
        original_eval_fn = getattr(trainer, 'evaluate', None)
        trainer.evaluate = lambda eval_dataset=None, ignore_keys=None, metric_key_prefix="eval": eval_with_memory_optimization(trainer, original_eval_fn, eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        # 학습 시작 전 메모리 정리
        logger.info("🧹 학습 시작 전 GPU 메모리 정리...")
        clear_gpu_memory()
        
        # DataLoader 최적화 (메모리 절약)
        if hasattr(trainer.args, 'dataloader_num_workers'):
            if trainer.args.dataloader_num_workers is None or trainer.args.dataloader_num_workers > 0:
                logger.info(f"🔧 DataLoader num_workers를 0으로 설정 (메모리 절약)")
                trainer.args.dataloader_num_workers = 1
        
        # Log initial memory state
        log_gpu_memory(logger, "TRAINING_START")
        
        # Enable checkpoint debug mode for detailed error messages
        logger.info("🔍 Enabling gradient checkpointing debug mode...")
        torch.utils.checkpoint.set_checkpoint_debug_enabled(True)
        
        # Start training with progress monitoring
        start_time = time.time()
        
        enable_profiler = bool(int(os.getenv("PROFILE_TRAINING", "0")))
        if enable_profiler:
            from torch.profiler import profile, record_function, ProfilerActivity
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                trainer.train()
                profiler_table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
                wandb.log({"profiler_table": wandb.Table(data=[profiler_table])})
        else:
            with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):
                trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed successfully in {training_time:.2f} seconds")
        
    except torch.OutOfMemoryError as e:
        # CUDA OOM 전용 처리
        handle_cuda_oom(e, trainer, logger)
        raise e
        
    except MemoryError as e:
        # 로컬 RAM OOM 전용 처리
        handle_ram_oom(e, trainer, logger)
        raise e
        
    except KeyboardInterrupt as e:
        handle_training_exception(e, trainer, logger, context="training_keyboard_interrupt")
        raise e
        
    except RuntimeError as e:
        handle_training_exception(e, trainer, logger, context="training_runtime_error")
        raise e
        
    except Exception as e:
        handle_training_exception(e, trainer, logger, context="training")
        raise e
        
    finally:
        # 원래 eval 함수 복원
        # Save final model (실패해도 evaluation은 실행)
        model_saved = False
        try:
            print("Saving final model...")
            logger.info("💾 Saving final model...")
            if config.get("deepspeed_config") is not None:
                try:
                    trainer.deepspeed.save_checkpoint(training_args.output_dir)
                    logger.info("✅ DeepSpeed checkpoint saved")
                except Exception as ds_e:
                    logger.warning(f"⚠️ DeepSpeed checkpoint save failed: {ds_e}")
            
            trainer.save_model()
            logger.info("✅ Model saved")
            model_saved = True
        except Exception as save_e:
            logger.error(f"❌ Model save failed: {save_e}")
            log_error_context(logger, save_e, "model_save")
            # 모델 저장 실패해도 evaluation은 실행
        
        # Save tokenizer (실패해도 evaluation은 실행)
        try:
            tokenizer.save_pretrained(training_args.output_dir)
            logger.info("✅ Tokenizer saved")
        except Exception as tokenizer_e:
            logger.warning(f"⚠️ Tokenizer save failed: {tokenizer_e}")
        
        print("Training End")
        logger.info("🏁 Training End")
        
        if original_eval_fn:
            logger.debug("🔧 Restoring original evaluation function...")
            trainer.evaluate = original_eval_fn
        
        # 학습 종료 후 validation 실행 (항상 실행, 모델 저장 실패해도 실행)
        try:
            logger.info("\n" + "=" * 80)
            logger.info("🚀 Starting Post-Training Validation")
            logger.info("=" * 80)
            logger.info("⚠️ Note: Validation will run even if training was interrupted or model save failed")
            
            model_path = training_args.output_dir
            training_config_path = config_path
            
            # 모델이 저장되었는지 확인
            if not model_saved:
                logger.warning("⚠️ Model save failed, but validation will still attempt to run")
                logger.warning("⚠️ If validation fails, check if model files exist in output directory")
            
            # Config 파일 경로 찾기
            if training_config_path is None:
                # 기본 경로 시도
                default_config = "spectra_sft/config/spectra_small_config.json"
                if os.path.exists(default_config):
                    training_config_path = default_config
                    logger.info(f"📄 Using default config: {default_config}")
                else:
                    logger.warning("⚠️ Training config path not found, some validations may be skipped")
            
            validation_results = run_post_training_validation(
                model_path=model_path,
                training_config_path=training_config_path,
                output_dir=training_args.output_dir,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            logger.info("✅ Post-training validation completed")
            
        except Exception as e:
            logger.error(f"❌ Post-training validation failed: {e}")
            log_error_context(logger, e, "post_training_validation")
            # Validation 실패해도 학습은 완료된 것으로 간주
            import traceback
            logger.error(f"❌ Validation error traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    register_custom_optimizers()
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="SPECTRA SFT Training with Config File")
        parser.add_argument(
            "--config", 
            type=str, 
            default="spectra_sft/config/spectra_small_config.json",
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
        # wandb.init()은 Trainer가 자동으로 초기화하도록 함
        # DeepSpeed가 Trainer를 초기화할 때 wandb를 재초기화할 수 있으므로
        # 여기서 수동으로 초기화하지 않고 Trainer의 자동 초기화를 사용
        
        main(model_config, data_config, training_config, config_path=args.config)

    except Exception as e:
        logger.error(f"❌ Fatal error in main: {str(e)}")
        log_error_context(logger, e, "main_function")
        
        # OOM 에러인 경우 상세 정보 수집 및 저장
        error_msg = str(e)
        is_memory_error = (
            "CUDA out of memory" in error_msg or
            "CUBLAS_STATUS_ALLOC_FAILED" in error_msg or
            "cublasCreate" in error_msg
        )
        
        if is_memory_error:
            logger.error("❌ Fatal OOM error detected in main function")
            logger.error("💾 Collecting and saving error information...")
            try:
                # trainer 객체가 있는지 확인 (없을 수도 있음)
                trainer = None
                if 'trainer' in locals():
                    trainer = locals()['trainer']
                elif 'trainer' in globals():
                    trainer = globals()['trainer']
                
                if trainer is not None:
                    output_dir = None
                    if hasattr(trainer, 'args') and hasattr(trainer.args, 'output_dir'):
                        output_dir = trainer.args.output_dir
                    elif hasattr(trainer, 'training_args') and hasattr(trainer.training_args, 'output_dir'):
                        output_dir = trainer.training_args.output_dir
                    
                    error_file = save_oom_error_info(logger, trainer, e, batch_info=None, output_dir=output_dir)
                    if error_file:
                        logger.error(f"✅ Fatal OOM 에러 정보가 저장되었습니다: {error_file}")
                else:
                    # trainer가 없는 경우에도 환경 정보만이라도 저장
                    logger.error("⚠️ Trainer 객체를 찾을 수 없어 환경 정보만 수집합니다...")
                    try:
                        env_info = collect_environment_info()
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        error_file = f"logs/fatal_oom_error_info_{timestamp}.json"
                        os.makedirs("logs", exist_ok=True)
                        with open(error_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                'timestamp': timestamp,
                                'environment': env_info,
                                'error': {
                                    'error_type': type(e).__name__,
                                    'error_message': str(e),
                                    'error_traceback': traceback.format_exc()
                                }
                            }, f, indent=2, ensure_ascii=False, default=str)
                        logger.error(f"✅ Fatal OOM 에러 정보가 저장되었습니다: {error_file}")
                    except Exception as save_e:
                        logger.error(f"❌ 에러 정보 저장 실패: {save_e}")
            except Exception as collect_e:
                logger.error(f"❌ 에러 정보 수집 실패: {collect_e}")
        
        # Log final memory state
        if torch.cuda.is_available():
            logger.error("❌ Final GPU memory state:")
            logger.error(f"❌ Memory summary:\n{torch.cuda.memory_summary()}")
            logger.error(f"❌ Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
            logger.error(f"❌ Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f}GB")
        
        # Re-raise the exception
        raise e
