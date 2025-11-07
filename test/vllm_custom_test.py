#!/usr/bin/env python3
"""
G3MoE vLLM Integration Test - Enhanced Version
===============================================

Enhanced vLLM integration test with:
- Support for both LLM (high-level) and LLMEngine (low-level) APIs
- Better error handling and diagnostics
- Model validation before vLLM loading
- Comprehensive logging
"""

import os
import sys
import torch
import time
import logging
import json
import shutil
from typing import Optional, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["HF_HOME"] = "/mls/conan/training_logs/vllm_test_save"

# Check dependencies
VLLM_AVAILABLE = False
VLLM_VERSION = None
try:
    import vllm
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    VLLM_VERSION = getattr(vllm, '__version__', 'unknown')
    # Try to import low-level API
    try:
        from vllm import LLMEngine, EngineArgs
        LLMENGINE_AVAILABLE = True
    except ImportError:
        LLMENGINE_AVAILABLE = False
except ImportError:
    LLMENGINE_AVAILABLE = False

G3MOE_AVAILABLE = False
try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from models.g3moe_model import G3MoEForConditionalGeneration, G3MoEModel, G3MoETextModel
    from models.g3moe_config import G3MoETextConfig, G3MoEConfig
    G3MOE_AVAILABLE = True
    
    # Register with transformers
    if VLLM_AVAILABLE:
        from transformers.modeling_utils import VLMS
        AutoConfig.register("g3moe", G3MoEConfig)
        AutoModel.register(G3MoEConfig, G3MoEModel)
        AutoConfig.register("g3moe_text", G3MoETextConfig)
        AutoModel.register(G3MoETextConfig, G3MoETextModel)
        AutoModelForCausalLM.register(G3MoEConfig, G3MoEForConditionalGeneration)
        VLMS.append("g3moe")

        from vllm import ModelRegistry
        ModelRegistry.register_model("G3MoEModel", G3MoEModel)
        ModelRegistry.register_model("G3MoETextModel", G3MoETextModel)
        ModelRegistry.register_model("G3MoEForCausalLM", G3MoEForConditionalGeneration)
except ImportError as e:
    import traceback
    traceback.print_exc()
    G3MOE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} M"
    else:
        return str(number)


def validate_model_files(model_path: str) -> Dict[str, bool]:
    """Validate that all required model files exist"""
    required_files = {
        'config.json': os.path.exists(os.path.join(model_path, 'config.json')),
        'model files': any(os.path.exists(os.path.join(model_path, f)) 
                          for f in ['pytorch_model.bin', 'model.safetensors'] + 
                                  [f'model-{i:05d}-of-{n:05d}.safetensors' 
                                   for i in range(10) for n in range(1, 10)]),
        'tokenizer': any(os.path.exists(os.path.join(model_path, f)) 
                        for f in ['tokenizer.json', 'tokenizer_config.json']),
        'model code': any(os.path.exists(os.path.join(model_path, f)) 
                         for f in ['g3moe_model.py', 'modeling_g3moe.py']),
    }
    return required_files

def save_g3moe_for_vllm(
    save_path: str,
    base_model: Optional[str] = "google/gemma-3-4b-it",
) -> bool:
    """
    Save G3MoE model in HuggingFace compatible format for vLLM.
    
    Args:
        save_path: Path to save the model
        base_model: Base model to use for tokenizer/config defaults
        config_overrides: Optional config overrides
    """
    if not G3MOE_AVAILABLE:
        logger.error("G3MoE not available")
        return False
     
    try:
        from transformers import AutoProcessor, AutoConfig
        
        logger.info(f"Saving G3MoE model to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        default_config = AutoConfig.from_pretrained(base_model)
        default_config = default_config.to_dict()
        # Default config
        default_config["text_config"].update({
            "model_type": "g3moe_text",
            "n_shared_experts": 1,
            "n_routed_experts": 3,
            "n_group": 2,
            "topk_group": 2,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 22,
            "router_aux_loss_coef": 0.001,
            "router_jitter_noise": 0.01,
            "input_jitter_noise": 0.01,
            "model_type": "g3moe_text",
            "rope_scaling": {
                "rope_type": "linear",
                "factor": 8.0
            },
            "use_bfloat16": True,
        })
        default_config.update({"model_type": "g3moe"})
        # Create config
        # text_config = G3MoETextConfig(**default_config)
        config = G3MoEConfig(**default_config)
        
        # Create and initialize model
        logger.info("Creating G3MoE model...")
        model = G3MoEForConditionalGeneration(config)
        model.init_weights()
        logger.info(f"Model size: {format_parameters(model.num_parameters())}")

        # Save model and config
        logger.info("Saving model weights...")
        model.save_pretrained(save_path, safe_serialization=False)
        config.save_pretrained(save_path)
        
        # Save tokenizer (try base model first, fallback to gemma)
        logger.info("Saving tokenizer...")
        try:
            if base_model:
                tokenizer = AutoProcessor.from_pretrained(base_model)
            else:
                tokenizer = AutoProcessor.from_pretrained(base_model)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {base_model}, using {base_model}: {e}")
            tokenizer = AutoProcessor.from_pretrained(base_model)
        
        tokenizer.save_pretrained(save_path)
        
        # Copy model source files (required for trust_remote_code)
        logger.info("Copying model source files...")
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_files = ['g3moe_config.py', 'g3moe_model.py']
        
        for file in model_files:
            src = os.path.join(models_dir, file)
            if os.path.exists(src):
                dst = os.path.join(save_path, file)
                shutil.copy2(src, dst)
                logger.debug(f"Copied {file} to {save_path}")
            else:
                logger.warning(f"Model file not found: {src}")
        
        # Update config.json with auto_map and required fields
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Ensure required fields for vLLM
        config_dict.update({
            "model_type": "g3moe",
            "architectures": ["G3MoEForConditionalGeneration"],
            # auto_map is critical for vLLM to load custom models
            "auto_map": {
                "AutoConfig": "g3moe_config.G3MoEConfig",
                "AutoModel": "g3moe_model.G3MoEModel",
                "AutoModelForCausalLM": "g3moe_model.G3MoEForConditionalGeneration"
            }
        })
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Validate saved files
        validation = validate_model_files(save_path)
        logger.info("Model file validation:")
        for file_type, exists in validation.items():
            status = "✅" if exists else "❌"
            logger.info(f"  {status} {file_type}: {exists}")
        
        logger.info(f"✅ Model saved successfully to {save_path}")
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to save model: {e}")
        traceback.print_exc()
        return False


def test_vllm_inference_high_level(model_path: str) -> bool:
    """
    Test G3MoE inference with vLLM using high-level LLM API.
    This is the recommended approach for most use cases.
    """
    if not VLLM_AVAILABLE:
        logger.error("vLLM not available")
        return False
    
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        return False
    
    try:
        logger.info("Testing vLLM inference with high-level LLM API...")
        
        # Validate model files first
        validation = validate_model_files(model_path)
        missing = [k for k, v in validation.items() if not v]
        if missing:
            logger.warning(f"Missing files: {missing}")
        
        # Load model with vLLM (high-level API)
        logger.info(f"Loading model from {model_path}...")
        llm = LLM(
            model=model_path,
            trust_remote_code=True,  # Required for custom models
            max_model_len=512,
            gpu_memory_utilization=0.7,
            dtype="bfloat16",
            tensor_parallel_size=1,  # Single GPU for testing
        )
        
        logger.info("Model loaded successfully!")
        
        # Test generation
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=50,
            top_p=0.95,
        )
        prompts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
        ]
        
        logger.info("Generating responses...")
        outputs = llm.generate(prompts, sampling_params)
        
        logger.info("Generation results:")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            logger.info(f"\n--- Prompt {i+1} ---")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated_text}")
            logger.info(f"Tokens: {len(output.outputs[0].token_ids)}")
        
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"vLLM inference failed: {e}")
        traceback.print_exc()
        return False


def test_vllm_inference_low_level(model_path: str) -> bool:
    """
    Test G3MoE inference with vLLM using low-level LLMEngine API.
    This provides more control but is more complex.
    """
    if not VLLM_AVAILABLE or not LLMENGINE_AVAILABLE:
        logger.error("vLLM LLMEngine not available")
        return False
    
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        return False
    
    try:
        logger.info("Testing vLLM inference with low-level LLMEngine API...")
        
        # Create engine args
        engine_args = EngineArgs(
            model=model_path,
            trust_remote_code=True,
            max_model_len=512,
            gpu_memory_utilization=0.7,
            dtype="bfloat16",
            tensor_parallel_size=1,
        )
        
        # Create engine
        engine = LLMEngine.from_engine_args(engine_args)
        logger.info("Engine created successfully!")
        
        # Test generation
        sampling_params = SamplingParams(temperature=0.8, max_tokens=30)
        request_id = "test_request_0"
        
        # Add request
        engine.add_request(
            request_id=request_id,
            prompt="Hello, how are you?",
            sampling_params=sampling_params,
        )
        
        # Process requests
        outputs = []
        while engine.has_unfinished_requests():
            step_outputs = engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        
        logger.info("Generation results:")
        for output in outputs:
            logger.info(f"Generated: {output.outputs[0].text}")
        
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"vLLM LLMEngine inference failed: {e}")
        traceback.print_exc()
        return False


def test_model_loading_with_transformers(model_path: str) -> bool:
    """
    Test if model can be loaded with transformers (prerequisite for vLLM).
    """
    if not G3MOE_AVAILABLE:
        logger.error("G3MoE not available")
        return False
    
    try:
        logger.info("Testing model loading with transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        logger.info("✅ Model loaded successfully with transformers")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✅ Tokenizer loaded successfully")
        
        # Test forward pass
        test_text = "Hello, how are you?"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info("✅ Forward pass successful")
        logger.info(f"Output shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"Transformers loading failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function with comprehensive testing"""
    logger.info("=" * 60)
    logger.info("G3MoE vLLM Integration Test - Enhanced Version")
    logger.info("=" * 60)
    
    # Environment check
    logger.info("\n=== Environment Check ===")
    logger.info(f"vLLM available: {VLLM_AVAILABLE}")
    if VLLM_AVAILABLE:
        logger.info(f"vLLM version: {VLLM_VERSION}")
        logger.info(f"LLMEngine available: {LLMENGINE_AVAILABLE}")
    logger.info(f"G3MoE available: {G3MOE_AVAILABLE}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    if not G3MOE_AVAILABLE:
        logger.error("❌ G3MoE not available - cannot proceed")
        return
    
    if not VLLM_AVAILABLE:
        logger.warning("⚠️  vLLM not available - will only test model saving and transformers loading")
    
    # Test model saving
    test_path = f"/mls/conan/training_logs/vllm_test_save/modules/transformers_modules/test_g3moe_model_{os.getpid()}_{int(time.time())}"
    
    logger.info("\n=== Test 1: Save G3MoE Model ===")
    if not save_g3moe_for_vllm(test_path):
        logger.error("❌ Model saving failed - cannot proceed with other tests")
        return
    
    logger.info("✅ Model saving successful")
    
    # Test transformers loading (prerequisite for vLLM)
    logger.info("\n=== Test 2: Transformers Model Loading ===")
    transformers_success = test_model_loading_with_transformers(test_path)
    if not transformers_success:
        logger.error("❌ Transformers loading failed - vLLM will likely fail too")
        logger.info("💡 Fix transformers loading issues first")
    
    # Test vLLM inference (if available)
    if VLLM_AVAILABLE:
        logger.info("\n=== Test 3: vLLM Inference (High-Level API) ===")
        vllm_high_success = test_vllm_inference_high_level(test_path)
        
        if LLMENGINE_AVAILABLE:
            logger.info("\n=== Test 4: vLLM Inference (Low-Level API) ===")
            vllm_low_success = test_vllm_inference_low_level(test_path)
        else:
            vllm_low_success = False
            logger.info("⚠️  LLMEngine API not available - skipping low-level test")
    else:
        vllm_high_success = False
        vllm_low_success = False
        logger.info("⚠️  vLLM not available - skipping vLLM tests")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("=== Test Summary ===")
    logger.info("=" * 60)
    logger.info(f"✅ Model saving: {'PASS' if True else 'FAIL'}")
    logger.info(f"{'✅' if transformers_success else '❌'} Transformers loading: {'PASS' if transformers_success else 'FAIL'}")
    if VLLM_AVAILABLE:
        logger.info(f"{'✅' if vllm_high_success else '❌'} vLLM (high-level): {'PASS' if vllm_high_success else 'FAIL'}")
        if LLMENGINE_AVAILABLE:
            logger.info(f"{'✅' if vllm_low_success else '❌'} vLLM (low-level): {'PASS' if vllm_low_success else 'FAIL'}")
    
    # Recommendations
    logger.info("\n=== Recommendations ===")
    if not transformers_success:
        logger.info("💡 Fix transformers loading issues first:")
        logger.info("   - Check model files are complete")
        logger.info("   - Verify auto_map in config.json")
        logger.info("   - Ensure model source files are copied")
    elif VLLM_AVAILABLE and not vllm_high_success:
        logger.info("💡 vLLM integration tips:")
        logger.info("   - Ensure trust_remote_code=True")
        logger.info("   - Check vLLM version compatibility")
        logger.info("   - Verify model implements required interfaces")
        logger.info("   - Check vLLM logs for detailed error messages")
    elif VLLM_AVAILABLE and vllm_high_success:
        logger.info("🎉 vLLM integration successful!")
        logger.info(f"💡 Model ready for deployment at: {test_path}")
    
    # Cleanup option
    logger.info("\n=== Cleanup ===")
    cleanup = input(f"Delete test model at {test_path}? (y/N): ").strip().lower()
    if cleanup == 'y':
        try:
            if os.path.exists(test_path):
                shutil.rmtree(test_path)
                logger.info(f"✅ Cleaned up {test_path}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    else:
        logger.info(f"💡 Test model preserved at: {test_path}")


if __name__ == "__main__":
    main() 