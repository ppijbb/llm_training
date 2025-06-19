#!/usr/bin/env python3
"""
G3MoE vLLM Integration Test - Simplified Version
=================================================

Essential functionality for testing G3MoE with vLLM.
Removed bloat and focused on actual working features.
"""

import os
import sys
import torch
import logging
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check dependencies
try:
    from vllm import LLM, SamplingParams, EngineArgs
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from models.g3moe_model import G3MoEForCausalLM
    from models.g3moe_config import G3MoETextConfig, G3MoEConfig
    G3MOE_AVAILABLE = True
    if VLLM_AVAILABLE:
        from vllm import ModelRegistry
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        # AutoConfig.register("g3moe", G3MoEConfig)
        AutoConfig.register("g3moe_text", G3MoETextConfig)
        AutoModel.register(G3MoETextConfig, G3MoEForCausalLM)
        AutoModelForCausalLM.register(G3MoETextConfig, G3MoEForCausalLM)
        ModelRegistry.register_model(model_arch="G3MoEForCausalLM", model_cls=G3MoEForCausalLM)
except ImportError:
    G3MOE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_g3moe_for_vllm(save_path: str) -> bool:
    """
    Save G3MoE model in HuggingFace compatible format for vLLM.
    This is the ONLY function that actually works for vLLM integration.
    """
    if not G3MOE_AVAILABLE:
        logger.error("G3MoE not available")
        return False
    
    try:
        from transformers import AutoTokenizer
        import json
        import shutil

        logger.info(f"Saving G3MoE model to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        # Create and save model
        text_config = G3MoETextConfig(
            **{
                "n_shared_experts": 1,
                "n_routed_experts": 5, # 256, 15, 6
                "n_group": 4,
                "topk_group": 8,
                # "num_key_value_heads": base_config['text_config']['num_attention_heads'],
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 18,
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
        config = G3MoEConfig(text_config=text_config)
        model = G3MoEForCausalLM(config.text_config)
        model.init_weights()
        
        # Save model and config
        model.save_pretrained(save_path)
        config.save_pretrained(save_path)
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(save_path)
        
        # Copy model files
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        for file in ['g3moe_config.py', 'g3moe_model.py', '__init__.py']:
            src = os.path.join(models_dir, file)
            if os.path.exists(src):
                shutil.copy2(src, save_path)
        
        # Update config.json with auto_map
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config_dict.update({
            "model_type": "g3moe_text",
            "architectures": ["G3MoEForCausalLM"],
            "auto_map": {
                "AutoConfig": "g3moe_config.G3MoETextConfig",
                "AutoModelForCausalLM": "g3moe_model.G3MoEForCausalLM"
            }
        })
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"✅ Model saved successfully to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False


def test_vllm_inference(model_path: str) -> bool:
    """
    Test G3MoE inference with vLLM.
    Note: This will likely fail due to vLLM-transformers integration issues.
    """
    if not VLLM_AVAILABLE:
        logger.error("vLLM not available")
        return False
    
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        return False
    
    try:
        logger.info("Testing vLLM inference...")

        # Load model with vLLM
        llm = LLM(
            **dict(
                model=model_path,
                model_impl="transformers",
                trust_remote_code=True,
                max_model_len=512,
                gpu_memory_utilization=0.7,)
        )
        
        # Test generation
        sampling_params = SamplingParams(temperature=0.8, max_tokens=30)
        prompts = ["Hello, how are you?"]
        
        outputs = llm.generate(prompts, sampling_params)
        
        for output in outputs:
            logger.info(f"Prompt: {output.prompt}")
            logger.info(f"Generated: {output.outputs[0].text}")
        
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"vLLM inference failed: {e}")
        return False


def main():
    """Main test function - only essential functionality"""
    logger.info("=== G3MoE vLLM Integration Test ===")
    
    # Environment check
    logger.info(f"vLLM available: {VLLM_AVAILABLE}")
    logger.info(f"G3MoE available: {G3MOE_AVAILABLE}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    
    if not (VLLM_AVAILABLE and G3MOE_AVAILABLE):
        logger.error("Required dependencies not available")
        return
    
    # Test model saving
    test_path = "./test_g3moe_model"
    
    logger.info("\n=== Test 1: Save G3MoE Model ===")
    if save_g3moe_for_vllm(test_path):
        logger.info("✅ Model saving successful")
        
        # Test vLLM inference
        logger.info("\n=== Test 2: vLLM Inference ===")
        if test_vllm_inference(test_path):
            logger.info("✅ vLLM inference successful")
        else:
            logger.error("❌ vLLM inference failed (expected)")
            logger.info("This is expected due to transformers-vLLM integration issues")
    else:
        logger.error("❌ Model saving failed")
    
    # Cleanup
    try:
        import shutil
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
            logger.info(f"Cleaned up {test_path}")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")
    
    logger.info("\n=== Summary ===")
    logger.info("✅ Working: G3MoE model saving in HF format")
    logger.info("❌ Not working: vLLM inference with custom model")
    logger.info("💡 Recommendation: Use native transformers for now")


if __name__ == "__main__":
    main() 