"""
Example usage of Universal MoE Upcycling Module

This module demonstrates how to convert any pretrained HuggingFace model
into a Mixture of Experts (MoE) model using the universal upcycling functionality.
"""

from transformers import AutoModelForCausalLM, AutoConfig
from models.universal_moe import upcycle_model_to_moe, UniversalMoERouter
from models.g3moe_model import G3MoEMLP


def example_qwen3_to_moe():
    """Example: Convert Qwen3 to MoE"""
    print("Loading Qwen3 model...")
    # Note: Qwen3 may not be available yet, using Qwen2.5-7B as fallback
    # Replace with actual Qwen3 model name when available
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Instruct")

    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_experts": 2,
        "num_experts_per_tok": 4,
        "router_dim": 128,
        "n_shared_experts": 1,
        "first_k_dense_replace": 2,  # Keep first 2 layers as dense
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.0,
        "freeze_shared_experts": True,
        "balancing_strength": 0.01,
        "ema_alpha": 0.99,
        "hidden_activation": "silu",
    }
    
    print("Upcycling Qwen3 to MoE...")
    model = upcycle_model_to_moe(
        model, 
        moe_config,
        expert_module_class=G3MoEMLP,
        layer_start_idx=2,  # Start from layer 2
        verbose=True
    )
    
    print("Conversion complete!")
    return model


def example_gemma3_to_moe():
    """Example: Convert Gemma 3 to MoE"""
    print("Loading Gemma 3 model...")
    # Try Gemma 3 first, fallback to Gemma 2 if not available
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_experts": 3,
        "num_experts_per_tok": 4,
        "router_dim": 128,
        "n_shared_experts": 1,
        "first_k_dense_replace": 2,  # Keep first 2 layers as dense
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.0,
        "freeze_shared_experts": True,
        "balancing_strength": 0.01,
        "ema_alpha": 0.99,
        "hidden_activation": "gelu",
    }
    
    print("Upcycling Gemma 3 to MoE...")
    model = upcycle_model_to_moe(
        model,
        moe_config,
        expert_module_class=G3MoEMLP,
        layer_start_idx=2,  # Start from layer 2
        verbose=True
    )
    
    print("Conversion complete!")
    return model


def example_llama31_to_moe():
    """Example: Convert Llama 3.1 to MoE"""
    print("Loading Llama 3.1 model...")
    # Use Llama 3.1 8B (closest to 7B limit) or smaller if available
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-3B-Instruct")
    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_experts": 32,
        "num_experts_per_tok": 8,
        "router_dim": 128,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,  # Keep first 3 layers as dense
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.0,
        "freeze_shared_experts": True,
        "balancing_strength": 0.01,
        "ema_alpha": 0.99,
        "hidden_activation": "silu",
    }
    
    print("Upcycling Llama 3.1 to MoE...")
    model = upcycle_model_to_moe(
        model,
        moe_config,
        expert_module_class=G3MoEMLP,
        layer_start_idx=3,  # Start from layer 3
        verbose=True
    )
    
    print("Conversion complete!")
    return model


def example_gpt_oss_to_moe():
    """Example: Convert GPT-OSS to MoE"""
    print("Loading GPT-OSS model...")
    # Try OpenAI GPT-OSS model, fallback to GPT-2 if not available
    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
    # Define MoE configuration
    moe_config = {
        "hidden_size": model.config.n_embd if hasattr(model.config, 'n_embd') else model.config.hidden_size,
        "intermediate_size": model.config.n_inner if hasattr(model.config, 'n_inner') else model.config.intermediate_size,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "router_dim": 128,
        "n_shared_experts": 1,
        "first_k_dense_replace": 0,  # Convert all layers
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.0,
        "freeze_shared_experts": True,
        "balancing_strength": 0.01,
        "ema_alpha": 0.99,
        "hidden_activation": "gelu",
    }
    
    print("Upcycling GPT-OSS to MoE...")
    model = upcycle_model_to_moe(
        model,
        moe_config,
        expert_module_class=G3MoEMLP,
        verbose=True
    )
    
    print("Conversion complete!")
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Universal MoE Upcycling Examples")
    print("Using latest models: Qwen 3, Gemma 3, Llama 3.1, GPT-OSS")
    print("=" * 60)
    
    # Run examples with latest models (all <= 7B)
    print("\n1. Qwen 3 (0.6B) to MoE:")
    print("-" * 60)
    model = example_qwen3_to_moe()
    del model
    
    print("\n2. Gemma 3 (1B) to MoE:")
    print("-" * 60)
    model = example_gemma3_to_moe()
    del model
    
    print("\n3. Llama 3.1 (3B) to MoE:")
    print("-" * 60)
    model = example_llama31_to_moe()
    del model
    
    print("\n4. GPT-OSS (20B) to MoE:")
    print("Note: GPT-OSS exceeds 7B limit (20B and 120B available)")
    print("-" * 60)
    model = example_gpt_oss_to_moe()
    del model
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
