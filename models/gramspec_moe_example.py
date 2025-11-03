"""
Example usage of GramSpecMoE (Gram Matrix-based Specialization Routing)

This module demonstrates how to convert any pretrained HuggingFace model
into a Mixture of Experts (MoE) model using GramSpecRouter routing method.
GramSpec uses Gram matrix-based orthogonal constraints for expert specialization
and diversity.
"""
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from .gramspec_moe import upcycle_model_to_moe, GramSpecRouter
from .g3moe_model import G3MoEMLP

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def format_parameters(number):
    """Format parameter count in human-readable format"""
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.4f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.4f} M"
    else:
        return str(number)


def count_parameters(model):
    """Count total number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_size_comparison(params_before, model_after, model_name):
    """Print comparison of model sizes before and after upcycling"""
    params_after = count_parameters(model_after)
    
    print(f"\n{'='*60}")
    print(f"Model Size Comparison: {model_name}")
    print(f"{'='*60}")
    print(f"Before upcycling: {format_parameters(params_before)} ({params_before:,} parameters)")
    print(f"After upcycling:  {format_parameters(params_after)} ({params_after:,} parameters)")
    
    if params_before > 0:
        ratio = params_after / params_before
        increase = params_after - params_before
        print(f"Increase:         {format_parameters(increase)} ({increase:,} parameters)")
        print(f"Size ratio:       {ratio:.4f}x ({ratio*100:.2f}% of original)")
    print(f"{'='*60}\n")


def create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128):
    """
    Create input from actual text using tokenizer with chat template
    
    Args:
        tokenizer: The tokenizer to use
        test_text: Test text to encode (default: simple question)
        max_length: Maximum sequence length (default: 128)
    
    Returns:
        Dictionary with input_ids and attention_mask
    """
    # Create a simple conversation format
    messages = [
        {"role": "user", "content": test_text}
    ]
    
    # Try to use apply_chat_template if available, otherwise just tokenize
    try:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            # Use chat template
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                max_length=max_length,
                truncation=True,
                return_dict=True,
                return_tensors="pt"
            )
        else:
            # Fallback to simple tokenization
            tokenized = tokenizer(
                test_text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )
    except Exception as e:
        # If chat template fails, fallback to simple tokenization
        print(f"Warning: Chat template failed ({e}), using simple tokenization")
        tokenized = tokenizer(
            test_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }


def measure_forward_speed(model, dummy_input, num_runs=10, warmup_runs=3, device=None):
    """
    Measure forward pass speed for a model
    
    Args:
        model: The model to test
        dummy_input: Dictionary with input_ids and attention_mask
        num_runs: Number of runs to average (default: 10)
        warmup_runs: Number of warmup runs (default: 3)
        device: Device to run on (None = auto-detect from model)
    
    Returns:
        Dictionary with timing statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    model.to(device)
    
    # Move inputs to device
    input_ids = dummy_input["input_ids"].to(device)
    attention_mask = dummy_input["attention_mask"].to(device)
    
    # Warmup runs
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Synchronize CUDA if using GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual measurement
    times = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # Calculate throughput (tokens per second)
    seq_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]
    throughput = (seq_len * batch_size) / avg_time
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "throughput": throughput,
        "times": times
    }


def measure_generation_speed(model, tokenizer, dummy_input, max_new_tokens=32, num_runs=10, warmup_runs=3, device=None):
    """
    Measure generation speed (forward + generate) for a model
    
    Args:
        model: The model to test
        tokenizer: The tokenizer
        dummy_input: Dictionary with input_ids and attention_mask
        max_new_tokens: Maximum number of new tokens to generate (default: 32)
        num_runs: Number of runs to average (default: 10)
        warmup_runs: Number of warmup runs (default: 3)
        device: Device to run on (None = auto-detect from model)
    
    Returns:
        Dictionary with timing statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    model.to(device)
    
    # Move inputs to device
    input_ids = dummy_input["input_ids"].to(device)
    attention_mask = dummy_input["attention_mask"].to(device)
    
    # Warmup runs
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
    
    # Synchronize CUDA if using GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual measurement
    times = []
    generated_tokens = []
    generated_text = None  # Store single generated text sample (deterministic, so all runs are same)
    
    with torch.inference_mode():
        for run_idx in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Count generated tokens (excluding input)
            num_generated = output.shape[1] - input_ids.shape[1]
            generated_tokens.append(num_generated)
            
            # Decode generated text (only store once, deterministic so all runs produce same output)
            if run_idx == 0:
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # Calculate throughput (tokens per second)
    avg_generated_tokens = sum(generated_tokens) / len(generated_tokens)
    throughput = avg_generated_tokens / avg_time if avg_time > 0 else 0
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "throughput": throughput,
        "avg_generated_tokens": avg_generated_tokens,
        "times": times,
        "generated_tokens": generated_tokens,
        "generated_text": generated_text  # Store single generated text (deterministic)
    }


def print_speed_comparison(speed_before, speed_after, model_name, is_generation=False):
    """Print comparison of speeds before and after upcycling (horizontal layout)"""
    # Calculate speed change metrics
    if speed_before['avg_time'] > 0:
        speed_ratio = speed_before['avg_time'] / speed_after['avg_time']
        time_diff = (speed_after['avg_time'] - speed_before['avg_time']) * 1000
        time_diff_percent = (time_diff / (speed_before['avg_time'] * 1000)) * 100
    
    # Format values
    before_avg = f"{speed_before['avg_time']*1000:.2f} ms"
    before_minmax = f"{speed_before['min_time']*1000:.2f} ms / {speed_before['max_time']*1000:.2f} ms"
    before_std = f"{speed_before['std_time']*1000:.2f} ms"
    before_throughput = f"{speed_before['throughput']:.2f} tokens/sec"
    
    after_avg = f"{speed_after['avg_time']*1000:.2f} ms"
    after_minmax = f"{speed_after['min_time']*1000:.2f} ms / {speed_after['max_time']*1000:.2f} ms"
    after_std = f"{speed_after['std_time']*1000:.2f} ms"
    after_throughput = f"{speed_after['throughput']:.2f} tokens/sec"
    
    if speed_before['avg_time'] > 0:
        change_ratio = f"{speed_ratio:.4f}x"
        if time_diff > 0:
            change_text = f"Slower by: {time_diff:.2f} ms ({time_diff_percent:.2f}%)"
        else:
            change_text = f"Faster by: {abs(time_diff):.2f} ms ({abs(time_diff_percent):.2f}%)"
    else:
        change_ratio = "N/A"
        change_text = "N/A"
    
    # Add generation-specific metrics if available
    if is_generation:
        before_tokens = f"{speed_before.get('avg_generated_tokens', 0):.1f} tokens"
        after_tokens = f"{speed_after.get('avg_generated_tokens', 0):.1f} tokens"
    
    # Print horizontal comparison table
    title = "Generation Speed Comparison" if is_generation else "Forward Speed Comparison"
    
    # Define column widths for consistent alignment
    col1_width = 20
    col2_width = 30
    col3_width = 30
    separator = " │ "
    
    print(f"\n{'='*90}")
    print(f"{title}: {model_name}".center(90))
    print(f"{'='*90}")
    
    # Header row
    header = f"{'Metric':<{col1_width}}{separator}{'Before upcycling':<{col2_width}}{separator}{'After upcycling':<{col3_width}}"
    print(header)
    
    # Separator row - exactly match header width
    sep_row = f"{'─'*col1_width}──{'─'*col2_width}──{'─'*col3_width}"
    print(sep_row)
    
    # Data rows
    print(f"{'Avg time':<{col1_width}}{separator}{before_avg:<{col2_width}}{separator}{after_avg:<{col3_width}}")
    print(f"{'Min/Max':<{col1_width}}{separator}{before_minmax:<{col2_width}}{separator}{after_minmax:<{col3_width}}")
    print(f"{'Std dev':<{col1_width}}{separator}{before_std:<{col2_width}}{separator}{after_std:<{col3_width}}")
    if is_generation:
        print(f"{'Avg tokens':<{col1_width}}{separator}{before_tokens:<{col2_width}}{separator}{after_tokens:<{col3_width}}")
    print(f"{'Throughput':<{col1_width}}{separator}{before_throughput:<{col2_width}}{separator}{after_throughput:<{col3_width}}")
    
    # Bottom separator
    print(sep_row)
    
    # Summary rows
    print(f"{'Speed ratio':<{col1_width}}{separator}{change_ratio:<{col2_width}}{separator}{'':<{col3_width}}")
    print(f"{'Change':<{col1_width}}{separator}{change_text:<{col2_width}}{separator}{'':<{col3_width}}")
    print(f"{'='*90}")
    
    # Print generated text samples if available
    if is_generation:
        before_text = speed_before.get('generated_text', None)
        after_text = speed_after.get('generated_text', None)
        
        if before_text or after_text:
            print(f"\n{'Generated Text Comparison':<90}")
            print(f"{'─'*90}")
            
            if before_text:
                print(f"\nBefore upcycling:")
                # Wrap long text for better readability
                text_lines = []
                max_line_len = 88
                words = before_text.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_len:
                        current_line += (word + " ") if current_line else word
                    else:
                        if current_line:
                            text_lines.append("  " + current_line)
                        current_line = word
                if current_line:
                    text_lines.append("  " + current_line)
                
                for line in text_lines:
                    print(line)
            
            if after_text:
                print(f"\nAfter upcycling:")
                # Wrap long text for better readability
                text_lines = []
                max_line_len = 88
                words = after_text.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_len:
                        current_line += (word + " ") if current_line else word
                    else:
                        if current_line:
                            text_lines.append("  " + current_line)
                        current_line = word
                if current_line:
                    text_lines.append("  " + current_line)
                
                for line in text_lines:
                    print(line)
            
            print(f"{'─'*90}\n")


def example_qwen3_to_moe():
    """Example: Convert Qwen3 to MoE"""
    print("Loading Qwen3 model...")
    model_name = "Qwen/Qwen3-0.6B"
    model_before = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Count parameters before upcycling
    params_before = count_parameters(model_before)
    print(f"Model loaded. Parameters before upcycling: {format_parameters(params_before)}")
    
    # Measure forward speed before upcycling
    print("Measuring forward speed before upcycling...")
    text_input = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_before = measure_forward_speed(model_before, text_input, num_runs=10, warmup_runs=3)
    
    # Measure generation speed BEFORE upcycling (must be done before upcycling modifies model in-place)
    print("Measuring generation speed before upcycling (max_new_tokens=32)...")
    gen_speed_before = measure_generation_speed(model_before, tokenizer, text_input, max_new_tokens=32, num_runs=5, warmup_runs=2)
    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model_before.config.hidden_size,
        "intermediate_size": model_before.config.intermediate_size,
        "num_experts": 2,
        "num_experts_per_tok": 1,  # Must be <= num_experts
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
    model_after = upcycle_model_to_moe(
        model_before, 
        moe_config,
        expert_module_class=G3MoEMLP,
        layer_start_idx=2,  # Start from layer 2
        verbose=True
    )
    
    print("Conversion complete!")
    print_size_comparison(params_before, model_after, "Qwen3")
    
    # Measure forward speed after upcycling
    print("Measuring forward speed after upcycling...")
    text_input_after = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_after = measure_forward_speed(model_after, text_input_after, num_runs=10, warmup_runs=3)
    print_speed_comparison(speed_before, speed_after, "Qwen3")
    
    # Measure generation speed AFTER upcycling
    print("Measuring generation speed after upcycling (max_new_tokens=32)...")
    gen_speed_after = measure_generation_speed(model_after, tokenizer, text_input_after, max_new_tokens=32, num_runs=5, warmup_runs=2)
    print_speed_comparison(gen_speed_before, gen_speed_after, "Qwen3", is_generation=True)
    
    return model_after


def example_gemma3_to_moe():
    """Example: Convert Gemma 3 to MoE"""
    print("Loading Gemma 3 model...")
    model_name = "google/gemma-3-1b-it"
    model_before = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Count parameters before upcycling
    params_before = count_parameters(model_before)
    print(f"Model loaded. Parameters before upcycling: {format_parameters(params_before)}")
    
    # Measure forward speed before upcycling
    print("Measuring forward speed before upcycling...")
    text_input = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_before = measure_forward_speed(model_before, text_input, num_runs=10, warmup_runs=3)
    
    # Measure generation speed BEFORE upcycling (must be done before upcycling modifies model in-place)
    print("Measuring generation speed before upcycling (max_new_tokens=32)...")
    gen_speed_before = measure_generation_speed(model_before, tokenizer, text_input, max_new_tokens=32, num_runs=5, warmup_runs=2)
    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model_before.config.hidden_size,
        "intermediate_size": model_before.config.intermediate_size,
        "num_experts": 3,
        "num_experts_per_tok": 1,  # Must be <= num_experts
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
    model_after = upcycle_model_to_moe(
        model_before,
        moe_config,
        expert_module_class=G3MoEMLP,
        layer_start_idx=2,  # Start from layer 2
        verbose=True
    )
    
    print("Conversion complete!")
    print_size_comparison(params_before, model_after, "Gemma 3")
    
    # Measure forward speed after upcycling
    print("Measuring forward speed after upcycling...")
    text_input_after = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_after = measure_forward_speed(model_after, text_input_after, num_runs=10, warmup_runs=3)
    print_speed_comparison(speed_before, speed_after, "Gemma 3")
    
    # Measure generation speed AFTER upcycling
    print("Measuring generation speed after upcycling (max_new_tokens=32)...")
    gen_speed_after = measure_generation_speed(model_after, tokenizer, text_input_after, max_new_tokens=32, num_runs=5, warmup_runs=2)
    print_speed_comparison(gen_speed_before, gen_speed_after, "Gemma 3", is_generation=True)
    
    return model_after


def example_llama31_to_moe():
    """Example: Convert Llama 3.2 to MoE"""
    print("Loading Llama 3.2 model...")
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model_before = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Count parameters before upcycling
    params_before = count_parameters(model_before)
    print(f"Model loaded. Parameters before upcycling: {format_parameters(params_before)}")
    
    # Measure forward speed before upcycling
    print("Measuring forward speed before upcycling...")
    text_input = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_before = measure_forward_speed(model_before, text_input, num_runs=10, warmup_runs=3)
    
    # Measure generation speed BEFORE upcycling (must be done before upcycling modifies model in-place)
    print("Measuring generation speed before upcycling (max_new_tokens=32)...")
    gen_speed_before = measure_generation_speed(model_before, tokenizer, text_input, max_new_tokens=32, num_runs=5, warmup_runs=2)
    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model_before.config.hidden_size,
        "intermediate_size": model_before.config.intermediate_size,
        "num_experts": 3,
        "num_experts_per_tok": 1,  # Must be <= num_experts
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
    
    print("Upcycling Llama 3.2 to MoE...")
    model_after = upcycle_model_to_moe(
        model_before,
        moe_config,
        expert_module_class=G3MoEMLP,
        layer_start_idx=3,  # Start from layer 3
        verbose=True
    )
    
    print("Conversion complete!")
    print_size_comparison(params_before, model_after, "Llama 3.2")
    
    # Measure forward speed after upcycling
    print("Measuring forward speed after upcycling...")
    text_input_after = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_after = measure_forward_speed(model_after, text_input_after, num_runs=10, warmup_runs=3)
    print_speed_comparison(speed_before, speed_after, "Llama 3.2")
    
    # Measure generation speed AFTER upcycling
    print("Measuring generation speed after upcycling (max_new_tokens=32)...")
    gen_speed_after = measure_generation_speed(model_after, tokenizer, text_input_after, max_new_tokens=32, num_runs=5, warmup_runs=2)
    print_speed_comparison(gen_speed_before, gen_speed_after, "Llama 3.2", is_generation=True)
    
    return model_after


def example_gpt_oss_to_moe():
    """Example: Convert GPT-OSS to MoE"""
    print("Loading GPT-OSS model...")
    model_name = "openai/gpt-oss-20b"
    # Try OpenAI GPT-OSS model, fallback to GPT-2 if not available
    try:
        model_before = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        print("GPT-OSS model not found. Skipping this example.")
        return None
    
    # Count parameters before upcycling
    params_before = count_parameters(model_before)
    print(f"Model loaded. Parameters before upcycling: {format_parameters(params_before)}")
    
    # Measure forward speed before upcycling
    print("Measuring forward speed before upcycling...")
    text_input = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_before = measure_forward_speed(model_before, text_input, num_runs=10, warmup_runs=3)
    
    # Measure generation speed BEFORE upcycling (must be done before upcycling modifies model in-place)
    print("Measuring generation speed before upcycling (max_new_tokens=32)...")
    gen_speed_before = measure_generation_speed(model_before, tokenizer, text_input, max_new_tokens=32, num_runs=5, warmup_runs=2)
    
    # Define MoE configuration
    moe_config = {
        "hidden_size": model_before.config.n_embd if hasattr(model_before.config, 'n_embd') else model_before.config.hidden_size,
        "intermediate_size": model_before.config.n_inner if hasattr(model_before.config, 'n_inner') else model_before.config.intermediate_size,
        "num_experts": 32,
        "num_experts_per_tok": 4, # oss 20B config ref
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
    model_after = upcycle_model_to_moe(
        model_before,
        moe_config,
        expert_module_class=G3MoEMLP,
        verbose=True
    )
    
    print("Conversion complete!")
    print_size_comparison(params_before, model_after, "GPT-OSS")
    
    # Measure forward speed after upcycling
    print("Measuring forward speed after upcycling...")
    text_input_after = create_text_input(tokenizer, test_text="What is the capital of France?", max_length=128)
    speed_after = measure_forward_speed(model_after, text_input_after, num_runs=10, warmup_runs=3)
    print_speed_comparison(speed_before, speed_after, "GPT-OSS")
    
    # Measure generation speed AFTER upcycling
    print("Measuring generation speed after upcycling (max_new_tokens=32)...")
    gen_speed_after = measure_generation_speed(model_after, tokenizer, text_input_after, max_new_tokens=32, num_runs=5, warmup_runs=2)
    print_speed_comparison(gen_speed_before, gen_speed_after, "GPT-OSS", is_generation=True)
    
    return model_after


if __name__ == "__main__":
    print("=" * 60)
    print("GramSpecMoE Upcycling Examples")
    print("Expression Orthogonal Routing for MoE")
    print("Using latest models: Qwen 3, Gemma 3, Llama 3.2, GPT-OSS")
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
    
    print("\n3. Llama 3.2 (3B) to MoE:")
    print("-" * 60)
    model = example_llama31_to_moe()
    del model
    
    print("\n4. GPT-OSS (20B) to MoE:")
    print("-" * 60)
    model = example_gpt_oss_to_moe()
    del model
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
