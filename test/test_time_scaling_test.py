import sys
import os
import torch
import time
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import Gemma3ForCausalLM, Gemma3Config
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.image_utils import load_image
from peft.peft_model import PeftModel
import tensorrt
import pprint
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (G3MoEConfig, G3MoEForCausalLM, G3MoETextConfig)

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch.compiler.disable()

# Additional safety: reset any existing dynamo state
torch._dynamo.reset()
print("TensorRT version:", tensorrt.__version__)

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} M"
    else:
        return str(number)

def measure_inference_time(model, inputs, num_runs=5):
    """Measure inference time for multiple runs"""
    times = []
    
    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            _ = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=10,
                do_sample=False
            )
    
    # Actual measurement
    with torch.inference_mode():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=10,
                do_sample=False
            )
            end_time = time.time()
            times.append(end_time - start_time)
    
    return times

def test_time_scaling_experiment():
    """Test different scaling configurations and measure performance"""
    
    base_model_name = "google/gemma-3-4b-it"
    base_config = Gemma3Config.from_pretrained(base_model_name)
    base_config = base_config.to_dict()
    
    # Test different scaling configurations
    scaling_configs = [
        {
            "name": "Baseline (No Scaling)",
            "config": {
                "n_shared_experts": 1,
                "n_routed_experts": 5,
                "n_group": 4,
                "topk_group": 8,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 18,
                "router_aux_loss_coef": 0.001,
                "router_jitter_noise": 0.01,
                "input_jitter_noise": 0.01,
                "model_type": "g3moe_text",
                "no_rope_layer_interval": 4,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 1.0  # No scaling
                },
                "use_bfloat16": True,
            }
        },
        {
            "name": "Conservative Scaling (2x)",
            "config": {
                "n_shared_experts": 1,
                "n_routed_experts": 5,
                "n_group": 4,
                "topk_group": 8,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 18,
                "router_aux_loss_coef": 0.001,
                "router_jitter_noise": 0.01,
                "input_jitter_noise": 0.01,
                "model_type": "g3moe_text",
                "no_rope_layer_interval": 4,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 2.0  # 2x scaling
                },
                "use_bfloat16": True,
            }
        },
        {
            "name": "Aggressive Scaling (8x)",
            "config": {
                "n_shared_experts": 1,
                "n_routed_experts": 5,
                "n_group": 4,
                "topk_group": 8,
                "num_experts_per_tok": 2,
                "first_k_dense_replace": 18,
                "router_aux_loss_coef": 0.001,
                "router_jitter_noise": 0.01,
                "input_jitter_noise": 0.01,
                "model_type": "g3moe_text",
                "no_rope_layer_interval": 4,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 8.0  # 8x scaling
                },
                "use_bfloat16": True,
            }
        }
    ]
    
    # Prepare test input
    tokenizer = AutoProcessor.from_pretrained(base_model_name)
    with open("/home/conan_jung/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
        tokenizer.chat_template = f.read()
    
    test_text = """
안녕하세요.<end_of_turn><eos>
<bos><start_of_turn>system
You are a helpful assistant named Sparkle.
Always answer in shortest possible sentence.
But you should remember... Try to answer in Korean.😉
<end_of_turn><eos>
<start_of_turn>user
this is the test text message. now you must instruct the model to generate a response to this message.
"""
    
    test_input = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": test_text}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in Korean."},
                    {"type": "image", "url": "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg"}
                ]
            }
        ],
        add_generation_prompt=True,
    )
    
    # Load test image
    image = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")
    
    inputs = tokenizer(
        text=test_input,
        images=[image],
        return_tensors="pt"
    )
    
    results = []
    
    for config_info in scaling_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config_info['name']}")
        print(f"{'='*60}")
        
        try:
            # Create model config
            model_config_dict = base_config.copy()
            model_config_dict['text_config'].update(config_info['config'])
            model_config_dict.update(model_config_dict['text_config'])
            model_config = G3MoEConfig(**model_config_dict)
            
            # Load model
            print("Loading model...")
            model = G3MoEForCausalLM.from_pretrained(
                pretrained_model_name_or_path=base_model_name,
                config=model_config,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
            
            inputs = inputs.to(model.device)
            
            print(f"Model parameters: {format_parameters(model.num_parameters())}")
            print(f"Scaling factor: {config_info['config']['rope_scaling']['factor']}")
            
            # Measure inference time
            print("Measuring inference time...")
            inference_times = measure_inference_time(model, inputs, num_runs=5)
            
            # Test generation quality
            print("Testing generation quality...")
            with torch.inference_mode():
                response = tokenizer.batch_decode(
                    model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        generation_config=GenerationConfig(
                            device=model.device,
                            max_new_tokens=20,
                            do_sample=True,
                            top_p=0.9,
                            top_k=1,
                            temperature=0.7,
                            repetition_penalty=1.2,
                            length_penalty=1.0,
                        ),
                        tokenizer=tokenizer
                    )
                )[0]
            
            generated_text = response[len(test_text):].split("<start_of_turn>model\n")[-1]
            
            # Store results
            result = {
                "config_name": config_info['name'],
                "scaling_factor": config_info['config']['rope_scaling']['factor'],
                "avg_inference_time": sum(inference_times) / len(inference_times),
                "min_inference_time": min(inference_times),
                "max_inference_time": max(inference_times),
                "std_inference_time": torch.std(torch.tensor(inference_times)).item(),
                "generated_text": generated_text,
                "model_parameters": model.num_parameters()
            }
            
            results.append(result)
            
            print(f"Average inference time: {result['avg_inference_time']:.4f}s")
            print(f"Generated text: {generated_text}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with {config_info['name']}: {str(e)}")
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST TIME SCALING RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n{result['config_name']}:")
        print(f"  Scaling Factor: {result['scaling_factor']}x")
        print(f"  Avg Inference Time: {result['avg_inference_time']:.4f}s")
        print(f"  Min/Max Time: {result['min_inference_time']:.4f}s / {result['max_inference_time']:.4f}s")
        print(f"  Std Dev: {result['std_inference_time']:.4f}s")
        print(f"  Generated: {result['generated_text'][:100]}...")
    
    # Performance comparison
    if len(results) >= 2:
        baseline = results[0]
        print(f"\n{'='*80}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        
        for result in results[1:]:
            speedup = baseline['avg_inference_time'] / result['avg_inference_time']
            print(f"\n{result['config_name']} vs Baseline:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Time difference: {baseline['avg_inference_time'] - result['avg_inference_time']:.4f}s")

if __name__ == "__main__":
    test_time_scaling_experiment() 