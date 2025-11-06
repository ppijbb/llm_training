import sys
import os
import copy
import torch
from tqdm.auto import tqdm
import logging
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration, Gemma3Config, Gemma3Model
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.image_utils import load_image

from peft.peft_model import PeftModel
import tensorrt
import random
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (#G2MoEConfig, G2MoEForCausalLM, G2MoETextConfig,
                    G3MoEModel, G3MoETextModel,
                    G3MoEConfig, G3MoEForCausalLM, G3MoEForConditionalGeneration, G3MoETextConfig)
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import VLMS
AutoConfig.register("g3moe", G3MoEConfig)
AutoConfig.register("g3moe", G3MoEConfig)
AutoConfig.register("g3moe_text", G3MoETextConfig)
AutoModel.register(G3MoEConfig, G3MoEModel)
AutoModel.register(G3MoETextConfig, G3MoETextModel)
AutoModelForCausalLM.register(G3MoETextConfig, G3MoEForCausalLM)
VLMS.append("g3moe")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch.compiler.disable()

# Additional safety: reset any existing dynamo state
torch._dynamo.reset()
print("version of tensorrt: " ,tensorrt.__version__)

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.4f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.4f} M"
    else:
        return str(number)


base_model_name = "google/gemma-3-4b-it"
model_architecture = G3MoEForConditionalGeneration
base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
base_config = base_config.to_dict()
moe_config = {
        "n_shared_experts": 1,
        "n_routed_experts": 6, # 256, 15, 6
        "n_group": 4,
        "topk_group": 8,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 18,
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.01,
        "input_jitter_noise": 0.01,
        "model_type": "g3moe_text",
        "no_rope_layer_interval": 0,
        "rope_scaling":{
            "rope_type": "yarn",
            "factor": 8.0
        },
        "use_bfloat16": True
    }
if "text_config" not in base_config:
    base_config['text_config'] = copy.deepcopy(base_config)

base_config['text_config'].update(moe_config)
base_config.update(base_config['text_config'])
model_config = G3MoEConfig(**base_config)
model_config.model_type = "gemma3"
model_config.text_config.model_type = "gemma3_text"
# BitsAndBytesConfig int-4 config
model_config.architectures = [
    "G3MoEForConditionalGeneration", 
    # "G3MoEModel", 
    # "G3MoEForCausalLM"
    ]
    
def count_active_parameters(model, top_k=None, verbose=True):
    """
    Inference 시 실제 활성화되는 파라미터 수를 계산합니다.
    
    Args:
        model: G3MoE 모델
        top_k: 활성화되는 expert 수 (None이면 config에서 가져옴)
        verbose: 상세 출력 여부
    
    Returns:
        dict: 활성화 파라미터 정보
    """
    # Config에서 MoE 설정 가져오기
    if hasattr(model, 'config'):
        config = model.config
        if hasattr(config, 'text_config'):
            config = config.text_config
    else:
        raise ValueError("Model must have config attribute")
    
    n_routed_experts = getattr(config, 'n_routed_experts', 0)
    n_shared_experts = getattr(config, 'n_shared_experts', 1)
    num_experts_per_tok = top_k if top_k is not None else getattr(config, 'num_experts_per_tok', 2)
    first_k_dense_replace = getattr(config, 'first_k_dense_replace', 0)
    num_hidden_layers = getattr(config, 'num_hidden_layers', 0)
    
    # MoE 레이어 수 계산 (first_k_dense_replace 이후 레이어만 MoE)
    num_moe_layers = max(0, num_hidden_layers - first_k_dense_replace)
    
    # 전체 파라미터 카운트
    total_params = sum(p.numel() for p in model.parameters())
    
    # 카테고리별 파라미터 카운트
    embedding_params = 0
    attention_params = 0
    shared_expert_params = 0
    routed_expert_params = 0
    dense_mlp_params = 0  # Dense 레이어의 MLP (MoE가 아닌 레이어)
    router_params = 0
    global_router_params = 0  # Global router (공유됨, 한 번만 카운트)
    norm_params = 0
    lm_head_params = 0
    vision_params = 0
    other_params = 0
    
    # Global router는 한 번만 카운트해야 함
    global_router_seen = set()
    
    # 레이어 인덱스 추출 헬퍼 함수
    def get_layer_idx(name):
        """레이어 인덱스 추출 (예: 'model.layers.5.moe' -> 5)"""
        import re
        match = re.search(r'\.layers\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return -1
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        layer_idx = get_layer_idx(name)
        is_moe_layer = layer_idx >= first_k_dense_replace if layer_idx >= 0 else False
        
        # Vision tower 파라미터
        if 'vision_tower' in name or 'vision_model' in name:
            vision_params += param_count
        # Embedding 파라미터
        elif 'embed' in name.lower():
            embedding_params += param_count
        # Attention 파라미터
        elif 'self_attn' in name or 'attn' in name:
            attention_params += param_count
        # Global Router 파라미터 (공유되므로 한 번만 카운트)
        elif 'global_router' in name or ('router' in name and 'global' in name and 'language_model' in name):
            # Global router는 한 번만 카운트
            router_key = 'global_router'
            if router_key not in global_router_seen:
                global_router_params += param_count
                global_router_seen.add(router_key)
        # Shared Expert 파라미터 (항상 활성화)
        elif 'shared_experts' in name or 'shared_expert' in name:
            shared_expert_params += param_count
        # Routed Expert 파라미터 (MoE 레이어의 experts만)
        elif 'experts' in name and 'shared' not in name:
            # MoE 레이어의 experts만 카운트
            if is_moe_layer or 'moe' in name.lower():
                routed_expert_params += param_count
            else:
                # Dense 레이어의 MLP는 다른 곳에서 처리
                other_params += param_count
        # Dense MLP 파라미터 (MoE가 아닌 레이어의 MLP, 항상 활성화)
        elif ('mlp' in name.lower() or 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name) and \
             'moe' not in name.lower() and 'expert' not in name.lower():
            # MoE가 아닌 레이어의 MLP만 카운트
            if not is_moe_layer:
                dense_mlp_params += param_count
            else:
                other_params += param_count
        # Router 파라미터 (로컬 router, 각 레이어마다 있지만 항상 활성화)
        elif 'router' in name or 'gate' in name:
            # Global router가 아닌 경우만 카운트
            if 'global' not in name:
                router_params += param_count
        # LayerNorm 파라미터
        elif 'norm' in name.lower() or 'layernorm' in name.lower():
            norm_params += param_count
        # LM Head 파라미터
        elif 'lm_head' in name or 'score' in name:
            lm_head_params += param_count
        else:
            other_params += param_count
    
    # Forward pass에서 모든 레이어가 활성화됩니다!
    # 단, MoE 레이어 내의 Routed Experts만 sparse activation (토큰당 top_k개만)
    
    # 항상 활성화되는 파라미터 (모든 레이어)
    always_active_params = (
        embedding_params +
        attention_params +  # 모든 레이어의 attention
        shared_expert_params +  # 모든 MoE 레이어의 shared expert
        dense_mlp_params +  # Dense 레이어의 MLP (모두 활성화)
        global_router_params +  # Global router (항상 활성화)
        router_params +  # 모든 router (항상 활성화)
        norm_params +  # 모든 LayerNorm
        lm_head_params +
        vision_params +
        other_params
    )
    
    # Routed Experts만 sparse activation
    # 각 토큰마다 top_k개만 선택되므로, 전체 routed experts의 일부만 활성화
    if n_routed_experts > 0:
        activation_ratio = num_experts_per_tok / n_routed_experts
        active_routed_expert_params = int(routed_expert_params * activation_ratio)
    else:
        activation_ratio = 0.0
        active_routed_expert_params = 0
    
    # 전체 활성화 파라미터 = 항상 활성화 + Routed Experts 중 활성화 부분
    active_params = always_active_params + active_routed_expert_params
    activation_rate = active_params / total_params if total_params > 0 else 0.0
    
    result = {
        'total_params': total_params,
        'active_params': active_params,
        'activation_rate': activation_rate,
        'breakdown': {
            'embedding': embedding_params,
            'attention': attention_params,
            'shared_experts': shared_expert_params,
            'dense_mlp': dense_mlp_params,
            'routed_experts_total': routed_expert_params,
            'routed_experts_active': active_routed_expert_params,
            'global_router': global_router_params,
            'router': router_params,
            'norm': norm_params,
            'lm_head': lm_head_params,
            'vision': vision_params,
            'other': other_params,
        },
        'config': {
            'n_routed_experts': n_routed_experts,
            'n_shared_experts': n_shared_experts,
            'num_experts_per_tok': num_experts_per_tok,
            'num_moe_layers': num_moe_layers,
            'first_k_dense_replace': first_k_dense_replace,
            'num_hidden_layers': num_hidden_layers,
        }
    }
    
    if verbose:
        print("\n" + "="*80)
        print("MoE Model Active Parameter Analysis")
        print("="*80)
        print(f"\n📊 Configuration:")
        print(f"  - Total Hidden Layers: {num_hidden_layers}")
        print(f"  - MoE Layers: {num_moe_layers} (starting from layer {first_k_dense_replace})")
        print(f"  - Routed Experts: {n_routed_experts}")
        print(f"  - Shared Experts per Layer: {n_shared_experts}")
        print(f"  - Active Experts per Token (top_k): {num_experts_per_tok}")
        print(f"  - Expert Activation Ratio: {activation_ratio:.4f} ({num_experts_per_tok}/{n_routed_experts})")
        
        print(f"\n📈 Parameter Breakdown:")
        print(f"  Total Parameters:           {format_parameters(total_params):>15} (100.00%)")
        print(f"  Active Parameters:         {format_parameters(active_params):>15} ({activation_rate*100:.2f}%)")
        print(f"\n  Always Active Components:")
        print(f"    - Embedding:              {format_parameters(embedding_params):>15} ({embedding_params/total_params*100:.2f}%)")
        print(f"    - Attention:              {format_parameters(attention_params):>15} ({attention_params/total_params*100:.2f}%)")
        print(f"    - Dense MLP:              {format_parameters(dense_mlp_params):>15} ({dense_mlp_params/total_params*100:.2f}%)")
        print(f"    - Shared Experts:         {format_parameters(shared_expert_params):>15} ({shared_expert_params/total_params*100:.2f}%)")
        print(f"    - Global Router:          {format_parameters(global_router_params):>15} ({global_router_params/total_params*100:.2f}%)")
        print(f"    - Local Router:           {format_parameters(router_params):>15} ({router_params/total_params*100:.2f}%)")
        print(f"    - LayerNorm:               {format_parameters(norm_params):>15} ({norm_params/total_params*100:.2f}%)")
        print(f"    - LM Head:                 {format_parameters(lm_head_params):>15} ({lm_head_params/total_params*100:.2f}%)")
        print(f"    - Vision Tower:            {format_parameters(vision_params):>15} ({vision_params/total_params*100:.2f}%)")
        print(f"    - Other:                   {format_parameters(other_params):>15} ({other_params/total_params*100:.2f}%)")
        print(f"\n  Routed Experts (Sparse Activation):")
        print(f"    - Total Routed Experts:    {format_parameters(routed_expert_params):>15} ({routed_expert_params/total_params*100:.2f}%)")
        print(f"    - Active Routed Experts:  {format_parameters(active_routed_expert_params):>15} ({active_routed_expert_params/total_params*100:.2f}%)")
        print(f"      (Only {activation_ratio*100:.2f}% of routed experts are active)")
        
        print(f"\n💡 Key Insight:")
        print(f"  During inference:")
        print(f"    - All layers are activated (attention, MLP, shared experts, router, etc.)")
        print(f"    - Only Routed Experts use sparse activation: {activation_ratio*100:.2f}% per token")
        print(f"    - Overall active parameters: {activation_rate*100:.2f}% of total")
        print(f"    - Inactive parameters: {format_parameters(total_params - active_params)}")
        print(f"    - Efficiency: {format_parameters(active_params)} active / {format_parameters(total_params)} total")
        print("="*80 + "\n")
    
    return result

def test_train_forward():
    """Tests the model's forward pass in training mode."""
    print("\n--- Starting Training Mode Forward Pass Test ---")

    # 1. Load model for training test
    print("Loading model for training test...")
    train_test_model = model_architecture.from_pretrained(
        pretrained_model_name_or_path=base_model_name,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_3",
    ).to("cuda")
    train_test_model.train()
    print("Model loaded and set to training mode.")

    # 2. Create dummy inputs
    print("Creating dummy inputs...")
    tokenizer = AutoProcessor.from_pretrained(base_model_name, use_fast=True)
    
    try:
        image_size = train_test_model.config.vision_config.image_size
    except AttributeError:
        print("Could not find vision_config.image_size, defaulting to a standard size.")
        image_size = 224 # A common default

    test_input = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in Korean."},
                    # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                    # {"type": "image", "url": "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg"},
                    {"type": "image"}
                ]
            }
        ],
        # tokenize=True,
        add_generation_prompt=True,
        # return_tensors="pt",
        # return_dict=True,
    )

    # Load images
    # image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
    image = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")
    
    inputs = tokenizer(
        text=test_input.replace("<bos>", "")[:-1],
        images=image,
        return_tensors="pt").to(train_test_model.device)

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
        
    # Create labels for loss calculation
    inputs["labels"] = inputs.input_ids.clone()
    print(f"Inputs created with input_ids shape: {inputs.input_ids.shape}")

    # 3. Forward and backward pass
    try:
        print("Performing forward pass...")
        outputs = train_test_model(**inputs)
        loss = outputs.loss
        print(f"Forward pass successful. Loss: {loss.item()}")

        print("Performing backward pass...")
        loss.backward()
        print("Backward pass successful.")

        # 4. Check for gradients
        grad_found = False
        for name, param in train_test_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"Gradient found for parameter: {name}. Mean grad: {param.grad.abs().mean().item()}")
                grad_found = True
                break
        if not grad_found:
            print("Warning: No gradients were found after backward pass. Check model setup.")

    except Exception as e:
        print(f"An error occurred during forward/backward pass: {e}")
        import traceback
        traceback.print_exc()

    print("--- Training Mode Forward Pass Test Finished ---\n")


def main():
    logging.getLogger("transformers.processing_utils").setLevel(logging.WARN)
    test_model = model_architecture.from_pretrained(
        pretrained_model_name_or_path=base_model_name,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_3",
        # trust_remote_code=True,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_quant_storage=torch.bfloat16)
        ).to("cuda").eval()
    # test_model = PeftModel.from_pretrained(test_model, "/mnt/disks/local-ssd/training_logs/outputs/")
    tokenizer = AutoProcessor.from_pretrained("google/gemma-3-4b-it", use_fast=True)
    with open("/home/conan/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
        tokenizer.chat_template = f.read()
    # logging.set_verbosity_warning()
    test_text = f"""
안녕하세요.<end_of_turn>
<start_of_turn>system
You are a helpful assistant named Sparkle.
Always answer in shortest possible sentence.
But you should remember... Try to answer with Korean.😉<end_of_turn>
<start_of_turn>user
this is the test text message. now you must instruct the model to generate a response to this message.<end_of_turn>
""" * 1

    test_input = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": test_text.strip()}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in Korean."},
                    # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                    # {"type": "image", "url": "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg"},
                    {"type": "image"}
                ]
            }
        ],
        # tokenize=True,
        add_generation_prompt=True,
        # return_tensors="pt",
        # return_dict=True,
    )
    sample_image_urls = [
        "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg",
        "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
        "https://i.namu.wiki/i/6OBqKb_V51DWQbN4UZ6VpeBgRNnBoyivWjd5DqWHvgnMDTAFaDtYhri0zafTw1mESEkNgx1NiHE9XZlhSUrP-r3_Ahetkd9FtLY01RvEWJisz_7Qyx8832b_HZeK6YghWHtY9sPn7WQlYAz9wJ11ew.webp",
        "https://ocr.space/Content/Images/table-ocr-original.webp",
        "https://storage.googleapis.com/kagglesdsdata/datasets/3419046/6067948/images/1058.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20251028%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251028T235242Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=8d70ef579e163c8648a49345cd458dc69f2242de2e5f7a52821f9a12856237ea8d042e6a172bbb7c4e31f8b0f982815fa53347aab2ead5b3dae39a21df0297c202ea2f3d6e05d98d61fd2260a22dd98700bb7de3c7b32232b8a7fbd1909462226bc0f5e6fd3af80adffd04b5bb2145ad16656d1a2bfa1ea02b6b8515f3f1881e4ac9a71e97e134f638f3a58db111c55a4f25abd81973edce796e1d531eff09222e86669b46d8bc9c766838062164083ba20d1feb253748313f496f78623d78bf65f30605e5a2f8e38c3fb506b6fcaf74924735639244c4003fd9dff7ec580ff26d792fb693593cb3f0b4a88f68f599f0b8937e77da68e02939a30308fcdcccec"
    ]
    ran_image = random.choice(sample_image_urls)
    image = load_image(ran_image)
    
    inputs = tokenizer(
        text=test_input.replace("<bos>", "")[:-1],
        images=image,
        return_tensors="pt").to(test_model.device)
    
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    logging.getLogger("transformers.processing_utils").setLevel(logging.INFO)
    print(test_model)
    # print(test_model.config)
    print(format_parameters(test_model.num_parameters()))
    print("Test Sequence Length:", inputs.input_ids.shape[1])
    
    # 활성화 파라미터 측정
    print("\n" + "="*80)
    print("Measuring Active Parameters During Inference")
    print("="*80)
    active_param_info = count_active_parameters(test_model, verbose=True)

    with torch.inference_mode():
        # torch._dynamo.config.capture_dynamic_output_shape_ops = True
        fast_inputs =tokenizer(text="What's poppin?", return_tensors="pt")
        del fast_inputs["token_type_ids"]
        response =tokenizer.batch_decode(
            test_model.generate(
                **fast_inputs.to(test_model.device),
                generation_config=GenerationConfig(
                    device=test_model.device,
                ),
                tokenizer=tokenizer
            )
        )[0]
        print(response)
        response = tokenizer.batch_decode(
            test_model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    device=test_model.device,
                    # max_new_tokens=10,
                    # do_sample=True,
                    # top_p=0.9,
                    # top_k=1,
                    # temperature=0.7,
                    # repetition_penalty=1.2,
                    # length_penalty=1.0,
                    # num_beams=1,
                    # num_beam_groups=1,
                    # num_beam_hyps=1
                    ),
                tokenizer=tokenizer
                )
            )[0]
        print(test_text)
        print("--- Model Response ---")
        print(response[len(test_text):].split("<start_of_turn>model\n")[-1])


def check_params_diff():
    
    model_1 = G3MoEForCausalLM.from_pretrained(base_model_name, config=model_config, dtype=torch.bfloat16) # G3MoEForCausalLM.from_pretrained(base_model_name, config=model_config, torch_dtype=torch.bfloat16)
    model_2 = G3MoEForConditionalGeneration.from_pretrained(base_model_name, config=model_config, dtype=torch.bfloat16)
    vision_model = Gemma3Model.from_pretrained(base_model_name, dtype=torch.bfloat16)

    def architecture_diff_check(item):
        return "moe" not in item and "mlp" not in item and "router" not in item

    def compare_model_weights(model_a, model_b, prefix_a="model_1", prefix_b="model_2"):
        """
        Compare the weights of two torch.nn.Module models and print the names of layers with different weights.
        """
        print("Start comparing weights of the two models...")
        state_dict_a = model_a.state_dict()
        state_dict_b = model_b.state_dict()

        # Find common keys
        common_keys = set(state_dict_a.keys()) & set(state_dict_b.keys())
        different_layers = []
        progress_bar =tqdm(sorted(common_keys), total=len(common_keys), desc="Comparing weights")
        for key in progress_bar:
            progress_bar.set_description(f"Comparing weights: {key}")
            tensor_a = state_dict_a[key]
            tensor_b = state_dict_b[key]
            if not torch.allclose(tensor_a, tensor_b, atol=1e-6, rtol=1e-5):
                different_layers.append(key)

        if different_layers:
            print("Layers with different weights:")
            for layer in different_layers:
                print(f" - {layer}")
        else:
            print("All weights match between the two models.")

        # Optionally, print keys only in one model
        only_in_a = set(item for item in state_dict_a.keys() - state_dict_b.keys() if architecture_diff_check(item))
        only_in_b = set(item for item in state_dict_b.keys() - state_dict_a.keys() if architecture_diff_check(item))
        if only_in_a:
            print(f"\nKeys only in {prefix_a}({len(only_in_a)}):")
            # for k in sorted(only_in_a):
            #     print(f" - {k}")
        if only_in_b:
            print(f"\nKeys only in {prefix_b}({len(only_in_b)}):")
            # for k in sorted(only_in_b):
            #     print(f" - {k}")

    # Compare the weights of the two models
    compare_model_weights(
        model_a=vision_model, 
        model_b=model_2.model, 
        prefix_a="BaseModel", 
        prefix_b="ConditionalGeneration")
    compare_model_weights(
        model_a=vision_model.vision_tower, 
        model_b=model_2.vision_tower, 
        prefix_a="VisionModel", 
        prefix_b="VisionModel_loaded")
    compare_model_weights(
        model_a=vision_model.language_model, 
        model_b=model_1.model, 
        prefix_a="VisionModel", 
        prefix_b="VisionModel_loaded")


if __name__ == "__main__":
    # check_params_diff()
    test_train_forward()
    main()
    # model_1 = G3MoEForConditionalGeneration.from_pretrained(base_model_name, config=model_config, dtype=torch.bfloat16)
    # model_2 = Gemma3ForConditionalGeneration.from_pretrained(base_model_name, dtype=torch.bfloat16)
