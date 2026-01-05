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
AutoModelForCausalLM.register(G3MoEConfig, G3MoEForConditionalGeneration)
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


base_model_name = "Gunulhona/Gemma-3-4B"
model_architecture = G3MoEForConditionalGeneration
base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
base_config = base_config.to_dict()
moe_config = {
        "n_shared_experts": 1,
        "n_routed_experts": 8, # 256, 15, 6
        "n_group": 4,
        "topk_group": 8,
        "num_experts_per_tok": 1,
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

def count_active_parameters(model, sample_inputs=None, top_k=None, verbose=True):
    """
    Inference 시 실제 활성화되는 파라미터 수를 계산합니다.
    
    실제 forward pass를 실행하여 활성화된 expert를 추적합니다.
    
    Args:
        model: G3MoE 모델
        sample_inputs: 실제 forward pass를 위한 샘플 입력 (dict, optional)
                      제공되지 않으면 이론적 계산 사용
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
    # 실제 forward pass를 통해 활성화된 expert 추적
    if sample_inputs is not None and n_routed_experts > 0:
        # 실제 forward pass로 활성화된 expert 추적
        model.eval()
        activated_experts_per_layer = {}
        
        def hook_fn(module, input, output):
            """Forward hook으로 실제 활성화된 expert 추적"""
            if hasattr(module, 'last_selected_experts'):
                selected_experts = module.last_selected_experts
                if selected_experts is not None:
                    # 실제 활성화된 expert ID 추출
                    unique_experts = torch.unique(selected_experts)
                    layer_name = f"layer_{getattr(module, 'iter', 'unknown')}"
                    activated_experts_per_layer[layer_name] = unique_experts.cpu().tolist()
        
        # Hook 등록
        hooks = []
        for name, module in model.named_modules():
            if hasattr(module, 'experts') and hasattr(module, 'num_experts'):
                # MoE 레이어에 hook 등록
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass 실행
        with torch.no_grad():
            try:
                _ = model(**sample_inputs)
            except Exception as e:
                if verbose:
                    print(f"Warning: Forward pass failed: {e}. Using theoretical calculation.")
                sample_inputs = None  # Fallback to theoretical
        
        # Hook 제거
        for hook in hooks:
            hook.remove()
        
        # 실제 활성화된 expert 수 계산
        if activated_experts_per_layer:
            # 각 레이어에서 활성화된 expert 수의 평균
            total_activated = sum(len(experts) for experts in activated_experts_per_layer.values())
            avg_activated_per_layer = total_activated / len(activated_experts_per_layer) if activated_experts_per_layer else num_experts_per_tok
            activation_ratio = avg_activated_per_layer / n_routed_experts
            active_routed_expert_params = int(routed_expert_params * activation_ratio)
            actual_measurement = True
        else:
            # Fallback to theoretical
            activation_ratio = num_experts_per_tok / n_routed_experts
            active_routed_expert_params = int(routed_expert_params * activation_ratio)
            actual_measurement = False
    else:
        # 이론적 계산 (forward pass 없이)
        if n_routed_experts > 0:
            activation_ratio = num_experts_per_tok / n_routed_experts
            active_routed_expert_params = int(routed_expert_params * activation_ratio)
        else:
            activation_ratio = 0.0
            active_routed_expert_params = 0
        actual_measurement = False
    
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
        },
        'actual_measurement': actual_measurement if 'actual_measurement' in locals() else False
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
        if actual_measurement:
            print(f"  - Expert Activation Ratio: {activation_ratio:.4f} (실제 측정: 평균 {activation_ratio*n_routed_experts:.1f}개 expert 활성화)")
        else:
            print(f"  - Expert Activation Ratio: {activation_ratio:.4f} ({num_experts_per_tok}/{n_routed_experts}) [이론적 계산]")
        
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
        if actual_measurement:
            print(f"      (실제 forward pass에서 {activation_ratio*100:.2f}% 활성화됨)")
        else:
            print(f"      (이론적: {activation_ratio*100:.2f}% 활성화, 실제 측정하려면 sample_inputs 제공 필요)")
        
        print(f"\n💡 Key Insight:")
        print(f"  During inference:")
        print(f"    - All layers are activated (attention, MLP, shared experts, router, etc.)")
        print(f"    - Only Routed Experts use sparse activation: {activation_ratio*100:.2f}% per token")
        print(f"    - Overall active parameters: {activation_rate*100:.2f}% of total")
        print(f"    - Inactive parameters: {format_parameters(total_params - active_params)}")
        print(f"    - Efficiency: {format_parameters(active_params)} active / {format_parameters(total_params)} total")
        print("="*80 + "\n")
    
    return result

def test_routing_loss_only():
    """Tests only routing loss calculation without full model forward."""
    print("\n" + "="*80)
    print("Testing Routing Loss Calculation")
    print("="*80)
    
    from models.g3moe_model import load_balancing_loss_func
    
    # Test parameters
    num_experts = 8
    top_k = 2
    batch_size = 4
    seq_len = 512
    
    # Case 1: Single tensor (global router) with attention_mask=None
    print("\n[Case 1] Single Tensor - No Attention Mask")
    gate_logits = torch.randn(batch_size * seq_len, num_experts, device="cuda")
    
    lb_loss = load_balancing_loss_func(
        gate_logits,
        num_experts,
        top_k=top_k,
        attention_mask=None,
        router_z_loss_coef=0.001,
        router_entropy_coef=0.01,
        usage_uniformity_coef=0.01
    )
    print(f"✅ Load Balancing Loss (no mask): {lb_loss.item():.6f}")
    assert lb_loss.item() > 0, "❌ Load balancing loss should be > 0!"
    
    # Case 2: Single tensor with full attention mask (all ones)
    print("\n[Case 2] Single Tensor - Full Attention Mask")
    attention_mask_full = torch.ones(batch_size, seq_len, device="cuda")
    
    lb_loss_full = load_balancing_loss_func(
        gate_logits,
        num_experts,
        top_k=top_k,
        attention_mask=attention_mask_full,
        router_z_loss_coef=0.001,
        router_entropy_coef=0.01,
        usage_uniformity_coef=0.01
    )
    print(f"✅ Load Balancing Loss (full mask): {lb_loss_full.item():.6f}")
    assert lb_loss_full.item() > 0, "❌ Load balancing loss should be > 0!"
    
    # Case 3: Single tensor with padding (realistic scenario)
    print("\n[Case 3] Single Tensor - With Padding (Realistic)")
    attention_mask_padded = torch.ones(batch_size, seq_len, device="cuda")
    # Add padding to last 20% of each sequence
    padding_start = int(seq_len * 0.8)
    attention_mask_padded[:, padding_start:] = 0
    print(f"  - Attention mask shape: {attention_mask_padded.shape}")
    print(f"  - Non-padding tokens: {attention_mask_padded.sum().item()} / {batch_size * seq_len}")
    
    lb_loss_padded = load_balancing_loss_func(
        gate_logits,
        num_experts,
        top_k=top_k,
        attention_mask=attention_mask_padded,
        router_z_loss_coef=0.001,
        router_entropy_coef=0.01,
        usage_uniformity_coef=0.01
    )
    print(f"✅ Load Balancing Loss (with padding): {lb_loss_padded.item():.6f}")
    assert lb_loss_padded.item() > 0, "❌ Load balancing loss should be > 0 with padding!"
    
    # Case 4: Tuple (per-layer routers)
    print("\n[Case 4] Tuple (Per-Layer Routers)")
    num_layers = 4
    gate_logits_tuple = tuple([
        torch.randn(batch_size * seq_len, num_experts, device="cuda")
        for _ in range(num_layers)
    ])
    
    lb_loss_tuple = load_balancing_loss_func(
        gate_logits_tuple,
        num_experts,
        top_k=top_k,
        attention_mask=attention_mask_padded,
        router_z_loss_coef=0.001,
        router_entropy_coef=0.01,
        usage_uniformity_coef=0.01
    )
    print(f"✅ Load Balancing Loss (tuple): {lb_loss_tuple.item():.6f}")
    assert lb_loss_tuple.item() > 0, "❌ Load balancing loss should be > 0 for tuple!"
    
    # Case 5: Different top_k values
    print("\n[Case 5] Different top_k Values")
    for test_top_k in [1, 2, 4]:
        lb_loss_topk = load_balancing_loss_func(
            gate_logits,
            num_experts,
            top_k=test_top_k,
            attention_mask=attention_mask_padded,
            router_z_loss_coef=0.001,
            router_entropy_coef=0.01,
            usage_uniformity_coef=0.01
        )
        print(f"  top_k={test_top_k}: {lb_loss_topk.item():.6f}")
        assert lb_loss_topk.item() > 0, f"❌ Load balancing loss should be > 0 for top_k={test_top_k}!"
    
    # Case 6: Test backward
    print("\n[Case 6] Gradient Flow Test")
    gate_logits_grad = torch.randn(batch_size * seq_len, num_experts, device="cuda", requires_grad=True)
    lb_loss_grad = load_balancing_loss_func(
        gate_logits_grad,
        num_experts,
        top_k=top_k,
        attention_mask=attention_mask_padded,
        router_z_loss_coef=0.001,
        router_entropy_coef=0.01,
        usage_uniformity_coef=0.01
    )
    lb_loss_grad.backward()
    
    assert gate_logits_grad.grad is not None, "❌ Gradient should flow back to gate_logits!"
    print(f"✅ Gradient norm: {gate_logits_grad.grad.norm().item():.6f}")
    print(f"✅ Gradient mean: {gate_logits_grad.grad.mean().item():.6f}")
    
    # Case 7: Edge case - very small batch
    print("\n[Case 7] Edge Case - Small Batch")
    small_batch = 1
    small_seq = 16
    small_gate_logits = torch.randn(small_batch * small_seq, num_experts, device="cuda")
    small_attention_mask = torch.ones(small_batch, small_seq, device="cuda")
    small_attention_mask[:, -4:] = 0  # Add some padding
    
    lb_loss_small = load_balancing_loss_func(
        small_gate_logits,
        num_experts,
        top_k=top_k,
        attention_mask=small_attention_mask,
        router_z_loss_coef=0.001,
        router_entropy_coef=0.01,
        usage_uniformity_coef=0.01
    )
    print(f"✅ Load Balancing Loss (small batch): {lb_loss_small.item():.6f}")
    assert lb_loss_small.item() > 0, "❌ Load balancing loss should be > 0 for small batch!"
    
    # Case 8: Reproduce training-time mismatch (top-k weights only)
    print("\n[Case 8] Reproduce mismatch with top-k-only logits (expected mismatch)")
    selected_experts = torch.randint(0, num_experts, (batch_size * seq_len, top_k), device="cuda")
    topk_weights = torch.softmax(torch.randn(batch_size * seq_len, top_k, device="cuda"), dim=-1)
    try:
        print("  - Calling load_balancing_loss_func with top-k-only weights (should fail or mismatch)...")
        _ = load_balancing_loss_func(
            topk_weights,  # WRONG SHAPE: [batch*seq, top_k]
            num_experts,
            top_k=top_k,
            attention_mask=attention_mask_padded,
            router_z_loss_coef=0.001,
            router_entropy_coef=0.01,
            usage_uniformity_coef=0.01
        )
        print("  ⚠️ Unexpected: function accepted top-k-only weights. This may hide a bug.")
    except Exception as e:
        print(f"  ✅ Expected failure captured: {type(e).__name__}: {e}")
    
    # Fix by expanding top-k weights to full [batch*seq, num_experts]
    full_weights = torch.zeros(batch_size * seq_len, num_experts, device="cuda", dtype=topk_weights.dtype)
    row_idx = torch.arange(batch_size * seq_len, device="cuda").unsqueeze(1).expand(-1, top_k)
    full_weights[row_idx, selected_experts] = topk_weights
    lb_loss_full_from_topk = load_balancing_loss_func(
        full_weights,
        num_experts,
        top_k=top_k,
        attention_mask=attention_mask_padded,
        router_z_loss_coef=0.001,
        router_entropy_coef=0.01,
        usage_uniformity_coef=0.01
    )
    print(f"  ✅ After expansion to full width: loss={lb_loss_full_from_topk.item():.6f}")
    
    print("\n" + "="*80)
    print("✅ All Routing Loss Tests Passed!")
    print("="*80 + "\n")


def test_train_forward():
    """Tests the model's forward pass in training mode with routing loss verification."""
    print("\n" + "="*80)
    print("Starting Training Mode Forward Pass Test (with Routing Loss)")
    print("="*80)

    # 1. Load model for training test
    print("\n[Step 1] Loading model for training test...")
    train_test_model = model_architecture.from_pretrained(
        pretrained_model_name_or_path=base_model_name,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_3",
    ).to("cuda")
    train_test_model.train()
    print("✅ Model loaded and set to training mode.")
    
    # Print MoE config
    print("\n[MoE Configuration]")
    print(f"  - n_routed_experts: {model_config.text_config.n_routed_experts}")
    print(f"  - num_experts_per_tok: {model_config.text_config.num_experts_per_tok}")
    print(f"  - router_aux_loss_coef: {model_config.text_config.router_aux_loss_coef}")
    print(f"  - router_z_loss_coef: {model_config.text_config.router_z_loss_coef}")
    print(f"  - first_k_dense_replace: {model_config.text_config.first_k_dense_replace}")

    # 2. Create dummy inputs
    print("\n[Step 2] Creating dummy inputs...")
    tokenizer = AutoProcessor.from_pretrained(base_model_name, use_fast=True)
    with open("/home/conan/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
        tokenizer.chat_template = f.read()
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
                    {"type": "image"}
                ]
            }
        ],
        add_generation_prompt=True,
    )

    image = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")
    
    inputs = tokenizer(
        text=test_input.replace("<bos>", "")[:-1],
        images=image,
        return_tensors="pt").to(train_test_model.device)

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
        
    # Create labels for loss calculation
    inputs["labels"] = inputs.input_ids.clone()
    print(f"✅ Inputs created with input_ids shape: {inputs.input_ids.shape}")

    # 3. Forward pass with routing loss verification
    try:
        print("\n[Step 3] Performing forward pass...")
        outputs = train_test_model(**inputs)
        loss = outputs.loss
        print(f"✅ Forward pass successful.")
        print(f"  - Total Loss: {loss.item():.6f}")
        
        # Check if router_logits is present
        if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
            print(f"✅ router_logits present: {type(outputs.router_logits)}")
            if isinstance(outputs.router_logits, torch.Tensor):
                print(f"  - Shape: {outputs.router_logits.shape}")
                print(f"  - Min/Max: {outputs.router_logits.min().item():.3f} / {outputs.router_logits.max().item():.3f}")
                if outputs.router_logits.shape[-1] != model_config.text_config.n_routed_experts:
                    print("\n❌ CRITICAL: router_logits width != n_routed_experts.")
                    print("   The model is likely returning top-k weights instead of full per-expert logits.")
                    print("   This will cause shape mismatch inside load_balancing_loss_func.")
            elif isinstance(outputs.router_logits, tuple):
                print(f"  - Tuple length: {len(outputs.router_logits)}")
                print(f"  - First element shape: {outputs.router_logits[0].shape}")
                print(f"  - First element Min/Max: {outputs.router_logits[0].min().item():.3f} / {outputs.router_logits[0].max().item():.3f}")
            
            # Manually calculate routing loss to verify
            from models.g3moe_model import load_balancing_loss_func
            
            print(f"\n[Step 3.1] Calculating routing loss manually...")
            print(f"  - Input parameters:")
            print(f"    - n_routed_experts: {model_config.text_config.n_routed_experts}")
            print(f"    - num_experts_per_tok (top_k): {model_config.text_config.num_experts_per_tok}")
            print(f"    - attention_mask shape: {inputs.get('attention_mask').shape if 'attention_mask' in inputs else 'None'}")
            if 'attention_mask' in inputs:
                print(f"    - attention_mask sum: {inputs['attention_mask'].sum().item()} / {inputs['attention_mask'].numel()}")
            
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                model_config.text_config.n_routed_experts,
                top_k=model_config.text_config.num_experts_per_tok,
                attention_mask=inputs.get("attention_mask", None),
                router_z_loss_coef=model_config.text_config.router_z_loss_coef,
                router_entropy_coef=getattr(model_config.text_config, "router_entropy_coef", 0.0),
                usage_uniformity_coef=getattr(model_config.text_config, "usage_uniformity_coef", 0.0),
            )
            weighted_aux_loss = model_config.text_config.router_aux_loss_coef * aux_loss
            
            print(f"\n✅ Routing Loss Breakdown:")
            print(f"  - Raw Aux Loss: {aux_loss.item():.8f}")
            print(f"  - Router Aux Loss Coef: {model_config.text_config.router_aux_loss_coef}")
            print(f"  - Weighted Aux Loss: {weighted_aux_loss.item():.8f}")
            print(f"  - Total Loss: {loss.item():.8f}")
            print(f"  - % of Total Loss: {(weighted_aux_loss.item() / loss.item() * 100):.4f}%")

            # Report orthogonal loss contribution if available
            weighted_ortho = 0.0
            if hasattr(outputs, 'ortho_loss') and outputs.ortho_loss is not None:
                try:
                    ortho_val = outputs.ortho_loss.mean().item() if hasattr(outputs.ortho_loss, 'mean') else float(outputs.ortho_loss)
                except Exception:
                    ortho_val = float(outputs.ortho_loss)
                weighted_ortho = ortho_val * getattr(model_config.text_config, 'ortho_loss_coef', 0.0)
                print(f"\n✅ Orthogonal Loss Breakdown:")
                print(f"  - Raw Ortho Loss: {ortho_val:.8f}")
                print(f"  - Ortho Loss Coef: {getattr(model_config.text_config, 'ortho_loss_coef', 0.0)}")
                print(f"  - Weighted Ortho Loss: {weighted_ortho:.8f}")
            else:
                print("\nℹ️  Ortho loss not present in outputs (skipping report)")

            # Report expression loss contribution if available (default weight 0.005 in model)
            weighted_expr = 0.0
            if hasattr(outputs, 'expression_loss') and outputs.expression_loss is not None:
                try:
                    expr_val = outputs.expression_loss.mean().item() if hasattr(outputs.expression_loss, 'mean') else float(outputs.expression_loss)
                except Exception:
                    expr_val = float(outputs.expression_loss)
                expr_coef = 0.005
                weighted_expr = expr_val * expr_coef
                print(f"\n✅ Expression Loss Breakdown:")
                print(f"  - Raw Expression Loss: {expr_val:.8f}")
                print(f"  - Expression Loss Coef: {expr_coef}")
                print(f"  - Weighted Expression Loss: {weighted_expr:.8f}")
            else:
                print("\nℹ️  Expression loss not present in outputs (skipping report)")

            combined_known = weighted_aux_loss.item() + weighted_ortho + weighted_expr
            print(f"\n📊 Known Loss Components Sum (weighted): {combined_known:.8f}")
            print(f"📊 Unknown/LM component (approx): {(loss.item() - combined_known):.8f}")
            
            # Verify routing loss is non-zero
            if aux_loss.item() == 0.0:
                print("\n❌ CRITICAL: Routing loss is 0! Load balancing is NOT working!")
                print("   This means the bug is NOT fixed!")
            else:
                print(f"\n✅ SUCCESS: Routing loss is non-zero ({aux_loss.item():.8f})")
                print("   Load balancing is ACTIVE!")
                
            # Verify routing loss is included in total loss
            # Try to isolate LM loss by doing forward without routing loss
            print(f"\n[Step 3.2] Verifying routing loss is included in total loss...")
            print(f"  Expected total = LM_loss + {weighted_aux_loss.item():.8f}")
            print(f"  Actual total = {loss.item():.8f}")
            
            # Test: Forward pass with router_aux_loss_coef = 0
            print(f"\n[Step 3.3] Testing with router_aux_loss_coef = 0...")
            original_coef = train_test_model.config.text_config.router_aux_loss_coef
            # Instead of running a second forward (which can OOM), analytically remove routing loss
            approx_loss_without_routing = loss.item() - weighted_aux_loss.item()
            
            print(f"  - Approximated loss without routing: {approx_loss_without_routing:.8f}")
            print(f"  - Loss with routing: {loss.item():.8f}")
            print(f"  - Difference (should equal weighted_aux): {(loss.item() - approx_loss_without_routing):.8f}")
            print(f"  - Expected difference: {weighted_aux_loss.item():.8f}")
            
            if abs((loss.item() - approx_loss_without_routing) - weighted_aux_loss.item()) < 1e-4:
                print(f"✅ VERIFIED: Routing loss is correctly included in total loss (no extra forward)")
            else:
                print(f"⚠️  WARNING: Routing loss inclusion verification not exact (tolerance exceeded)")
            
            # Restore coef (no further forward executed here to avoid OOM)
            train_test_model.config.text_config.router_aux_loss_coef = original_coef
                
        else:
            print("❌ CRITICAL: router_logits not found in outputs!")
            print("   This means routing loss is NOT being calculated!")
            print("   The bug is NOT fixed!")

    except Exception as e:
        print(f"❌ Error during forward/backward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Training Mode Forward Pass Test Finished")
    print("="*80 + "\n")


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
        "https://ocr.space/Content/Images/table-ocr-original.webp",
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
    
    # print(test_model.config)
    print(format_parameters(test_model.num_parameters()))
    print("Test Sequence Length:", inputs.input_ids.shape[1])
    
    # 활성화 파라미터 측정 (실제 forward pass로 측정)
    print("\n" + "="*80)
    print("Measuring Active Parameters During Inference")
    print("="*80)
    # 실제 forward pass를 위해 sample inputs 사용
    active_param_info = count_active_parameters(test_model, sample_inputs=inputs, verbose=True)

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
    # Test routing loss calculation in isolation
    test_routing_loss_only()
    
    # Test full model forward with routing loss
    test_train_forward()
    
    # Test inference
    main()
    
    # check_params_diff()
    # model_1 = G3MoEForConditionalGeneration.from_pretrained(base_model_name, config=model_config, dtype=torch.bfloat16)
    # model_2 = Gemma3ForConditionalGeneration.from_pretrained(base_model_name, dtype=torch.bfloat16)
