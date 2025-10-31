# coding=utf-8
"""
Universal MoE Upcycling and Routing Module

This module provides universal functions and classes for converting any pretrained transformer model
into a Mixture of Experts (MoE) model. It extracts the routing logic and upcycling functionality
from G3MoE to make them reusable for any HuggingFace model.

Key Features:
- Universal Router: Works with any model architecture
- Universal MoE Block: Converts MLP layers to MoE blocks
- Universal Upcycling: Copies pretrained MLP weights to MoE experts
- Dynamic Layer Discovery: Automatically finds layers and MLPs in any model structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any, Callable
from tqdm.auto import tqdm
import warnings

# Import necessary components from G3MoE
from models.g3moe_model import (
    G3MoEMLP,
    ExpressionProjector,
)


class UniversalMoERouter(nn.Module):
    """
    Universal Router that can work with any model architecture.
    Extracts routing logic from G3MoERouter and makes it config-agnostic.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        router_dim: int = 128,
        balancing_strength: float = 0.01,
        ema_alpha: float = 0.99,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router_dim = router_dim
        self.balancing_strength = balancing_strength
        self.ema_alpha = ema_alpha
        self.register_buffer("expert_load_ema", torch.zeros(self.num_experts), persistent=True)

        self.load_balancer = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.num_experts * self.router_dim,
            num_layers=1,
            bias=False,
            batch_first=True,
        )
        
        # Global expression projector: hidden_size → router_dim
        self.expression_projector = ExpressionProjector(
            self.hidden_size, 
            self.router_dim, 
            self.num_experts, 
            method='precomputed'
        )

    def forward(self, x, hn, top_k=2, jitter_eps=0.01, training=True):
        """
        Forward pass for routing.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            hn: Hidden state for GRU [1, batch_size, num_experts * router_dim]
            top_k: Number of experts to select
            jitter_eps: Jitter noise for routing
            training: Whether in training mode
            
        Returns:
            multiplier: Routing weights [batch*seq, top_k]
            selected_experts: Selected expert indices [batch*seq, top_k]
            expression_logits: Expression projection logits
            hn: Updated GRU hidden state
            speciality_penalty: Specialization penalty term
            cosine_similarities: Cosine similarity between expression and routing
            expression_loss: Orthogonal loss for expression projector
        """
        # GRU를 통한 전역 라우팅 (hn 활용)
        routing_logits, hn = self.load_balancer(x, hn)
        input_shape = routing_logits.shape[:-1]
        hidden_shape = (*input_shape, -1, self.router_dim)

        # Enhanced expression projection for expert specialization
        expression_logits = self.expression_projector(x)
        expression_logits = expression_logits.view(hidden_shape)
        
        routing_logits = routing_logits.view(hidden_shape)
        routing_logits = F.normalize(routing_logits, dim=-1)
        
        # Enhanced Gram matrix calculation
        batch_size, seq_len = input_shape
        routing_logits_reshaped = routing_logits.view(batch_size, seq_len, self.num_experts, self.router_dim)
        
        # Compute Gram matrix for each sequence position
        gram = torch.matmul(routing_logits_reshaped, routing_logits_reshaped.transpose(-2, -1))
        
        routing_i = torch.eye(self.num_experts, device=routing_logits.device)
        
        # Speciality penalty: encourage orthogonal expert representations
        speciality_penalty = torch.mean((F.normalize(gram - routing_i.unsqueeze(0).unsqueeze(0), dim=-1) ** 2).sum(dim=(-2,-1)))
        
        # Cosine similarity between expression and routing logits
        cosine_similarities = 1.0 - F.cosine_similarity(expression_logits, routing_logits, dim=-1)
        
        # Domain scores with speciality penalty
        domain_scores = cosine_similarities * (1.0 + speciality_penalty.unsqueeze(-1).unsqueeze(-1))
        
        # Sparsemixer를 통한 최종 expert 선택 및 가중치 계산
        domain_scores_flat = domain_scores.view(batch_size * seq_len, self.num_experts)
        
        # Simplified top-k routing to reduce computation
        top_k_logits, selected_experts = torch.topk(domain_scores_flat, top_k, dim=-1)
        multiplier = F.softmax(top_k_logits, dim=-1, dtype=torch.float32).type_as(domain_scores_flat)

        # Compute expression loss for the projection matrix
        expression_loss = self.expression_projector.orthogonal_loss()

        # ---- Adaptive filter logic for load balancing (applied during training) ----
        if self.training:
            with torch.no_grad():
                total_load = self.expert_load_ema.sum()
                if total_load > 0:
                    load_balancing_scores = self.expert_load_ema / total_load
                else:
                    load_balancing_scores = torch.zeros_like(self.expert_load_ema)
                
                adjustment = load_balancing_scores * self.balancing_strength * self.num_experts
                multiplier = multiplier - adjustment[selected_experts]

                current_load = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=[0, 1]).float()
                self.expert_load_ema.mul_(self.ema_alpha).add_(current_load, alpha=1.0 - self.ema_alpha)

        return multiplier, selected_experts, expression_logits, hn, speciality_penalty, cosine_similarities, expression_loss


class UniversalMoEBlock(nn.Module):
    """
    Universal MoE Block that can wrap any MLP layer.
    Extracts MoE logic from G3MoEGRINMoE and makes it config-agnostic.
    """

    def __init__(
        self,
        router: UniversalMoERouter,
        expert_module_class: type,
        expert_config: Any,
        num_experts: int,
        top_k: int = 2,
        n_shared_experts: int = 1,
        hidden_size: int = None,
        intermediate_size: int = None,
        router_jitter_noise: float = 0.01,
        input_jitter_noise: float = 0.0,
        freeze_shared_experts: bool = True,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_size or getattr(expert_config, 'hidden_size', 768)
        self.ffn_dim = self.hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_dim = router.router_dim
        self.router = router
        self.expert_module_class = expert_module_class
        
        # Create experts
        self.experts = nn.ModuleList([
            expert_module_class(expert_config, intermediate_size=intermediate_size) 
            for _ in range(self.num_experts)
        ])
        
        # Create shared experts
        shared_intermediate_size = (intermediate_size or getattr(expert_config, 'intermediate_size', 3072)) * n_shared_experts
        self.shared_experts = expert_module_class(
            expert_config, 
            intermediate_size=shared_intermediate_size
        )

        self.router_jitter_noise = router_jitter_noise
        self.input_jitter_noise = input_jitter_noise

        # Enhanced Expert Utilization
        self.register_buffer("expert_specialization_ema", torch.zeros(self.num_experts, self.hidden_dim), persistent=True)
        self.routing_temperature = nn.Parameter(torch.ones(1))
        self.specialization_strength = getattr(expert_config, "specialization_strength", 0.01)
        
        # Freeze shared experts if requested
        self.freeze_shared_experts = freeze_shared_experts
        if self.freeze_shared_experts:
            self._freeze_shared_experts()
    
    def _freeze_shared_experts(self):
        """Freeze shared experts parameters"""
        for param in self.shared_experts.parameters():
            param.requires_grad = False
    
    def _unfreeze_shared_experts(self):
        """Unfreeze shared experts parameters"""
        for param in self.shared_experts.parameters():
            param.requires_grad = True
    
    @torch._dynamo.disable
    def forward(self, hidden_states: torch.Tensor, global_routing_hn: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass through MoE block.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            global_routing_hn: GRU hidden state for routing
            
        Returns:
            final_hidden_states: Output tensor [batch_size, seq_len, hidden_dim]
            routing_info: Tuple of (router_logits, hn, speciality_loss, cosine_similarities, expression_loss)
        """
        residual = hidden_states
        final_hidden_states, routing_info = self._sparse_routing(hidden_states, global_routing_hn)
        router_logits, hn, speciality_loss, cosine_similarities, expression_loss = routing_info
        
        with torch.no_grad():
            pretrained_residual = self.shared_experts(residual)
        final_hidden_states = final_hidden_states + pretrained_residual * 1.0
        
        if self.training:
            final_hidden_states = final_hidden_states.requires_grad_(True)
            if router_logits is not None:
                router_logits = router_logits.requires_grad_(True)
        
        return final_hidden_states, (router_logits, hn, speciality_loss, cosine_similarities, expression_loss)
    
    @torch._dynamo.disable
    def _sparse_routing(self, hidden_states: torch.Tensor, global_routing_logits: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Sparse routing logic"""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.input_jitter_noise, 
                1.0 + self.input_jitter_noise
            )
        
        # Global router에서 전체 라우팅 처리
        router_output = self.router(
            hidden_states, 
            global_routing_logits,
            top_k=self.top_k,
            jitter_eps=self.router_jitter_noise,
            training=self.training
        )
        routing_weights, selected_experts, expression_logits, hn, speciality_loss, cosine_similarities, expression_loss = router_output

        assert routing_weights.isnan().sum() == 0, f"routing_weights is nan"

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )

        # One hot encode the selected experts
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            has_tokens = top_x.numel() > 0
            if has_tokens:
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()

                hidden_states_flat = hidden_states.view(batch_size * sequence_length, hidden_dim)
                current_state = hidden_states_flat[top_x_list]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list].unsqueeze(-1)

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
                
                # Update Specialization EMA
                if self.training:
                    with torch.no_grad():
                        hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
                        current_mean_hidden = hidden_states_flat[top_x_list].mean(dim=0)
                        self.expert_specialization_ema[expert_idx].mul_(self.router.ema_alpha).add_(
                            current_mean_hidden, 
                            alpha=1.0 - self.router.ema_alpha
                        )

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        if self.training:
            with torch.no_grad():
                self.last_selected_experts = selected_experts.detach()
                self.last_routing_weights = routing_weights.detach()
                self.last_num_experts = self.num_experts
        
        return final_hidden_states, (routing_weights, hn, speciality_loss, cosine_similarities, expression_loss)


def find_mlp_in_layer(layer: nn.Module) -> Optional[nn.Module]:
    """
    Find MLP module in a decoder layer.
    Supports various naming conventions used in different models.
    
    Args:
        layer: A decoder layer module
        
    Returns:
        MLP module if found, None otherwise
    """
    # Common MLP attribute names across different models
    mlp_names = ['mlp', 'feed_forward', 'ffn', 'ffw', 'moe']
    
    for name in mlp_names:
        if hasattr(layer, name):
            mlp = getattr(layer, name)
            # Check if it's actually an MLP-like module (has linear layers)
            if isinstance(mlp, nn.Module):
                # Check for common MLP structures
                if hasattr(mlp, 'gate_proj') or hasattr(mlp, 'fc1') or hasattr(mlp, 'c_fc'):
                    return mlp
                # For G3MoE style MLP
                if any(hasattr(mlp, attr) for attr in ['down_proj', 'dense_h_to_4h', 'dense_4h_to_h']):
                    return mlp
    
    return None


def find_layers_in_model(model: nn.Module) -> Optional[List[nn.Module]]:
    """
    Find decoder layers in a model.
    Supports various model structures (layers, decoder.layers, model.layers, etc.).
    
    Args:
        model: The transformer model
        
    Returns:
        List of decoder layers if found, None otherwise
    """
    # Common layer attribute names
    layer_names = ['layers', 'h', 'block', 'decoder_layers']
    
    # Try direct access
    for name in layer_names:
        if hasattr(model, name):
            layers = getattr(model, name)
            if isinstance(layers, nn.ModuleList) or isinstance(layers, list):
                return layers
    
    # Try nested access (model.model.layers, model.decoder.layers, etc.)
    for attr_name in ['model', 'decoder', 'transformer', 'encoder']:
        if hasattr(model, attr_name):
            submodel = getattr(model, attr_name)
            for name in layer_names:
                if hasattr(submodel, name):
                    layers = getattr(submodel, name)
                    if isinstance(layers, nn.ModuleList) or isinstance(layers, list):
                        return layers
    
    return None


def extract_config_info(config: Any) -> Dict[str, Any]:
    """
    Extract necessary configuration information from any config object.
    
    Args:
        config: Configuration object (can be dict, PretrainedConfig, or custom config)
        
    Returns:
        Dictionary with extracted configuration values
    """
    if isinstance(config, dict):
        cfg_dict = config
    else:
        # Try to convert to dict
        if hasattr(config, 'to_dict'):
            cfg_dict = config.to_dict()
        elif hasattr(config, '__dict__'):
            cfg_dict = config.__dict__.copy()
        else:
            cfg_dict = {}
    
    # Extract with defaults
    extracted = {
        'hidden_size': cfg_dict.get('hidden_size', cfg_dict.get('d_model', cfg_dict.get('n_embd', 768))),
        'intermediate_size': cfg_dict.get('intermediate_size', cfg_dict.get('ffn_dim', cfg_dict.get('d_ff', 3072))),
        'num_experts': cfg_dict.get('n_routed_experts', cfg_dict.get('num_experts', 8)),
        'num_experts_per_tok': cfg_dict.get('num_experts_per_tok', cfg_dict.get('top_k', 2)),
        'router_dim': cfg_dict.get('router_dim', 128),
        'n_shared_experts': cfg_dict.get('n_shared_experts', 1),
        'first_k_dense_replace': cfg_dict.get('first_k_dense_replace', 0),
        'router_jitter_noise': cfg_dict.get('router_jitter_noise', 0.01),
        'input_jitter_noise': cfg_dict.get('input_jitter_noise', 0.0),
        'freeze_shared_experts': cfg_dict.get('freeze_shared_experts', True),
        'balancing_strength': cfg_dict.get('balancing_strength', 0.01),
        'ema_alpha': cfg_dict.get('ema_alpha', 0.99),
        'hidden_activation': cfg_dict.get('hidden_activation', cfg_dict.get('activation_function', 'gelu')),
    }
    
    return extracted


def copy_mlp_weights_to_expert(source_mlp: nn.Module, target_expert: nn.Module) -> bool:
    """
    Copy weights from source MLP to target expert.
    Supports various MLP structures.
    
    Args:
        source_mlp: Source MLP module
        target_expert: Target expert module
        
    Returns:
        True if copying was successful, False otherwise
    """
    copied = False
    
    # G3MoE style (gate_proj, up_proj, down_proj)
    if hasattr(source_mlp, 'gate_proj') and hasattr(target_expert, 'gate_proj'):
        target_expert.gate_proj.weight.copy_(source_mlp.gate_proj.weight)
        if hasattr(source_mlp.gate_proj, 'bias') and hasattr(target_expert.gate_proj, 'bias'):
            if source_mlp.gate_proj.bias is not None and target_expert.gate_proj.bias is not None:
                target_expert.gate_proj.bias.copy_(source_mlp.gate_proj.bias)
        copied = True
    
    if hasattr(source_mlp, 'up_proj') and hasattr(target_expert, 'up_proj'):
        target_expert.up_proj.weight.copy_(source_mlp.up_proj.weight)
        if hasattr(source_mlp.up_proj, 'bias') and hasattr(target_expert.up_proj, 'bias'):
            if source_mlp.up_proj.bias is not None and target_expert.up_proj.bias is not None:
                target_expert.up_proj.bias.copy_(source_mlp.up_proj.bias)
        copied = True
    
    if hasattr(source_mlp, 'down_proj') and hasattr(target_expert, 'down_proj'):
        target_expert.down_proj.weight.copy_(source_mlp.down_proj.weight)
        if hasattr(source_mlp.down_proj, 'bias') and hasattr(target_expert.down_proj, 'bias'):
            if source_mlp.down_proj.bias is not None and target_expert.down_proj.bias is not None:
                target_expert.down_proj.bias.copy_(source_mlp.down_proj.bias)
        copied = True
    
    # GPT-2 / BERT style (c_fc, c_proj)
    if hasattr(source_mlp, 'c_fc') and hasattr(target_expert, 'c_fc'):
        target_expert.c_fc.weight.copy_(source_mlp.c_fc.weight)
        if hasattr(source_mlp.c_fc, 'bias') and hasattr(target_expert.c_fc, 'bias'):
            if source_mlp.c_fc.bias is not None and target_expert.c_fc.bias is not None:
                target_expert.c_fc.bias.copy_(source_mlp.c_fc.bias)
        copied = True
    
    if hasattr(source_mlp, 'c_proj') and hasattr(target_expert, 'c_proj'):
        target_expert.c_proj.weight.copy_(source_mlp.c_proj.weight)
        if hasattr(source_mlp.c_proj, 'bias') and hasattr(target_expert.c_proj, 'bias'):
            if source_mlp.c_proj.bias is not None and target_expert.c_proj.bias is not None:
                target_expert.c_proj.bias.copy_(source_mlp.c_proj.bias)
        copied = True
    
    # LLaMA / OPT style (gate_proj, up_proj, down_proj with different names)
    # Try fc1/fc2 as fallback
    if hasattr(source_mlp, 'fc1') and hasattr(target_expert, 'fc1'):
        target_expert.fc1.weight.copy_(source_mlp.fc1.weight)
        if hasattr(source_mlp.fc1, 'bias') and hasattr(target_expert.fc1, 'bias'):
            if source_mlp.fc1.bias is not None and target_expert.fc1.bias is not None:
                target_expert.fc1.bias.copy_(source_mlp.fc1.bias)
        copied = True
    
    if hasattr(source_mlp, 'fc2') and hasattr(target_expert, 'fc2'):
        target_expert.fc2.weight.copy_(source_mlp.fc2.weight)
        if hasattr(source_mlp.fc2, 'bias') and hasattr(target_expert.fc2, 'bias'):
            if source_mlp.fc2.bias is not None and target_expert.fc2.bias is not None:
                target_expert.fc2.bias.copy_(source_mlp.fc2.bias)
        copied = True
    
    return copied


@torch.no_grad()
def upcycle_model_to_moe(
    model: nn.Module,
    moe_config: Dict[str, Any],
    expert_module_class: type = G3MoEMLP,
    layer_start_idx: int = 0,
    layer_end_idx: Optional[int] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Universal function to upcycle any pretrained model to MoE.
    
    This function:
    1. Finds decoder layers in the model
    2. Creates a universal router
    3. Replaces MLP layers with MoE blocks
    4. Copies pretrained MLP weights to MoE experts
    
    Args:
        model: The pretrained transformer model to upcycle
        moe_config: Dictionary containing MoE configuration
            Required keys:
                - hidden_size: Hidden dimension size
                - intermediate_size: Intermediate MLP dimension
                - num_experts: Number of experts
                - num_experts_per_tok: Top-k experts per token
                - router_dim: Router dimension (default: 128)
                - n_shared_experts: Number of shared experts (default: 1)
                - first_k_dense_replace: First k layers to keep as dense (default: 0)
        expert_module_class: Class for expert MLP modules (default: G3MoEMLP)
        layer_start_idx: Starting layer index for conversion (default: 0)
        layer_end_idx: Ending layer index for conversion (None = all layers)
        verbose: Whether to show progress bar
        
    Returns:
        Model with MoE layers (modified in-place, also returns the model)
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from models.universal_moe import upcycle_model_to_moe, UniversalMoERouter
        >>> 
        >>> # Load a pretrained model
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> 
        >>> # Define MoE configuration
        >>> moe_config = {
        ...     "hidden_size": model.config.n_embd,
        ...     "intermediate_size": model.config.n_inner,
        ...     "num_experts": 8,
        ...     "num_experts_per_tok": 2,
        ...     "router_dim": 128,
        ...     "n_shared_experts": 1,
        ...     "first_k_dense_replace": 0,
        ... }
        >>> 
        >>> # Upcycle to MoE
        >>> model = upcycle_model_to_moe(model, moe_config)
    """
    # Extract configuration
    cfg = extract_config_info(moe_config)
    
    # Find layers in model
    layers = find_layers_in_model(model)
    if layers is None:
        raise ValueError(
            "Could not find decoder layers in model. "
            "Please ensure the model has 'layers', 'h', 'block', or 'decoder_layers' attribute."
        )
    
    # Create universal router (shared across all layers)
    router = UniversalMoERouter(
        hidden_size=cfg['hidden_size'],
        num_experts=cfg['num_experts'],
        router_dim=cfg['router_dim'],
        balancing_strength=cfg['balancing_strength'],
        ema_alpha=cfg['ema_alpha'],
    )
    
    # Determine layer range
    if layer_end_idx is None:
        layer_end_idx = len(layers)
    else:
        layer_end_idx = min(layer_end_idx, len(layers))
    
    layer_range = range(layer_start_idx, layer_end_idx)
    
    if verbose:
        processing = tqdm(
            enumerate(layer_range),
            total=len(layer_range),
            desc=f"Upcycling model to MoE",
            leave=False
        )
    else:
        processing = enumerate(layer_range)
    
    for enum_idx, layer_idx in processing:
        decoder_layer = layers[layer_idx]
        
        # Find MLP in layer
        source_mlp = find_mlp_in_layer(decoder_layer)
        if source_mlp is None:
            if verbose:
                processing.set_description(f"Layer {layer_idx}: No MLP found, skipping")
            continue
        
        # Skip if before first_k_dense_replace
        if layer_idx < cfg['first_k_dense_replace']:
            if verbose:
                processing.set_description(f"Layer {layer_idx}: Before first_k_dense_replace, skipping")
            continue
        
        # Create MoE block
        if verbose:
            processing.set_description(f"Layer {layer_idx}: Creating MoE block")
        
        # Create a dummy config object for expert_module_class
        class DummyConfig:
            def __init__(self):
                self.hidden_size = cfg['hidden_size']
                self.intermediate_size = cfg['intermediate_size']
                self.hidden_activation = cfg['hidden_activation']
        
        expert_config = DummyConfig()
        
        moe_block = UniversalMoEBlock(
            router=router,
            expert_module_class=expert_module_class,
            expert_config=expert_config,
            num_experts=cfg['num_experts'],
            top_k=cfg['num_experts_per_tok'],
            n_shared_experts=cfg['n_shared_experts'],
            hidden_size=cfg['hidden_size'],
            intermediate_size=cfg['intermediate_size'],
            router_jitter_noise=cfg['router_jitter_noise'],
            input_jitter_noise=cfg['input_jitter_noise'],
            freeze_shared_experts=cfg['freeze_shared_experts'],
        )
        
        # Replace MLP with MoE block
        # Try different attribute names
        for attr_name in ['mlp', 'feed_forward', 'ffn', 'ffw', 'moe']:
            if hasattr(decoder_layer, attr_name):
                setattr(decoder_layer, attr_name, moe_block)
                break
        
        # Copy weights from source MLP to experts
        if verbose:
            processing.set_description(f"Layer {layer_idx}: Copying weights to experts")
        
        # Copy to shared experts
        if hasattr(moe_block, 'shared_experts'):
            copy_mlp_weights_to_expert(source_mlp, moe_block.shared_experts)
        
        # Copy to routed experts
        if hasattr(moe_block, 'experts'):
            for expert_idx, expert in enumerate(moe_block.experts):
                if expert_idx % 10 == 0 and verbose:
                    processing.set_description(f"Layer {layer_idx}: Copying to expert {expert_idx}/{len(moe_block.experts)}")
                copy_mlp_weights_to_expert(source_mlp, expert)
        
        # Delete original MLP
        if verbose:
            processing.set_description(f"Layer {layer_idx}: Cleaning up")
        
        # Store reference to original MLP name for potential cleanup
        # Note: We keep the MLP in case it's needed for fallback, but mark it
        if hasattr(decoder_layer, 'mlp') and decoder_layer.mlp is not moe_block:
            # Store old MLP reference before replacement
            if not hasattr(decoder_layer, '_original_mlp'):
                decoder_layer._original_mlp = source_mlp
    
    if verbose:
        if isinstance(processing, tqdm):
            processing.set_description("Upcycling completed")
    
    return model


__all__ = [
    'UniversalMoERouter',
    'UniversalMoEBlock',
    'upcycle_model_to_moe',
    'find_mlp_in_layer',
    'find_layers_in_model',
    'extract_config_info',
    'copy_mlp_weights_to_expert',
]
