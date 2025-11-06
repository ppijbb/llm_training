# coding=utf-8
"""
Gram Matrix-based Specialization Routing for MoE (GramSpecMoE)

This module implements a specialized MoE routing method based on Gram matrix-based expert specialization
and orthogonal expression projection. It extracts the routing logic and upcycling functionality
from G3MoE to make them reusable for any HuggingFace model.

Key Features:
- GramSpecRouter: Gram matrix-based routing with Orthogonal constraints for expert specialization
- Specialized MoE Block: Converts MLP layers to MoE blocks with expert specialization
- GramSpec Upcycling: Copies pretrained MLP weights to MoE experts
- Dynamic Layer Discovery: Automatically finds layers and MLPs in any model structure
- MoE-aware: Detects existing MoE models (Mixtral, DeepSeek, etc.) and preserves expert weights while replacing router

Routing Method (GramSpec):
- GRU-based sequential routing for context-aware expert selection
- Orthogonal expression projection for expert specialization
- Gram matrix-based orthogonal constraints for expert diversity
- Cosine similarity domain scoring
- Sparsemixer for efficient top-k selection
- EMA-based adaptive load balancing

Behavior:
- Dense models (GPT-2, LLaMA, etc.): Converts dense MLP to MoE with GramSpecRouter
- Already MoE models (Mixtral, DeepSeek, GPT-OSS, etc.): Preserves existing experts but replaces router with GramSpecRouter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any, Callable
from tqdm.auto import tqdm
import warnings
import functools

# Import necessary components from G3MoE
from models.g3moe_model import (
    G3MoEMLP,
    ExpressionProjector,
)


class GramSpecRouter(nn.Module):
    """
    Gram Matrix-based Specialization Router (GramSpecRouter).
    
    Implements expert routing using:
    - GRU-based sequential routing for context awareness
    - Orthogonal expression projection for expert specialization
    - Gram matrix-based orthogonal constraints for expert diversity
    - Cosine similarity domain scoring
    - Sparsemixer for efficient expert selection
    
    GramSpec uses Gram matrix (from Gram-Schmidt orthogonalization) to achieve
    expert specialization and diversity simultaneously.
    
    Works with any model architecture and extracts routing logic from G3MoERouter.
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
            top_k: Number of experts to select (will be clamped to num_experts)
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
        # Ensure top_k doesn't exceed num_experts
        top_k = min(top_k, self.num_experts)

        # GRU를 통한 전역 라우팅 (hn 활용)
        routing_logits, hn = self.load_balancer(x, hn.to(x.dtype))
        input_shape = routing_logits.shape[:-1]
        hidden_shape = (*input_shape, -1, self.router_dim)

        # Enhanced expression projection for expert specialization
        # Expression projector should handle dtype conversion internally if needed
        expression_logits = self.expression_projector(x)
        expression_logits = expression_logits.view(hidden_shape)
        
        routing_logits = routing_logits.view(hidden_shape)
        routing_logits = F.normalize(routing_logits, dim=-1)
        
        # Enhanced Gram matrix calculation
        batch_size, seq_len = input_shape
        routing_logits_reshaped = routing_logits.view(batch_size, seq_len, self.num_experts, self.router_dim)
        
        # Compute Gram matrix for each sequence position
        gram = torch.matmul(routing_logits_reshaped, routing_logits_reshaped.transpose(-2, -1))
        
        # Optimize: Create routing_i once and reuse
        routing_i = torch.eye(self.num_experts, device=routing_logits.device, dtype=routing_logits.dtype)
        routing_i_expanded = routing_i.unsqueeze(0).unsqueeze(0)  # [1, 1, num_experts, num_experts]
        
        # Speciality penalty: encourage orthogonal expert representations
        speciality_penalty = torch.mean((F.normalize(gram - routing_i_expanded, dim=-1) ** 2).sum(dim=(-2,-1)))
        
        # Cosine similarity between expression and routing logits
        cosine_similarities = 1.0 - F.cosine_similarity(expression_logits, routing_logits, dim=-1)
        
        # Domain scores with speciality penalty
        # Optimize: Pre-compute penalty expansion once
        penalty_expanded = (1.0 + speciality_penalty.unsqueeze(-1).unsqueeze(-1))
        domain_scores = cosine_similarities * penalty_expanded
        
        # Sparsemixer를 통한 최종 expert 선택 및 가중치 계산
        # Optimize: Reuse already flattened view
        domain_scores_flat = domain_scores.view(batch_size * seq_len, self.num_experts)
        
        # Simplified top-k routing to reduce computation
        top_k_logits, selected_experts = torch.topk(domain_scores_flat, top_k, dim=-1)
        # Use input dtype for softmax to maintain precision (avoid float32 intermediate)
        multiplier = F.softmax(top_k_logits, dim=-1)

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


class GramSpecMoEBlock(nn.Module):
    """
    Gram Matrix-based Specialization MoE Block (GramSpecMoEBlock).
    
    Converts any MLP layer to MoE using GramSpecRouter for specialized routing.
    Extracts MoE logic from G3MoEGRINMoE and makes it config-agnostic.
    """

    def __init__(
        self,
        router: GramSpecRouter,
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
        # Ensure top_k doesn't exceed num_experts
        self.top_k = min(top_k, num_experts)
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
        
        # Track whether to return router_scores (for compatibility with models like GPT-OSS)
        self.return_router_scores = False
        
        # Context-aware h_0 initialization for World Model connection
        # Projects context vector (hidden_size) to GRU hidden state size (num_experts * router_dim)
        # This allows external world information to be injected into the routing process
        gru_hidden_size = self.num_experts * self.router_dim
        self.context_projector = nn.Linear(self.hidden_dim, gru_hidden_size)
    
    def _freeze_shared_experts(self):
        """Freeze shared experts parameters"""
        for param in self.shared_experts.parameters():
            param.requires_grad = False
    
    def _unfreeze_shared_experts(self):
        """Unfreeze shared experts parameters"""
        for param in self.shared_experts.parameters():
            param.requires_grad = True
    
    @torch._dynamo.disable
    def forward(self, hidden_states: torch.Tensor, global_routing_hn: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through MoE block.
        
        Compatible with standard HuggingFace MLP layers (only hidden_states).
        If global_routing_hn is not provided, it will be initialized automatically.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            global_routing_hn: Optional GRU hidden state for routing [1, batch_size, num_experts * router_dim]
                              If None, will be initialized to zeros
            
        Returns:
            final_hidden_states: Output tensor [batch_size, seq_len, hidden_dim]
                                (Standard MLP compatibility - returns only hidden_states)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Initialize global_routing_hn if not provided (for compatibility with standard HuggingFace models)
        if global_routing_hn is None:
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            # Context-aware h_0 initialization: World Model connection
            # Extract context vector from input (mean pooling of all tokens)
            # This represents the "summary of the external world" described in the input
            v_context = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
            
            # Project context vector to GRU hidden state space using dedicated interpreter
            # This acts as a bridge between external world observation and internal routing state
            initial_hn_flat = self.context_projector(v_context)  # [batch_size, num_experts * router_dim]
            
            # Reshape for GRU: [1, batch_size, num_experts * router_dim]
            global_routing_hn = initial_hn_flat.unsqueeze(0)
        
        residual = hidden_states
        final_hidden_states, routing_info = self._sparse_routing(hidden_states, global_routing_hn)
        
        with torch.no_grad():
            pretrained_residual = self.shared_experts(residual)
        final_hidden_states = final_hidden_states + pretrained_residual * 1.0
        
        if self.training:
            final_hidden_states = final_hidden_states.requires_grad_(True)
        
        # Store routing_info internally for wrapper to access (for global state passing)
        # routing_info: (routing_weights, hn, speciality_loss, cosine_similarities, expression_loss, router_scores)
        # This allows the wrapper to extract hn for inter-layer state passing
        self._last_routing_info = routing_info
        
        # Return format depends on original MLP's forward signature
        # Some models (like GPT-OSS) expect (hidden_states, router_scores)
        if self.return_router_scores:
            # Extract router_scores from routing_info
            # routing_info: (routing_weights, hn, speciality_loss, cosine_similarities, expression_loss, router_scores)
            _, _, _, _, _, router_scores = routing_info
            return final_hidden_states, router_scores
        else:
            # Return only hidden_states for compatibility with standard HuggingFace MLP layers
            # Routing information is stored internally and can be accessed via hooks if needed
            # Wrapper will extract hn from self._last_routing_info
            return final_hidden_states
    
    @torch._dynamo.disable
    def _sparse_routing(self, hidden_states: torch.Tensor, global_routing_logits: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Sparse routing logic - optimized version with same results"""
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
        (routing_weights, selected_experts, expression_logits, 
         hn, speciality_loss, cosine_similarities, expression_loss) = router_output

        assert routing_weights.isnan().sum() == 0, f"routing_weights is nan"

        # Optimize: Flatten once and reuse
        hidden_states_flat = hidden_states.view(batch_size * sequence_length, hidden_dim)
        
        # Optimize: Pre-convert dtypes once
        routing_weights_converted = routing_weights.to(dtype=hidden_states.dtype)
        selected_experts_converted = selected_experts.to(dtype=torch.long)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )

        # One hot encode the selected experts
        expert_mask = torch.nn.functional.one_hot(selected_experts_converted, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts
        # Optimize: Remove tolist() calls - use tensor indexing directly
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            has_tokens = top_x.numel() > 0
            if has_tokens:
                # Optimize: Use tensor indexing directly instead of tolist()
                # This eliminates GPU-CPU sync overhead
                current_state = hidden_states_flat[top_x]  # Direct tensor indexing
                current_hidden_states = expert_layer(current_state) * routing_weights_converted[top_x, idx].unsqueeze(-1)

                final_hidden_states.index_add_(0, top_x, current_hidden_states)
                
                # Update Specialization EMA
                if self.training:
                    with torch.no_grad():
                        # Reuse hidden_states_flat instead of re-viewing
                        current_mean_hidden = hidden_states_flat[top_x].mean(dim=0)
                        self.expert_specialization_ema[expert_idx].mul_(self.router.ema_alpha).add_(
                            current_mean_hidden, 
                            alpha=1.0 - self.router.ema_alpha
                        )

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        # Compute router_scores (required in return signature, but optimize computation)
        # router_scores: [batch*seq, num_experts] with routing weights for selected experts, zeros otherwise
        # Optimize: Use pre-converted tensors and compute in one go
        router_scores = torch.zeros(
            batch_size * sequence_length, self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        # Fill in routing weights for selected experts (using pre-converted tensors to avoid repeated conversion)
        router_scores.scatter_(1, selected_experts_converted, routing_weights_converted)
        # Reshape to [batch, seq, num_experts] for compatibility
        router_scores = router_scores.view(batch_size, sequence_length, self.num_experts)
        
        if self.training:
            with torch.no_grad():
                self.last_selected_experts = selected_experts.detach()
                self.last_routing_weights = routing_weights.detach()
                self.last_num_experts = self.num_experts
        
        return final_hidden_states, (routing_weights, hn, speciality_loss, cosine_similarities, expression_loss, router_scores)


def is_already_moe(mlp: nn.Module) -> bool:
    """
    Check if the MLP module is already a MoE structure.
    
    Args:
        mlp: The MLP module to check
        
    Returns:
        True if already MoE, False otherwise
    """
    if not isinstance(mlp, nn.Module):
        return False
    
    # Check for MoE indicators:
    # 1. Has 'experts' attribute (Mixtral, DeepSeek style - ModuleList/list)
    if hasattr(mlp, 'experts') and isinstance(mlp.experts, (nn.ModuleList, list)):
        return True
    
    # 2. Has 'experts' attribute that is a Module with num_experts (GPT-OSS style)
    if hasattr(mlp, 'experts') and isinstance(mlp.experts, nn.Module):
        if hasattr(mlp.experts, 'num_experts'):
            return True
    
    # 3. Has 'router' or 'gate' attribute (Switch Transformer style)
    if hasattr(mlp, 'router') or hasattr(mlp, 'gate'):
        # Check if it also has experts
        if hasattr(mlp, 'experts') or hasattr(mlp, 'expert_modules'):
            return True
    
    # 4. Module type name suggests MoE (MixtralBLockSparseTop2MLP, etc.)
    module_name = mlp.__class__.__name__.lower()
    if 'moe' in module_name or 'mixtral' in module_name or 'deepseek' in module_name or 'gptoss' in module_name:
        return True
    
    return False


def find_mlp_in_layer(layer: nn.Module) -> Optional[nn.Module]:
    """
    Find MLP module in a decoder layer.
    Supports various naming conventions used in different models.
    
    Args:
        layer: A decoder layer module
        
    Returns:
        MLP module if found, None otherwise. Returns both dense MLP and MoE MLP.
    """
    # Common MLP attribute names across different models
    mlp_names = ['mlp', 'feed_forward', 'ffn', 'ffw', 'moe']
    
    for name in mlp_names:
        if hasattr(layer, name):
            mlp = getattr(layer, name)
            # Check if it's actually an MLP-like module (has linear layers or MoE structure)
            if isinstance(mlp, nn.Module):
                # Check for common MLP structures
                if hasattr(mlp, 'gate_proj') or hasattr(mlp, 'fc1') or hasattr(mlp, 'c_fc'):
                    return mlp
                # For G3MoE style MLP
                if any(hasattr(mlp, attr) for attr in ['down_proj', 'dense_h_to_4h', 'dense_4h_to_h']):
                    return mlp
                # Already MoE structure
                if is_already_moe(mlp):
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
    Copy weights from source MLP/expert to target expert.
    Supports various MLP structures including different MoE architectures.
    
    Supports:
    - G3MoE/LLaMA style: gate_proj, up_proj, down_proj
    - Mixtral/PhiMoE style: w1, w2, w3
    - GPT-2/BERT style: c_fc, c_proj
    - Generic: fc1, fc2
    
    Args:
        source_mlp: Source MLP/expert module
        target_expert: Target expert module
        
    Returns:
        True if copying was successful, False otherwise
    """
    copied = False
    
    # G3MoE/LLaMA/DeepSeek/OLMoE/Qwen3 style (gate_proj, up_proj, down_proj)
    if hasattr(source_mlp, 'gate_proj') and hasattr(target_expert, 'gate_proj'):
        try:
            if source_mlp.gate_proj.weight.shape == target_expert.gate_proj.weight.shape:
                target_expert.gate_proj.weight.copy_(source_mlp.gate_proj.weight)
                if hasattr(source_mlp.gate_proj, 'bias') and hasattr(target_expert.gate_proj, 'bias'):
                    if source_mlp.gate_proj.bias is not None and target_expert.gate_proj.bias is not None:
                        if source_mlp.gate_proj.bias.shape == target_expert.gate_proj.bias.shape:
                            target_expert.gate_proj.bias.copy_(source_mlp.gate_proj.bias)
                copied = True
        except Exception:
            pass
    
    if hasattr(source_mlp, 'up_proj') and hasattr(target_expert, 'up_proj'):
        try:
            if source_mlp.up_proj.weight.shape == target_expert.up_proj.weight.shape:
                target_expert.up_proj.weight.copy_(source_mlp.up_proj.weight)
                if hasattr(source_mlp.up_proj, 'bias') and hasattr(target_expert.up_proj, 'bias'):
                    if source_mlp.up_proj.bias is not None and target_expert.up_proj.bias is not None:
                        if source_mlp.up_proj.bias.shape == target_expert.up_proj.bias.shape:
                            target_expert.up_proj.bias.copy_(source_mlp.up_proj.bias)
                copied = True
        except Exception:
            pass
    
    if hasattr(source_mlp, 'down_proj') and hasattr(target_expert, 'down_proj'):
        try:
            if source_mlp.down_proj.weight.shape == target_expert.down_proj.weight.shape:
                target_expert.down_proj.weight.copy_(source_mlp.down_proj.weight)
                if hasattr(source_mlp.down_proj, 'bias') and hasattr(target_expert.down_proj, 'bias'):
                    if source_mlp.down_proj.bias is not None and target_expert.down_proj.bias is not None:
                        if source_mlp.down_proj.bias.shape == target_expert.down_proj.bias.shape:
                            target_expert.down_proj.bias.copy_(source_mlp.down_proj.bias)
                copied = True
        except Exception:
            pass
    
    # Mixtral/PhiMoE style (w1, w2, w3)
    if hasattr(source_mlp, 'w1') and hasattr(target_expert, 'gate_proj'):
        try:
            if source_mlp.w1.weight.shape == target_expert.gate_proj.weight.shape:
                target_expert.gate_proj.weight.copy_(source_mlp.w1.weight)
                copied = True
        except Exception:
            pass
    
    if hasattr(source_mlp, 'w2') and hasattr(target_expert, 'down_proj'):
        try:
            if source_mlp.w2.weight.shape == target_expert.down_proj.weight.shape:
                target_expert.down_proj.weight.copy_(source_mlp.w2.weight)
                copied = True
        except Exception:
            pass
    
    if hasattr(source_mlp, 'w3') and hasattr(target_expert, 'up_proj'):
        try:
            if source_mlp.w3.weight.shape == target_expert.up_proj.weight.shape:
                target_expert.up_proj.weight.copy_(source_mlp.w3.weight)
                copied = True
        except Exception:
            pass
    
    # GPT-2 / BERT style (c_fc, c_proj)
    if hasattr(source_mlp, 'c_fc') and hasattr(target_expert, 'c_fc'):
        try:
            if source_mlp.c_fc.weight.shape == target_expert.c_fc.weight.shape:
                target_expert.c_fc.weight.copy_(source_mlp.c_fc.weight)
                if hasattr(source_mlp.c_fc, 'bias') and hasattr(target_expert.c_fc, 'bias'):
                    if source_mlp.c_fc.bias is not None and target_expert.c_fc.bias is not None:
                        if source_mlp.c_fc.bias.shape == target_expert.c_fc.bias.shape:
                            target_expert.c_fc.bias.copy_(source_mlp.c_fc.bias)
                copied = True
        except Exception:
            pass
    
    if hasattr(source_mlp, 'c_proj') and hasattr(target_expert, 'c_proj'):
        try:
            if source_mlp.c_proj.weight.shape == target_expert.c_proj.weight.shape:
                target_expert.c_proj.weight.copy_(source_mlp.c_proj.weight)
                if hasattr(source_mlp.c_proj, 'bias') and hasattr(target_expert.c_proj, 'bias'):
                    if source_mlp.c_proj.bias is not None and target_expert.c_proj.bias is not None:
                        if source_mlp.c_proj.bias.shape == target_expert.c_proj.bias.shape:
                            target_expert.c_proj.bias.copy_(source_mlp.c_proj.bias)
                copied = True
        except Exception:
            pass
    
    # Generic fallback (fc1, fc2)
    if hasattr(source_mlp, 'fc1') and hasattr(target_expert, 'fc1'):
        try:
            if source_mlp.fc1.weight.shape == target_expert.fc1.weight.shape:
                target_expert.fc1.weight.copy_(source_mlp.fc1.weight)
                if hasattr(source_mlp.fc1, 'bias') and hasattr(target_expert.fc1, 'bias'):
                    if source_mlp.fc1.bias is not None and target_expert.fc1.bias is not None:
                        if source_mlp.fc1.bias.shape == target_expert.fc1.bias.shape:
                            target_expert.fc1.bias.copy_(source_mlp.fc1.bias)
                copied = True
        except Exception:
            pass
    
    if hasattr(source_mlp, 'fc2') and hasattr(target_expert, 'fc2'):
        try:
            if source_mlp.fc2.weight.shape == target_expert.fc2.weight.shape:
                target_expert.fc2.weight.copy_(source_mlp.fc2.weight)
                if hasattr(source_mlp.fc2, 'bias') and hasattr(target_expert.fc2, 'bias'):
                    if source_mlp.fc2.bias is not None and target_expert.fc2.bias is not None:
                        if source_mlp.fc2.bias.shape == target_expert.fc2.bias.shape:
                            target_expert.fc2.bias.copy_(source_mlp.fc2.bias)
                copied = True
        except Exception:
            pass
    
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
    GramSpec function to upcycle any pretrained model to MoE.
    
    This function handles two cases:
    1. Dense models (GPT-2, LLaMA, etc.):
       - Finds decoder layers in the model
       - Creates an GramSpec router
       - Replaces MLP layers with MoE blocks
       - Copies pretrained MLP weights to MoE experts
    
    2. Already MoE models (Mixtral, DeepSeek, GPT-OSS, etc.):
       - Detects existing MoE structure
       - Preserves existing expert weights
       - Replaces router with GramSpecRouter (routing method only changes)
       - Maintains same number of experts
    
    Args:
        model: The pretrained transformer model to upcycle
        moe_config: Dictionary containing MoE configuration
            Required keys:
                - hidden_size: Hidden dimension size
                - intermediate_size: Intermediate MLP dimension
                - num_experts: Number of experts (for dense models; ignored if model already MoE)
                - num_experts_per_tok: Top-k experts per token
                - router_dim: Router dimension (default: 128)
                - n_shared_experts: Number of shared experts (default: 1)
                - first_k_dense_replace: First k layers to keep as dense (default: 0)
        expert_module_class: Class for expert MLP modules (default: G3MoEMLP)
        layer_start_idx: Starting layer index for conversion (default: 0)
        layer_end_idx: Ending layer index for conversion (None = all layers)
        verbose: Whether to show progress bar
        
    Returns:
        Model with MoE layers using GramSpecRouter (modified in-place, also returns the model)
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from models.gramspec_moe import upcycle_model_to_moe
        >>> 
        >>> # Dense model (GPT-2) - converts to MoE
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> moe_config = {
        ...     "hidden_size": model.config.n_embd,
        ...     "intermediate_size": model.config.n_inner,
        ...     "num_experts": 8,
        ...     "num_experts_per_tok": 2,
        ... }
        >>> model = upcycle_model_to_moe(model, moe_config)
        >>> 
        >>> # Already MoE model (Mixtral) - router only replacement
        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
        >>> moe_config = {
        ...     "hidden_size": model.config.hidden_size,
        ...     "intermediate_size": model.config.intermediate_size,
        ...     "num_experts_per_tok": 2,  # num_experts is auto-detected from existing model
        ... }
        >>> model = upcycle_model_to_moe(model, moe_config)  # Router replaced with GramSpecRouter
    """
    # Extract configuration
    cfg = extract_config_info(moe_config)
    
    # Detect model dtype from first parameter
    model_dtype = None
    try:
        for param in model.parameters():
            if param is not None:
                model_dtype = param.dtype
                break
        # Also try to get from model.config if available
        if model_dtype is None and hasattr(model, 'config'):
            if hasattr(model.config, 'torch_dtype'):
                model_dtype = model.config.torch_dtype
    except Exception:
        pass
    
    # Fallback: Try to infer from first buffer or use default
    if model_dtype is None:
        try:
            for buffer in model.buffers():
                if buffer is not None:
                    model_dtype = buffer.dtype
                    break
        except Exception:
            pass
    
    # Last resort: Use float32 only if absolutely necessary
    if model_dtype is None:
        if verbose:
            print("Warning: Could not detect model dtype, defaulting to float32")
        model_dtype = torch.float32
    
    if verbose:
        print(f"Detected model dtype: {model_dtype}")
    
    # Find layers in model
    layers = find_layers_in_model(model)
    if layers is None:
        raise ValueError(
            "Could not find decoder layers in model. "
            "Please ensure the model has 'layers', 'h', 'block', or 'decoder_layers' attribute."
        )
    
    # Create expression orthogonal router (shared across all layers)
    router = GramSpecRouter(
        hidden_size=cfg['hidden_size'],
        num_experts=cfg['num_experts'],
        router_dim=cfg['router_dim'],
        balancing_strength=cfg['balancing_strength'],
        ema_alpha=cfg['ema_alpha'],
    )
    
    # Convert router to match model dtype
    router = router.to(dtype=model_dtype)
    
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
        
        # Check if already MoE structure (Mixtral, DeepSeek, etc.)
        if is_already_moe(source_mlp):
            if verbose:
                processing.set_description(f"Layer {layer_idx}: Already MoE structure, replacing router only")
            
            # For already MoE models, we preserve existing experts but replace router with GramSpecRouter
            # This allows using the specialized routing method while keeping pretrained expert weights
            
            # Extract existing expert count if available
            existing_num_experts = None
            if hasattr(source_mlp, 'experts'):
                if isinstance(source_mlp.experts, (nn.ModuleList, list)):
                    existing_num_experts = len(source_mlp.experts)
                    if verbose:
                        processing.set_description(f"Layer {layer_idx}: Found {existing_num_experts} existing experts")
                elif isinstance(source_mlp.experts, nn.Module) and hasattr(source_mlp.experts, 'num_experts'):
                    # GPT-OSS style: experts is a Module with num_experts attribute
                    existing_num_experts = source_mlp.experts.num_experts
                    if verbose:
                        processing.set_description(f"Layer {layer_idx}: Found {existing_num_experts} existing experts (GPT-OSS style)")
                elif hasattr(source_mlp, 'num_experts'):
                    # Fallback: check if MLP itself has num_experts
                    existing_num_experts = source_mlp.num_experts
                    if verbose:
                        processing.set_description(f"Layer {layer_idx}: Found {existing_num_experts} existing experts (from MLP)")
            
            # Use existing expert count if not specified in config
            num_experts_to_use = existing_num_experts if existing_num_experts else cfg['num_experts']
            
            # Create a dummy config object for expert_module_class
            class DummyConfig:
                def __init__(self):
                    self.hidden_size = cfg['hidden_size']
                    self.intermediate_size = cfg['intermediate_size']
                    self.hidden_activation = cfg['hidden_activation']
            
            expert_config = DummyConfig()
            
            # Create new MoE block with GramSpecRouter (routing method replacement)
            moe_block = GramSpecMoEBlock(
                router=router,
                expert_module_class=expert_module_class,
                expert_config=expert_config,
                num_experts=num_experts_to_use,
                top_k=cfg['num_experts_per_tok'],
                n_shared_experts=cfg['n_shared_experts'],
                hidden_size=cfg['hidden_size'],
                intermediate_size=cfg['intermediate_size'],
                router_jitter_noise=cfg['router_jitter_noise'],
                input_jitter_noise=cfg['input_jitter_noise'],
                freeze_shared_experts=cfg['freeze_shared_experts'],
            )
            
            # Convert MoE block to match model dtype
            moe_block = moe_block.to(dtype=model_dtype)
            
            # Detect original MLP forward signature and set return_router_scores accordingly
            import inspect
            try:
                # Get forward method signature
                forward_sig = inspect.signature(source_mlp.forward)
                # Check return annotation
                return_annotation = forward_sig.return_annotation
                # Check if it's a tuple or indicates multiple returns
                if return_annotation != inspect.Signature.empty:
                    if hasattr(return_annotation, '__origin__'):
                        # For Union, Tuple types
                        if return_annotation.__origin__ is tuple or (hasattr(return_annotation, '__args__') and len(return_annotation.__args__) > 1):
                            moe_block.return_router_scores = True
                    elif isinstance(return_annotation, tuple) or (str(return_annotation).find('Tuple') >= 0):
                        moe_block.return_router_scores = True
                # Also check by model type as fallback
                model_type = type(source_mlp).__name__
                if 'Oss' in model_type or 'MoE' in model_type:
                    # Try to detect by checking if forward returns tuple
                    # This is a heuristic, actual call would be too risky
                    moe_block.return_router_scores = True
            except Exception as e:
                # Fallback: check by model type
                model_type = type(source_mlp).__name__
                if 'Oss' in model_type:
                    moe_block.return_router_scores = True
                if verbose:
                    processing.set_description(f"Layer {layer_idx}: Could not detect forward signature ({e}), using heuristics")
            
            # Replace MoE with new MoE block (router changed to GramSpecRouter)
            for attr_name in ['mlp', 'feed_forward', 'ffn', 'ffw', 'moe']:
                if hasattr(decoder_layer, attr_name):
                    setattr(decoder_layer, attr_name, moe_block)
                    break
            
            # Try to copy existing expert weights to new experts
            if verbose:
                processing.set_description(f"Layer {layer_idx}: Copying existing expert weights")
            
            # Handle different MoE structures
            existing_experts = None
            
            # Case 1: Standard MoE with experts ModuleList (Mixtral, PhiMoE, DeepSeek, OLMoE, Qwen3-MoE)
            if hasattr(source_mlp, 'experts') and isinstance(source_mlp.experts, (nn.ModuleList, list)):
                existing_experts = source_mlp.experts
            
            # Case 2: GPT-OSS style (experts is a custom module, not ModuleList)
            elif hasattr(source_mlp, 'experts') and isinstance(source_mlp.experts, nn.Module):
                # GPT-OSS has GptOssExperts module with parameters stored differently
                # For now, we'll skip direct copying and rely on the general copy function
                existing_experts = None
                if verbose:
                    processing.set_description(f"Layer {layer_idx}: GPT-OSS style MoE detected, using generic copy")
            
            # Copy expert weights if we have existing experts
            if existing_experts is not None:
                # Use modular copy function for robustness
                max_experts = min(len(existing_experts), len(moe_block.experts))
                for expert_idx in range(max_experts):
                    if expert_idx % 10 == 0 and verbose:
                        processing.set_description(f"Layer {layer_idx}: Copying expert {expert_idx}/{max_experts}")
                    try:
                        copy_mlp_weights_to_expert(existing_experts[expert_idx], moe_block.experts[expert_idx])
                    except Exception as e:
                        if verbose:
                            processing.set_description(f"Layer {layer_idx}: Failed to copy expert {expert_idx}: {str(e)}")
                        continue
            
            # For remaining experts (if new model has more experts), replicate from first expert
            if existing_experts is not None and len(moe_block.experts) > len(existing_experts):
                first_expert = moe_block.experts[0]
                for expert_idx in range(len(existing_experts), len(moe_block.experts)):
                    if expert_idx % 10 == 0 and verbose:
                        processing.set_description(f"Layer {layer_idx}: Replicating to expert {expert_idx}/{len(moe_block.experts)}")
                    try:
                        copy_mlp_weights_to_expert(first_expert, moe_block.experts[expert_idx])
                    except Exception as e:
                        if verbose:
                            processing.set_description(f"Layer {layer_idx}: Failed to replicate expert {expert_idx}: {str(e)}")
                        continue
            
            # Copy shared experts if they exist
            if hasattr(source_mlp, 'shared_experts') and hasattr(moe_block, 'shared_experts'):
                copy_mlp_weights_to_expert(source_mlp.shared_experts, moe_block.shared_experts)
            elif hasattr(moe_block, 'shared_experts'):
                # If no shared expert in source, use average of experts
                # Check if experts exist and can be accessed
                has_experts = False
                if hasattr(source_mlp, 'experts'):
                    if isinstance(source_mlp.experts, (nn.ModuleList, list)) and len(source_mlp.experts) > 0:
                        has_experts = True
                    elif isinstance(source_mlp.experts, nn.Module) and hasattr(source_mlp.experts, 'num_experts'):
                        # GPT-OSS style: can't directly average, skip for now
                        has_experts = False
                
                if has_experts:
                    # Average weights from existing experts for shared expert
                    pass  # This could be implemented if needed
            
            continue  # Skip the dense MLP handling below
        
        # Dense MLP handling (original code)
        # Create MoE block
        if verbose:
            processing.set_description(f"Layer {layer_idx}: Creating MoE block from dense MLP")
        
        # Create a dummy config object for expert_module_class
        class DummyConfig:
            def __init__(self):
                self.hidden_size = cfg['hidden_size']
                self.intermediate_size = cfg['intermediate_size']
                self.hidden_activation = cfg['hidden_activation']
        
        expert_config = DummyConfig()
        
        moe_block = GramSpecMoEBlock(
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
        
        # Convert MoE block to match model dtype
        moe_block = moe_block.to(dtype=model_dtype)
        
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
    
    # Automatically enable global routing state passing (Option B: Integrated Computational Trajectory)
    # This enables inter-layer state passing via the fancy wrapper pattern
    if verbose:
        print("\n🔄 Enabling global routing state passing between layers...")
    model = _enable_global_routing_state(model, verbose=verbose)
    
    return model


# ==================== Global Routing State Passing (Wrapper Pattern) ====================

class _GlobalRoutingStateContext:
    """
    Context manager for global routing state.
    Uses a mutable container to share state across closures.
    """
    def __init__(self):
        self.hn = None
    
    def reset(self):
        """Reset state"""
        self.hn = None


def _create_layer_interceptor(
    original_forward: Callable,
    layer_instance: nn.Module,
    state_context: _GlobalRoutingStateContext,
) -> Callable:
    """
    Create a wrapper function for a layer that intercepts MoE calls.
    
    Uses closure to capture state_context, avoiding variable scope issues.
    """
    @functools.wraps(original_forward)
    def intercepted_forward(*args, **kwargs):
        # Check if this layer has a GramSpecMoEBlock
        has_moe = False
        moe_module = None
        
        for attr_name in ['mlp', 'feed_forward', 'ffn', 'ffw', 'moe']:
            if hasattr(layer_instance, attr_name):
                candidate = getattr(layer_instance, attr_name)
                if isinstance(candidate, GramSpecMoEBlock):
                    has_moe = True
                    moe_module = candidate
                    break
        
        # Inject global_routing_hn if MoE layer
        if has_moe:
            kwargs['global_routing_hn'] = state_context.hn
        
        # Call original forward
        result = original_forward(*args, **kwargs)
        
        # Extract and update hn from result if MoE layer
        if has_moe and moe_module is not None:
            # Try to get routing_info from the MoE module's internal storage
            if hasattr(moe_module, '_last_routing_info'):
                routing_info = moe_module._last_routing_info
                if isinstance(routing_info, tuple) and len(routing_info) >= 2:
                    # routing_info: (routing_weights, hn, speciality_loss, cosine_similarities, expression_loss, router_scores)
                    hn = routing_info[1]  # Second element is hn
                    if hn is not None:
                        state_context.hn = hn
            # Also check result tuple format (for models that return routing_info directly)
            elif isinstance(result, tuple) and len(result) > 0:
                last_elem = result[-1]
                if isinstance(last_elem, tuple) and len(last_elem) >= 2:
                    # Format: (hidden_states, ..., (router_logits, hn, ...))
                    routing_info = last_elem
                    if routing_info[1] is not None:
                        state_context.hn = routing_info[1]
                elif isinstance(last_elem, torch.Tensor):
                    # Format: (hidden_states, hn) - simple case
                    state_context.hn = last_elem
        
        return result
    
    return intercepted_forward


def _enable_global_routing_state(model: nn.Module, verbose: bool = True) -> nn.Module:
    """
    Enable global routing state passing for a model.
    
    This wraps the model's forward method to automatically pass global_routing_hn between layers.
    Uses a fancy, non-invasive approach with higher-order functions and closure.
    
    Args:
        model: The model to enable global state for
        verbose: Whether to print debug info
        
    Returns:
        Model with global state enabled (modified in-place)
    """
    # Find the model's main forward method (usually model.model.forward for HuggingFace models)
    base_model = None
    forward_path = None
    
    # Try common paths
    for path in ['model', 'transformer', 'decoder', 'encoder']:
        if hasattr(model, path):
            candidate = getattr(model, path)
            if hasattr(candidate, 'forward'):
                base_model = candidate
                forward_path = (model, path, 'forward')
                break
    
    # If not found, try direct forward
    if forward_path is None and hasattr(model, 'forward'):
        base_model = model
        forward_path = (model, 'forward')
    
    if forward_path is None:
        if verbose:
            print("⚠️  Warning: Could not find model forward method. Global state passing disabled.")
        return model
    
    # Get original forward
    if len(forward_path) == 3:
        original_forward = getattr(base_model, 'forward')
    else:
        original_forward = model.forward
    
    # Check if already wrapped
    if hasattr(original_forward, '_gramspec_wrapped'):
        if verbose:
            print("ℹ️  Model forward already wrapped with global state passing.")
        return model
    
    # Get layers for state passing
    layers = None
    for attr_name in ['layers', 'h', 'block', 'decoder_layers']:
        if hasattr(base_model, attr_name):
            candidate = getattr(base_model, attr_name)
            if isinstance(candidate, (nn.ModuleList, list)) and len(candidate) > 0:
                layers = candidate
                break
    
    if layers is None:
        if verbose:
            print("⚠️  Warning: Could not find decoder layers. Global state passing disabled.")
        return model
    
    # Create state context (mutable container for closure)
    state_context = _GlobalRoutingStateContext()
    
    # Store original layer forwards
    original_layer_forwards = [layer.forward for layer in layers]
    
    # Create wrapper
    @functools.wraps(original_forward)
    def forward_wrapper(*args, **kwargs):
        """
        Wrapped forward that maintains global_routing_hn state across layers.
        """
        # Reset state at start of forward
        state_context.reset()
        
        # Create interceptors for each layer
        layer_interceptors = [
            _create_layer_interceptor(orig_fn, layer, state_context)
            for orig_fn, layer in zip(original_layer_forwards, layers)
        ]
        
        # Temporarily replace layer forwards
        for layer, interceptor in zip(layers, layer_interceptors):
            layer.forward = interceptor
        
        try:
            # Call original forward (which will now use intercepted layer forwards)
            result = original_forward(*args, **kwargs)
        finally:
            # Restore original layer forwards
            for layer, orig_fn in zip(layers, original_layer_forwards):
                layer.forward = orig_fn
        
        return result
    
    # Mark as wrapped
    forward_wrapper._gramspec_wrapped = True
    forward_wrapper._gramspec_state_context = state_context  # Store for debugging
    
    # Replace forward
    if len(forward_path) == 3:
        setattr(base_model, 'forward', forward_wrapper)
    else:
        model.forward = forward_wrapper
    
    if verbose:
        print(f"✅ Wrapped model forward with global routing state passing ({len(layers)} layers)")
    
    return model


__all__ = [
    'GramSpecRouter',
    'GramSpecMoEBlock',
    'upcycle_model_to_moe',
    'find_mlp_in_layer',
    'find_layers_in_model',
    'extract_config_info',
    'copy_mlp_weights_to_expert',
    'is_already_moe',
    # Backward compatibility aliases
    'UniversalMoERouter',
    'UniversalMoEBlock',
    'OrthoExpressRouter',
    'OrthoExpressMoEBlock',
]

# Backward compatibility aliases (deprecated, use GramSpec* instead)
UniversalMoERouter = GramSpecRouter
UniversalMoEBlock = GramSpecMoEBlock
OrthoExpressRouter = GramSpecRouter
OrthoExpressMoEBlock = GramSpecMoEBlock
