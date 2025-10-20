"""
Custom Reward Functions for GRPO Training

This module provides various reward function implementations that can be used
with GRPO training to achieve different learning objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseRewardFunction(ABC):
    """Base class for all reward functions"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def compute_reward(
        self, 
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute reward for given logits"""
        pass
    
    def __call__(self, *args, **kwargs):
        return self.compute_reward(*args, **kwargs)


class SystematicRewardFunction(BaseRewardFunction):
    """
    Systematic Reward Function for GRPO
    
    This reward function implements the systematic approach where rewards
    are computed based on the overall system objectives rather than
    individual performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("systematic", config)
        
        # Systematic reward parameters
        self.consistency_weight = self.config.get("consistency_weight", 0.3)
        self.coherence_weight = self.config.get("coherence_weight", 0.3)
        self.helpfulness_weight = self.config.get("helpfulness_weight", 0.4)
        self.temperature = self.config.get("temperature", 1.0)
        
    def compute_reward(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute systematic reward based on multiple criteria
        
        Args:
            chosen_logits: Logits for chosen responses [batch_size, seq_len, vocab_size]
            rejected_logits: Logits for rejected responses [batch_size, seq_len, vocab_size]
            chosen_attention_mask: Attention mask for chosen responses
            rejected_attention_mask: Attention mask for rejected responses
            
        Returns:
            Reward tensor [batch_size]
        """
        batch_size = chosen_logits.size(0)
        rewards = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            # Compute individual reward components
            consistency_reward = self._compute_consistency_reward(
                chosen_logits[i], rejected_logits[i],
                chosen_attention_mask[i], rejected_attention_mask[i]
            )
            
            coherence_reward = self._compute_coherence_reward(
                chosen_logits[i], rejected_logits[i],
                chosen_attention_mask[i], rejected_attention_mask[i]
            )
            
            helpfulness_reward = self._compute_helpfulness_reward(
                chosen_logits[i], rejected_logits[i],
                chosen_attention_mask[i], rejected_attention_mask[i]
            )
            
            # Combine rewards with weights
            total_reward = (
                self.consistency_weight * consistency_reward +
                self.coherence_weight * coherence_reward +
                self.helpfulness_weight * helpfulness_reward
            )
            
            rewards[i] = total_reward
        
        return rewards
    
    def _compute_consistency_reward(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute consistency reward based on logit stability"""
        # Compute log probabilities
        chosen_log_probs = F.log_softmax(chosen_logits / self.temperature, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits / self.temperature, dim=-1)
        
        # Compute entropy (lower entropy = more consistent)
        chosen_entropy = -torch.sum(chosen_log_probs * torch.exp(chosen_log_probs), dim=-1)
        rejected_entropy = -torch.sum(rejected_log_probs * torch.exp(rejected_log_probs), dim=-1)
        
        # Average over sequence length
        chosen_consistency = torch.mean(chosen_entropy[chosen_mask.bool()])
        rejected_consistency = torch.mean(rejected_entropy[rejected_mask.bool()])
        
        # Reward higher consistency (lower entropy)
        return chosen_consistency - rejected_consistency
    
    def _compute_coherence_reward(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute coherence reward based on sequence smoothness"""
        # Compute log probabilities
        chosen_log_probs = F.log_softmax(chosen_logits / self.temperature, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits / self.temperature, dim=-1)
        
        # Compute sequence smoothness (lower variance = more coherent)
        chosen_variance = torch.var(chosen_log_probs, dim=-1)
        rejected_variance = torch.var(rejected_log_probs, dim=-1)
        
        # Average over sequence length
        chosen_coherence = torch.mean(chosen_variance[chosen_mask.bool()])
        rejected_coherence = torch.mean(rejected_variance[rejected_mask.bool()])
        
        # Reward higher coherence (lower variance)
        return rejected_coherence - chosen_coherence
    
    def _compute_helpfulness_reward(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute helpfulness reward based on response quality"""
        # Compute log probabilities
        chosen_log_probs = F.log_softmax(chosen_logits / self.temperature, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits / self.temperature, dim=-1)
        
        # Compute average log probability (higher = more confident)
        chosen_confidence = torch.mean(chosen_log_probs[chosen_mask.bool()])
        rejected_confidence = torch.mean(rejected_log_probs[rejected_mask.bool()])
        
        # Reward higher confidence
        return chosen_confidence - rejected_confidence


class GroupRelativeRewardFunction(BaseRewardFunction):
    """
    Group Relative Reward Function for GRPO
    
    This reward function implements the group-relative approach where rewards
    are computed based on relative performance within a group rather than
    absolute performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("group_relative", config)
        
        self.group_size = self.config.get("group_size", 4)
        self.relative_weight = self.config.get("relative_weight", 0.7)
        self.absolute_weight = self.config.get("absolute_weight", 0.3)
        self.temperature = self.config.get("temperature", 1.0)
        
    def compute_reward(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        group_indices: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute group-relative reward
        
        Args:
            group_indices: Group indices for each sample [batch_size]
        """
        batch_size = chosen_logits.size(0)
        rewards = torch.zeros(batch_size, device=self.device)
        
        # If no group indices provided, treat each sample as its own group
        if group_indices is None:
            group_indices = torch.arange(batch_size, device=self.device)
        
        # Compute absolute rewards first
        absolute_rewards = self._compute_absolute_rewards(
            chosen_logits, rejected_logits,
            chosen_attention_mask, rejected_attention_mask
        )
        
        # Compute relative rewards within groups
        for group_id in torch.unique(group_indices):
            group_mask = (group_indices == group_id)
            group_chosen = chosen_logits[group_mask]
            group_rejected = rejected_logits[group_mask]
            group_chosen_mask = chosen_attention_mask[group_mask]
            group_rejected_mask = rejected_attention_mask[group_mask]
            
            # Compute relative performance within group
            relative_rewards = self._compute_relative_rewards(
                group_chosen, group_rejected,
                group_chosen_mask, group_rejected_mask
            )
            
            # Combine absolute and relative rewards
            group_absolute = absolute_rewards[group_mask]
            group_rewards = (
                self.relative_weight * relative_rewards +
                self.absolute_weight * group_absolute
            )
            
            rewards[group_mask] = group_rewards
        
        return rewards
    
    def _compute_absolute_rewards(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute absolute rewards based on log probability differences"""
        chosen_log_probs = F.log_softmax(chosen_logits / self.temperature, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits / self.temperature, dim=-1)
        
        # Compute average log probability
        chosen_avg = torch.mean(chosen_log_probs[chosen_mask.bool()], dim=1)
        rejected_avg = torch.mean(rejected_log_probs[rejected_mask.bool()], dim=1)
        
        return chosen_avg - rejected_avg
    
    def _compute_relative_rewards(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute relative rewards within a group"""
        group_size = chosen_logits.size(0)
        relative_rewards = torch.zeros(group_size, device=self.device)
        
        # Compute pairwise comparisons within group
        for i in range(group_size):
            for j in range(group_size):
                if i != j:
                    # Compare chosen[i] vs rejected[j]
                    chosen_i_log_probs = F.log_softmax(chosen_logits[i] / self.temperature, dim=-1)
                    rejected_j_log_probs = F.log_softmax(rejected_logits[j] / self.temperature, dim=-1)
                    
                    chosen_i_avg = torch.mean(chosen_i_log_probs[chosen_mask[i].bool()])
                    rejected_j_avg = torch.mean(rejected_j_log_probs[rejected_mask[j].bool()])
                    
                    relative_rewards[i] += chosen_i_avg - rejected_j_avg
        
        # Normalize by group size
        relative_rewards /= (group_size - 1)
        
        return relative_rewards


class MultiObjectiveRewardFunction(BaseRewardFunction):
    """
    Multi-Objective Reward Function for GRPO
    
    This reward function combines multiple objectives to create a comprehensive
    reward signal that balances different aspects of model performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("multi_objective", config)
        
        # Objective weights
        self.objectives = self.config.get("objectives", {
            "accuracy": 0.3,
            "fluency": 0.2,
            "coherence": 0.2,
            "helpfulness": 0.3
        })
        
        self.temperature = self.config.get("temperature", 1.0)
        
    def compute_reward(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute multi-objective reward"""
        batch_size = chosen_logits.size(0)
        rewards = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            total_reward = 0.0
            
            for objective, weight in self.objectives.items():
                if objective == "accuracy":
                    obj_reward = self._compute_accuracy_reward(
                        chosen_logits[i], rejected_logits[i],
                        chosen_attention_mask[i], rejected_attention_mask[i]
                    )
                elif objective == "fluency":
                    obj_reward = self._compute_fluency_reward(
                        chosen_logits[i], rejected_logits[i],
                        chosen_attention_mask[i], rejected_attention_mask[i]
                    )
                elif objective == "coherence":
                    obj_reward = self._compute_coherence_reward(
                        chosen_logits[i], rejected_logits[i],
                        chosen_attention_mask[i], rejected_attention_mask[i]
                    )
                elif objective == "helpfulness":
                    obj_reward = self._compute_helpfulness_reward(
                        chosen_logits[i], rejected_logits[i],
                        chosen_attention_mask[i], rejected_attention_mask[i]
                    )
                else:
                    obj_reward = 0.0
                
                total_reward += weight * obj_reward
            
            rewards[i] = total_reward
        
        return rewards
    
    def _compute_accuracy_reward(self, chosen_logits, rejected_logits, chosen_mask, rejected_mask):
        """Compute accuracy-based reward"""
        chosen_log_probs = F.log_softmax(chosen_logits / self.temperature, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits / self.temperature, dim=-1)
        
        chosen_confidence = torch.mean(torch.max(chosen_log_probs, dim=-1)[0][chosen_mask.bool()])
        rejected_confidence = torch.mean(torch.max(rejected_log_probs, dim=-1)[0][rejected_mask.bool()])
        
        return chosen_confidence - rejected_confidence
    
    def _compute_fluency_reward(self, chosen_logits, rejected_logits, chosen_mask, rejected_mask):
        """Compute fluency-based reward"""
        chosen_log_probs = F.log_softmax(chosen_logits / self.temperature, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits / self.temperature, dim=-1)
        
        # Compute perplexity (lower is better)
        chosen_perplexity = torch.exp(-torch.mean(chosen_log_probs[chosen_mask.bool()]))
        rejected_perplexity = torch.exp(-torch.mean(rejected_log_probs[rejected_mask.bool()]))
        
        return rejected_perplexity - chosen_perplexity
    
    def _compute_coherence_reward(self, chosen_logits, rejected_logits, chosen_mask, rejected_mask):
        """Compute coherence-based reward"""
        chosen_log_probs = F.log_softmax(chosen_logits / self.temperature, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits / self.temperature, dim=-1)
        
        # Compute sequence smoothness
        chosen_smoothness = torch.mean(torch.var(chosen_log_probs, dim=-1)[chosen_mask.bool()])
        rejected_smoothness = torch.mean(torch.var(rejected_log_probs, dim=-1)[rejected_mask.bool()])
        
        return rejected_smoothness - chosen_smoothness
    
    def _compute_helpfulness_reward(self, chosen_logits, rejected_logits, chosen_mask, rejected_mask):
        """Compute helpfulness-based reward"""
        chosen_log_probs = F.log_softmax(chosen_logits / self.temperature, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits / self.temperature, dim=-1)
        
        # Compute average log probability
        chosen_avg = torch.mean(chosen_log_probs[chosen_mask.bool()])
        rejected_avg = torch.mean(rejected_log_probs[rejected_mask.bool()])
        
        return chosen_avg - rejected_avg


class RewardFunctionFactory:
    """Factory class for creating reward functions"""
    
    @staticmethod
    def create_reward_function(
        reward_type: str,
        config: Dict[str, Any] = None
    ) -> BaseRewardFunction:
        """Create a reward function of the specified type"""
        
        if reward_type == "systematic":
            return SystematicRewardFunction(config)
        elif reward_type == "group_relative":
            return GroupRelativeRewardFunction(config)
        elif reward_type == "multi_objective":
            return MultiObjectiveRewardFunction(config)
        else:
            raise ValueError(f"Unknown reward function type: {reward_type}")
    
    @staticmethod
    def get_available_reward_functions() -> List[str]:
        """Get list of available reward function types"""
        return ["systematic", "group_relative", "multi_objective"]


# Predefined reward function configurations
REWARD_CONFIGS = {
    "systematic_default": {
        "consistency_weight": 0.3,
        "coherence_weight": 0.3,
        "helpfulness_weight": 0.4,
        "temperature": 1.0
    },
    
    "group_relative_default": {
        "group_size": 4,
        "relative_weight": 0.7,
        "absolute_weight": 0.3,
        "temperature": 1.0
    },
    
    "multi_objective_default": {
        "objectives": {
            "accuracy": 0.3,
            "fluency": 0.2,
            "coherence": 0.2,
            "helpfulness": 0.3
        },
        "temperature": 1.0
    },
    
    "systematic_balanced": {
        "consistency_weight": 0.33,
        "coherence_weight": 0.33,
        "helpfulness_weight": 0.34,
        "temperature": 0.8
    },
    
    "group_relative_aggressive": {
        "group_size": 8,
        "relative_weight": 0.9,
        "absolute_weight": 0.1,
        "temperature": 1.2
    }
}


def create_reward_function(
    reward_type: str = "systematic",
    config_name: str = "default"
) -> BaseRewardFunction:
    """Create a reward function with predefined configuration"""
    
    # Get configuration
    if config_name == "default":
        config_key = f"{reward_type}_default"
    else:
        config_key = f"{reward_type}_{config_name}"
    
    config = REWARD_CONFIGS.get(config_key, {})
    
    # Create reward function
    return RewardFunctionFactory.create_reward_function(reward_type, config)


