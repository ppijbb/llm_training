# coding=utf-8
"""
GramSpec MoE 논문 검증을 위한 분석 도구

이 모듈은 GramSpec MoE의 핵심 주장을 검증하기 위한 다양한 지표를 제공합니다:
1. Expert Specialization Metrics: Expert들이 실제로 다른 기능을 수행하는지
2. Gram Matrix Quality: Gram matrix의 직교성 수준
3. Routing Decision Quality: 라우팅 결정의 품질
4. Sequential Context Utilization: GRU 기반 sequential routing의 효과
5. Computational Efficiency: 계산 오버헤드 분석
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json


class GramSpecAnalyzer:
    """
    GramSpec MoE의 핵심 주장을 검증하기 위한 분석 도구
    
    논문에서 검증해야 할 주요 포인트:
    1. Gram matrix 기반 orthogonal constraint가 expert diversity를 향상시키는가?
    2. Orthogonal expression projection이 expert specialization을 촉진하는가?
    3. Sequential routing (GRU)이 context-aware routing을 가능하게 하는가?
    4. Domain scoring (cosine similarity + speciality penalty)이 효과적인가?
    """
    
    def __init__(self, num_experts: int, router_dim: int = 128):
        self.num_experts = num_experts
        self.router_dim = router_dim
        self.reset()
    
    def reset(self):
        """분석 결과 초기화"""
        self.expert_activation_patterns = defaultdict(list)
        self.gram_matrix_history = []
        self.expression_projection_history = []
        self.routing_decision_history = []
        self.cosine_similarity_history = []
        self.speciality_penalty_history = []
        
    def analyze_routing_step(
        self,
        routing_logits: torch.Tensor,  # [batch*seq, num_experts, router_dim]
        expression_logits: torch.Tensor,  # [batch*seq, num_experts, router_dim]
        selected_experts: torch.Tensor,  # [batch*seq, top_k]
        routing_weights: torch.Tensor,  # [batch*seq, top_k]
        speciality_penalty: float,
        cosine_similarities: torch.Tensor,  # [batch, seq, num_experts]
        expression_loss: float,
        expert_outputs: Optional[Dict[int, torch.Tensor]] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        단일 routing step 분석
        
        Returns:
            분석 결과 딕셔너리
        """
        metrics = {}
        
        # 1. Gram Matrix Quality Metrics
        gram_metrics = self._compute_gram_matrix_quality(routing_logits)
        metrics.update(gram_metrics)
        
        # 2. Expert Specialization Metrics
        specialization_metrics = self._compute_expert_specialization(
            routing_logits, expression_logits, selected_experts, routing_weights
        )
        metrics.update(specialization_metrics)
        
        # 3. Routing Decision Quality
        routing_quality = self._compute_routing_decision_quality(
            selected_experts, routing_weights, cosine_similarities
        )
        metrics.update(routing_quality)
        
        # 4. Expression Projection Effectiveness
        expression_metrics = self._compute_expression_projection_quality(
            expression_logits, routing_logits, cosine_similarities
        )
        metrics.update(expression_metrics)
        
        # 5. Speciality Penalty Contribution
        metrics['speciality_penalty'] = float(speciality_penalty)
        metrics['expression_loss'] = float(expression_loss)
        
        # 6. Expert Activation Diversity (if expert outputs available)
        if expert_outputs is not None:
            diversity_metrics = self._compute_expert_output_diversity(expert_outputs)
            metrics.update(diversity_metrics)
        
        # 7. Context Utilization (if hidden states available)
        if hidden_states is not None:
            context_metrics = self._compute_context_utilization(hidden_states, selected_experts)
            metrics.update(context_metrics)
        
        # 히스토리 저장
        self.gram_matrix_history.append(gram_metrics.get('gram_matrix_orthogonality', 0.0))
        self.speciality_penalty_history.append(speciality_penalty)
        self.cosine_similarity_history.append(cosine_similarities.mean().item())
        
        return metrics
    
    def _compute_gram_matrix_quality(self, routing_logits: torch.Tensor) -> Dict[str, float]:
        """
        Gram matrix의 품질 지표 계산
        
        검증 포인트: Gram matrix가 identity matrix에 가까워질수록 expert들이 orthogonal한가?
        """
        # routing_logits: [batch*seq, num_experts, router_dim]
        # 또는 [batch, seq, num_experts, router_dim]
        
        if routing_logits.dim() == 3:
            # [batch*seq, num_experts, router_dim] -> [num_experts, router_dim] (평균)
            routing_mean = routing_logits.mean(dim=0)  # [num_experts, router_dim]
        elif routing_logits.dim() == 4:
            # [batch, seq, num_experts, router_dim] -> [num_experts, router_dim] (평균)
            routing_mean = routing_logits.mean(dim=[0, 1])  # [num_experts, router_dim]
        else:
            raise ValueError(f"Unexpected routing_logits shape: {routing_logits.shape}")
        
        # Normalize
        routing_normalized = F.normalize(routing_mean, dim=-1)  # [num_experts, router_dim]
        
        # Compute Gram matrix: G = R @ R^T
        gram_matrix = torch.matmul(routing_normalized, routing_normalized.T)  # [num_experts, num_experts]
        
        # Identity matrix
        identity = torch.eye(self.num_experts, device=gram_matrix.device, dtype=gram_matrix.dtype)
        
        # Orthogonality metrics
        gram_deviation = gram_matrix - identity
        gram_orthogonality = 1.0 - torch.norm(gram_deviation, p='fro') / (self.num_experts * np.sqrt(2))
        
        # Diagonal dominance: diagonal elements should be close to 1
        diagonal_quality = gram_matrix.diag().mean().item()
        
        # Off-diagonal sparsity: off-diagonal elements should be close to 0
        off_diagonal_mask = ~torch.eye(self.num_experts, dtype=bool, device=gram_matrix.device)
        off_diagonal_mean = gram_matrix[off_diagonal_mask].abs().mean().item()
        
        # Condition number (for numerical stability)
        try:
            eigenvals = torch.linalg.eigvalsh(gram_matrix)
            condition_number = (eigenvals.max() / eigenvals.min()).item() if eigenvals.min() > 1e-8 else float('inf')
        except:
            condition_number = float('inf')
        
        return {
            'gram_matrix_orthogonality': gram_orthogonality.item(),
            'gram_diagonal_quality': diagonal_quality,
            'gram_off_diagonal_sparsity': off_diagonal_mean,
            'gram_condition_number': condition_number,
            'gram_frobenius_norm': torch.norm(gram_matrix, p='fro').item(),
        }
    
    def _compute_expert_specialization(
        self,
        routing_logits: torch.Tensor,
        expression_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Expert specialization 지표 계산
        
        검증 포인트: 각 expert가 서로 다른 input pattern에 반응하는가?
        """
        # Expert별 activation pattern 분석
        # selected_experts: [batch*seq, top_k]
        # routing_weights: [batch*seq, top_k]
        
        selected_flat = selected_experts.flatten()  # [batch*seq * top_k]
        weights_flat = routing_weights.flatten()  # [batch*seq * top_k]
        
        # Expert별 weighted activation count
        expert_activation_counts = torch.zeros(self.num_experts, device=selected_experts.device)
        expert_activation_weights = torch.zeros(self.num_experts, device=selected_experts.device)
        
        for expert_idx in range(self.num_experts):
            mask = (selected_flat == expert_idx)
            expert_activation_counts[expert_idx] = mask.sum().float()
            expert_activation_weights[expert_idx] = weights_flat[mask].sum()
        
        # Activation distribution entropy (higher = more balanced)
        activation_probs = expert_activation_weights / (expert_activation_weights.sum() + 1e-8)
        activation_entropy = -(activation_probs * torch.log(activation_probs + 1e-8)).sum()
        max_entropy = np.log(self.num_experts)
        normalized_entropy = activation_entropy / max_entropy
        
        # Expert specialization score: 각 expert가 고유한 표현 공간을 사용하는가?
        # Expression projection과 routing logits 간의 cosine similarity 분석
        if routing_logits.dim() == 3:
            routing_mean = routing_logits.mean(dim=0)  # [num_experts, router_dim]
            expression_mean = expression_logits.mean(dim=0)  # [num_experts, router_dim]
        elif routing_logits.dim() == 4:
            routing_mean = routing_logits.mean(dim=[0, 1])  # [num_experts, router_dim]
            expression_mean = expression_logits.mean(dim=[0, 1])  # [num_experts, router_dim]
        else:
            routing_mean = routing_logits
            expression_mean = expression_logits
        
        # Expert 간 cosine similarity (낮을수록 더 specialized)
        routing_normalized = F.normalize(routing_mean, dim=-1)
        expert_similarity_matrix = torch.matmul(routing_normalized, routing_normalized.T)
        off_diagonal_mask = ~torch.eye(self.num_experts, dtype=bool, device=expert_similarity_matrix.device)
        expert_similarity_mean = expert_similarity_matrix[off_diagonal_mask].mean()
        
        # Expression-routing alignment: 각 expert의 expression과 routing이 얼마나 일치하는가?
        expression_normalized = F.normalize(expression_mean, dim=-1)
        alignment_scores = F.cosine_similarity(expression_normalized, routing_normalized, dim=-1)
        alignment_mean = alignment_scores.mean()
        
        return {
            'expert_activation_entropy': normalized_entropy.item(),
            'expert_similarity_mean': expert_similarity_mean.item(),
            'expert_routing_expression_alignment': alignment_mean.item(),
            'expert_utilization_rate': (expert_activation_counts > 0).float().mean().item(),
        }
    
    def _compute_routing_decision_quality(
        self,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        cosine_similarities: torch.Tensor,
    ) -> Dict[str, float]:
        """
        라우팅 결정의 품질 지표 계산
        
        검증 포인트: Domain scoring (cosine similarity + speciality penalty)이 효과적인가?
        """
        # Routing confidence: 선택된 expert들의 가중치 분포
        routing_entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1).mean()
        max_routing_entropy = np.log(routing_weights.size(-1))
        normalized_routing_entropy = routing_entropy / max_routing_entropy
        
        # Cosine similarity utilization: domain scoring이 cosine similarity를 얼마나 활용하는가?
        # cosine_similarities: [batch, seq, num_experts]
        cosine_mean = cosine_similarities.mean().item()
        cosine_std = cosine_similarities.std().item()
        
        # Top-k selection quality: 선택된 expert들의 cosine similarity가 높은가?
        selected_experts_flat = selected_experts.flatten()  # [batch*seq * top_k]
        batch_size, seq_len = cosine_similarities.shape[:2]
        cosine_flat = cosine_similarities.view(batch_size * seq_len, -1)  # [batch*seq, num_experts]
        
        # 각 선택된 expert의 cosine similarity
        selected_cosine_scores = []
        for i in range(selected_experts.size(0)):
            for j in range(selected_experts.size(1)):
                expert_idx = selected_experts[i, j].item()
                cosine_idx = i % (batch_size * seq_len)
                if cosine_idx < cosine_flat.size(0):
                    selected_cosine_scores.append(cosine_flat[cosine_idx, expert_idx].item())
        
        selected_cosine_mean = np.mean(selected_cosine_scores) if selected_cosine_scores else 0.0
        
        return {
            'routing_confidence': normalized_routing_entropy.item(),
            'cosine_similarity_mean': cosine_mean,
            'cosine_similarity_std': cosine_std,
            'selected_expert_cosine_mean': selected_cosine_mean,
        }
    
    def _compute_expression_projection_quality(
        self,
        expression_logits: torch.Tensor,
        routing_logits: torch.Tensor,
        cosine_similarities: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Expression projection의 효과성 지표 계산
        
        검증 포인트: Orthogonal expression projection이 expert specialization을 촉진하는가?
        """
        # Expression projection의 orthogonal property
        if expression_logits.dim() == 3:
            expression_mean = expression_logits.mean(dim=0)  # [num_experts, router_dim]
        elif expression_logits.dim() == 4:
            expression_mean = expression_logits.mean(dim=[0, 1])  # [num_experts, router_dim]
        else:
            expression_mean = expression_logits
        
        expression_normalized = F.normalize(expression_mean, dim=-1)
        expression_gram = torch.matmul(expression_normalized, expression_normalized.T)
        expression_identity = torch.eye(self.num_experts, device=expression_gram.device, dtype=expression_gram.dtype)
        expression_orthogonality = 1.0 - torch.norm(expression_gram - expression_identity, p='fro') / (self.num_experts * np.sqrt(2))
        
        # Expression-routing complementarity: 서로 다른 정보를 담고 있는가?
        if routing_logits.dim() == 3:
            routing_mean = routing_logits.mean(dim=0)
        elif routing_logits.dim() == 4:
            routing_mean = routing_logits.mean(dim=[0, 1])
        else:
            routing_mean = routing_logits
        
        routing_normalized = F.normalize(routing_mean, dim=-1)
        expression_routing_similarity = F.cosine_similarity(expression_normalized, routing_normalized, dim=-1).mean()
        
        # Complementarity score: 낮을수록 expression과 routing이 서로 다른 정보를 담고 있음
        complementarity = 1.0 - expression_routing_similarity
        
        return {
            'expression_projection_orthogonality': expression_orthogonality.item(),
            'expression_routing_complementarity': complementarity.item(),
        }
    
    def _compute_expert_output_diversity(self, expert_outputs: Dict[int, torch.Tensor]) -> Dict[str, float]:
        """
        Expert output의 다양성 지표 계산
        
        검증 포인트: 각 expert가 실제로 다른 출력을 생성하는가?
        """
        if not expert_outputs:
            return {}
        
        expert_outputs_list = []
        for expert_idx in sorted(expert_outputs.keys()):
            output = expert_outputs[expert_idx]  # [num_tokens, hidden_dim]
            expert_outputs_list.append(output.mean(dim=0))  # [hidden_dim]
        
        if len(expert_outputs_list) < 2:
            return {}
        
        expert_outputs_tensor = torch.stack(expert_outputs_list)  # [num_experts, hidden_dim]
        expert_outputs_normalized = F.normalize(expert_outputs_tensor, dim=-1)
        
        # Expert output similarity matrix
        output_similarity = torch.matmul(expert_outputs_normalized, expert_outputs_normalized.T)
        off_diagonal_mask = ~torch.eye(len(expert_outputs_list), dtype=bool, device=output_similarity.device)
        output_diversity = 1.0 - output_similarity[off_diagonal_mask].mean().item()
        
        return {
            'expert_output_diversity': output_diversity,
        }
    
    def _compute_context_utilization(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Sequential context 활용도 지표 계산
        
        검증 포인트: GRU 기반 sequential routing이 context를 활용하는가?
        """
        # Sequential patterns: 연속된 토큰들이 같은 expert를 선택하는 경향
        # selected_experts: [batch*seq, top_k]
        batch_seq_len = selected_experts.size(0)
        
        if batch_seq_len < 2:
            return {}
        
        # 첫 번째 expert 선택 패턴
        first_expert = selected_experts[:, 0]  # [batch*seq]
        
        # Sequential consistency: 연속된 토큰들이 같은 expert를 선택하는 비율
        sequential_changes = (first_expert[1:] != first_expert[:-1]).sum().float()
        sequential_consistency = 1.0 - (sequential_changes / (batch_seq_len - 1))
        
        return {
            'sequential_routing_consistency': sequential_consistency.item(),
        }
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """전체 학습 과정에 대한 집계 지표"""
        if not self.gram_matrix_history:
            return {}
        
        return {
            'avg_gram_orthogonality': np.mean(self.gram_matrix_history),
            'std_gram_orthogonality': np.std(self.gram_matrix_history),
            'avg_speciality_penalty': np.mean(self.speciality_penalty_history),
            'avg_cosine_similarity': np.mean(self.cosine_similarity_history),
        }
    
    def save_analysis(self, filepath: str):
        """분석 결과 저장"""
        analysis_data = {
            'aggregated_metrics': self.get_aggregated_metrics(),
            'gram_matrix_history': self.gram_matrix_history,
            'speciality_penalty_history': self.speciality_penalty_history,
            'cosine_similarity_history': self.cosine_similarity_history,
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)

