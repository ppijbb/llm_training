# MoE 평가 지표 체크리스트

> **⚠️ 중요**: 이 체크리스트는 논문 실험을 위한 필수 지표들을 추적합니다.  
> **참고 문서**: `/home/conan/workspace/llm_training/paperworks/moe_routing_reference_complete.md`

---

## 📊 논문 필수 지표 현황 (2025년 11월 28일 기준)

### ✅ 측정 완료
- **Gram Orthogonality**: 0.94 ✅ (목표: > 0.90, 달성)
  - 구현: `gram_matrix_orthogonality`

### ⚠️ 측정 필요 (구현 완료, 측정 대기)
- **Expert Entropy**: 측정 필요 (목표: ≥ 2.7)
  - 구현: `expert_activation_entropy` (normalized entropy)
- **Sequential Routing Consistency**: 측정 필요 (목표: > 45%)
  - 구현: `sequential_routing_consistency`
- **Load Balance CV**: 현재 0.3 ❌ (목표: < 0.1, 개선 필요)
  - 구현: `load_balancing_coefficient`

### ❌ 구현 필요 (논문 필수 지표)
- **Expert Overlap (Jaccard Similarity)**: 구현 필요 (목표: < 15%)
  - 각 expert pair 간 token set overlap 계산
  - Jaccard similarity: `|Tokens(i) ∩ Tokens(j)| / |Tokens(i) ∪ Tokens(j)|`
- **Routing Consistency (Checkpoint 간)**: 구현 필요 (목표: > 85%)
  - Checkpoint 간 routing 결정 일관성 측정
  - 현재는 sequential consistency만 구현됨
- **Gini Coefficient**: 구현 필요 (목표: < 0.05, LPR: 0.035)
  - Load distribution inequality 측정
- **Min-max Expert Load Ratio**: 구현 필요 (목표: > 0.70, LPR: 0.70)
  - `min(expert_load) / max(expert_load)` 계산

---

## ✅ 구현 완료된 지표들

### 1. Load/Utilisation of Experts ✅
- `expert_token_counts`: Expert별 처리 토큰 수
- `expert_activation_counts`: Expert별 활성화 횟수
- `expert_weighted_counts`: 가중치를 고려한 expert별 토큰 수
- `expert_token_proportions`: Expert별 처리 비율
- `expert_utilization_rate`: 실제로 사용된 expert 비율

### 2. Capacity Factor / Capacity Usage ✅
- `capacity_factor`: Capacity factor (c in c * T/E)
- `ideal_capacity_per_expert`: Expert당 이상적인 capacity
- `capacity_usage`: 최대 expert load / ideal capacity
- `capacity_utilization`: 평균 expert load / ideal capacity

### 3. Routing Sparsity / Number of Experts Activated per Token ✅
- `avg_experts_per_token`: 토큰당 평균 활성화된 expert 수 (top_k)
- `routing_sparsity`: Routing sparsity (1 - avg_experts_per_token / num_experts)
- `num_active_experts`: 실제 활성화된 unique expert 수
- `expert_activation_ratio`: 활성화된 expert 비율

### 4. Expert Choice / Token Choice Routing Metrics ✅
- `routing_type`: "token_choice" (현재 구현)
- `token_choice_entropy`: Token choice의 엔트로피
- `routing_confidence`: Routing 결정의 신뢰도

### 5. Load Balancing Loss / Auxiliary Loss Metrics ✅
- `aux_loss`: Auxiliary loss (Switch Transformer, DeepSpeed MoE)
- `load_variance`: Load variance (정규화)
- `std_tokens_per_expert`: Expert당 토큰 수의 표준편차
- `maxvio`: Maximum violation (Loss-free balancing)
- `normalized_maxvio`: 정규화된 maxvio

### 6. Inference Cost / FLOPs / Utilized Parameters ⚠️
- `num_active_experts`: 활성화된 expert 수
- `expert_activation_ratio`: Expert 활성화 비율
- `total_expert_activations`: 총 expert 활성화 횟수
- `utilization_efficiency`: Utilization efficiency
- ⚠️ **FLOPs 계산**: 별도 `measure_efficiency.py`에서 수행 필요

### 7. Expert Specialization / Diversity Metrics ✅
- `expert_diversity_score`: Expert 다양성 점수
- `expert_similarity_mean`: Expert 간 평균 유사도
- `expert_similarity_std`: Expert 간 유사도 표준편차
- `expert_specialization_strength`: Specialization 강도
- `expert_output_diversity`: Expert output 다양성
- `expert_routing_expression_alignment`: Expression-routing alignment
- `expert_activation_entropy`: Expert activation entropy (normalized) ⚠️ 측정 필요
- ⚠️ **Expert Overlap (Jaccard Similarity)**: 구현 필요 ❌

### 8. Routing Consistency / Locality Metrics ✅
- `sequential_routing_consistency`: Sequential routing 일관성 ⚠️ 측정 필요 (목표: > 45%)
- `top_k_overlap`: 연속 토큰의 top-k expert 겹침 비율
- `routing_locality`: Routing locality (인접 토큰의 유사성)
- `expert_reuse_rate`: Expert 재사용 비율
- ⚠️ **Routing Consistency (Checkpoint 간)**: 구현 필요 ❌ (목표: > 85%)

### 9. Training Convergence Speed / Downstream Task Performance ❌
- ⚠️ **학습 중 측정 필요**: 이 지표는 학습 과정에서 측정해야 함
- 현재는 평가 시점의 지표만 제공

### 10. Fraction of Active Experts / Sparsity Ratio ✅
- `fraction_active_experts`: 활성화된 expert 비율
- `sparsity_ratio`: Sparsity ratio (1 - fraction_active_experts)
- `expert_utilization_rate`: Expert utilization rate

## 📊 추가 지표들

### Load Balancing Metrics
- `load_balancing_coefficient`: CV (Coefficient of Variation) ⚠️ 현재 0.3 ❌ (목표: < 0.1)
- `load_imbalance_ratio`: Load imbalance ratio
- `expert_efficiency`: Expert efficiency
- `lpr`: Layer-wise Performance Ratio
- ⚠️ **Gini Coefficient**: 구현 필요 ❌ (목표: < 0.05, LPR: 0.035)
- ⚠️ **Min-max Expert Load Ratio**: 구현 필요 ❌ (목표: > 0.70, LPR: 0.70)

### Gram Matrix Quality
- `gram_matrix_orthogonality`: Gram matrix 직교성
- `gram_diagonal_quality`: Diagonal quality
- `gram_off_diagonal_sparsity`: Off-diagonal sparsity

## 🔍 확인 방법

모든 지표는 `analyze_routing_step()` 메서드를 통해 계산되며, `get_aggregated_metrics()`에서 집계됩니다.

```python
analyzer = SPECTRAAnalyzer(num_experts=8, router_dim=128)
metrics = analyzer.analyze_routing_step(...)
aggregated = analyzer.get_aggregated_metrics()
```

---

## 📋 구현 우선순위 (논문 필수 지표)

### Phase 1: 즉시 구현 필요 (논문 핵심 지표)
1. **Expert Overlap (Jaccard Similarity)** ❌
   - 목표: < 15%
   - 구현 위치: `_compute_expert_specialization()` 메서드에 추가
   - 계산 방법: 각 expert pair 간 token set의 Jaccard similarity

2. **Gini Coefficient** ❌
   - 목표: < 0.05 (LPR: 0.035)
   - 구현 위치: `_compute_load_balancing_metrics()` 메서드에 추가
   - 계산 방법: Load distribution의 Gini coefficient

3. **Min-max Expert Load Ratio** ❌
   - 목표: > 0.70 (LPR: 0.70)
   - 구현 위치: `_compute_load_balancing_metrics()` 메서드에 추가
   - 계산 방법: `min(expert_load) / max(expert_load)`

### Phase 2: 측정 및 개선 (구현 완료)
1. **Expert Entropy** ⚠️
   - 현재: 측정 필요
   - 목표: ≥ 2.7
   - 구현: `expert_activation_entropy` (이미 구현됨)

2. **Sequential Routing Consistency** ⚠️
   - 현재: 측정 필요
   - 목표: > 45%
   - 구현: `sequential_routing_consistency` (이미 구현됨)

3. **Load Balance CV** ⚠️
   - 현재: 0.3 ❌
   - 목표: < 0.1
   - 구현: `load_balancing_coefficient` (이미 구현됨)
   - 개선 필요: Load balancing 메커니즘 튜닝

### Phase 3: 추가 구현 (향후)
1. **Routing Consistency (Checkpoint 간)** ❌
   - 목표: > 85%
   - 구현 방법: 여러 checkpoint에서 동일 입력에 대한 routing 결정 비교
   - 저장 필요: `routing_decision_history`에 checkpoint 정보 추가

---

## 📊 SOTA 비교 기준

### 실제 보고된 Metrics (2025년 11월 기준)
- **LPR (arxiv:2506.21328)**: 
  - Gini coefficient: 0.035 ✅
  - Min-max expert load ratio: 0.70 ✅
- **Switch Transformer (Fedus et al., 2021)**: 
  - Balanced utilization: 94.8% ✅
- **Expert Choice (Zhou et al., 2022)**: 
  - Training convergence: 2× faster ✅

### SPECTRA 목표값
- Expert Overlap: < 15%
- Gram Orthogonality: > 0.90 ✅ (현재 0.94)
- Expert Entropy: ≥ 2.7
- Load Balance CV: < 0.1 ❌ (현재 0.3)
- Gini Coefficient: < 0.05
- Min-max Expert Load Ratio: > 0.70
- Routing Consistency: > 85%
- Sequential Consistency: > 45%

---

## 🔄 업데이트 로그

- 2025-11-28: 논문 필수 지표 현황 추가, 측정 필요 항목 명시
- 향후 실험 결과에 따라 지속 업데이트 예정

