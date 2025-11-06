# MoE 모델 파라미터 계산 분석: 4B 모델의 활성화 파라미터 논박

## 사용자 주장 분석

**주장**: "4B 모델 기준으로 expert를 2개 쓰면 shared 포함해서 3개가 돌아가니까 12B가 active인거야?"

**결론**: **이 논리는 완전히 틀렸습니다.** 아래에서 상세히 논박합니다.

## MoE 모델의 파라미터 구조

### 1. 전체 모델 파라미터 구성

MoE 모델의 전체 파라미터는 다음과 같이 구성됩니다:

```
전체 파라미터 = 
  + Embedding 레이어
  + Attention 레이어들 (모든 레이어)
  + MoE 레이어들:
    - Router 파라미터
    - Shared Expert 파라미터 (1개)
    - Routed Expert 파라미터 (n_routed_experts 개)
  + LayerNorm 레이어들
  + Output Head (LM Head)
```

### 2. G3MoEMLP 파라미터 계산

```python
class G3MoEMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
```

**단일 Expert MLP 파라미터 수**:
```
Expert_MLP_params = gate_proj + up_proj + down_proj
                 = (hidden_size × intermediate_size) 
                 + (hidden_size × intermediate_size) 
                 + (intermediate_size × hidden_size)
                 = 3 × hidden_size × intermediate_size
```

### 3. MoE 레이어별 파라미터 분포

#### 각 MoE 레이어의 파라미터:
```
MoE_Layer_Params = 
  + Router: hidden_size × n_routed_experts (또는 router_dim 관련)
  + Shared Expert: 3 × hidden_size × (intermediate_size × n_shared_experts)
  + Routed Experts: n_routed_experts × (3 × hidden_size × intermediate_size)
```

#### 예시 (가정):
- `hidden_size = 1024`
- `intermediate_size = 4096`
- `n_routed_experts = 256`
- `n_shared_experts = 1`

```
단일 Expert 파라미터 = 3 × 1024 × 4096 = 12,582,912 ≈ 12.6M
Shared Expert 파라미터 = 3 × 1024 × (4096 × 1) = 12,582,912 ≈ 12.6M
전체 Routed Experts 파라미터 = 256 × 12.6M = 3,225.6M ≈ 3.2B

한 MoE 레이어의 전체 파라미터 ≈ 3.2B + 12.6M ≈ 3.2B
```

### 4. Inference 시 활성화 파라미터

**Inference 시 실제 활성화되는 파라미터**:

```
Active_Params = 
  + Embedding (항상 활성화)
  + Attention 레이어들 (모든 레이어, 항상 활성화)
  + MoE 레이어들:
    - Router (항상 활성화, 작은 비중)
    - Shared Expert (항상 활성화) 
    - 활성화된 Routed Experts (top_k 개만 활성화)
  + LayerNorm (항상 활성화)
  + Output Head (항상 활성화)
```

#### 예시 (top_k=2, n_routed_experts=256):
```
한 MoE 레이어에서:
  - Shared Expert: 12.6M (항상 활성화)
  - 활성화된 Routed Experts: 2 × 12.6M = 25.2M
  - Router: ~1M (무시 가능)
  
  → 한 MoE 레이어의 활성화 파라미터 ≈ 12.6M + 25.2M = 37.8M
```

## 사용자 주장의 논리적 오류

### 오류 1: 전체 파라미터와 활성화 파라미터의 혼동

**사용자 주장**: "Expert 2개 + Shared 1개 = 3개가 돌아가니까 3 × 4B = 12B"

**문제점**:
1. **4B는 전체 모델 파라미터**입니다. Expert만의 파라미터가 아닙니다.
2. Expert 2개 + Shared 1개가 "3개의 전체 모델"이 되는 것이 아닙니다.
3. 각 Expert는 **전체 모델의 일부**일 뿐입니다.

### 오류 2: Expert 파라미터 비율 오인

**실제 구조**:
```
전체 모델 4B = 
  + Embedding: ~X M
  + Attention (모든 레이어): ~Y M  
  + MoE 레이어들:
    - Shared Expert들 (모든 MoE 레이어): ~Z M
    - Routed Experts들 (모든 MoE 레이어, 256개): ~W B
  + 기타: ~V M

여기서 Routed Experts가 전체의 대부분을 차지합니다.
```

**Inference 시**:
```
활성화 파라미터 = 
  + Embedding (항상)
  + Attention (항상)
  + Shared Expert들 (항상)
  + 활성화된 Routed Experts만 (top_k × 레이어 수)
```

### 오류 3: Expert 수와 모델 크기의 관계 오해

**사용자 논리**: Expert 2개 = 2 × 4B = 8B (틀림)

**올바른 이해**:
- Expert 2개는 **전체 모델의 일부 구성요소**입니다
- 각 Expert는 **전체 모델 크기의 일부**만 담당합니다
- 예: 전체 모델이 4B이고 Expert가 256개라면, 각 Expert는 약 4B/256 ≈ 15.6M 정도

## 실제 계산 예시

### 시나리오: 4B 모델, top_k=2

**가정**:
- 전체 모델: 4B 파라미터
- MoE 레이어 수: 20개
- n_routed_experts: 256
- n_shared_experts: 1
- top_k: 2

**파라미터 분포 추정**:
```
전체 4B = 
  - Embedding: ~100M (2.5%)
  - Attention: ~800M (20%)
  - MoE 레이어들:
    - Shared Experts (20개): ~250M (6.25%)
    - Routed Experts (256 × 20): ~2.8B (70%)
  - 기타: ~50M (1.25%)
```

**Inference 시 활성화**:
```
활성화 파라미터 ≈
  - Embedding: ~100M (항상)
  - Attention: ~800M (항상)
  - Shared Experts: ~250M (항상)
  - 활성화된 Routed Experts: (2/256) × 2.8B ≈ 22M
  
  → 총 활성화 파라미터 ≈ 1.17B (약 29%)
```

### 핵심 계산식

```
활성화 비율 = (활성화된 Routed Experts / 전체 Routed Experts) × Routed_Expert_비율 + 나머지(항상 활성화)

활성화 파라미터 = 
  항상 활성화 부분 (Embedding + Attention + Shared + 기타)
  + (top_k / n_routed_experts) × Routed_Expert_전체_파라미터
```

## 정확한 활성화 파라미터 계산 방법

### 방법 1: 실제 모델에서 측정

```python
def count_active_parameters(model, top_k=2):
    """Inference 시 실제 활성화되는 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # 항상 활성화되는 부분
    always_active = 0
    for name, param in model.named_parameters():
        if 'embed' in name or 'attn' in name or 'shared_experts' in name or 'norm' in name or 'lm_head' in name:
            always_active += param.numel()
    
    # Routed Experts 부분
    routed_experts_total = 0
    for name, param in model.named_parameters():
        if 'experts' in name and 'shared' not in name:
            routed_experts_total += param.numel()
    
    # 활성화 비율
    n_routed_experts = model.config.n_routed_experts
    activation_ratio = top_k / n_routed_experts
    
    active_routed = routed_experts_total * activation_ratio
    
    active_params = always_active + active_routed
    activation_rate = active_params / total_params
    
    return active_params, activation_rate
```

### 방법 2: 이론적 계산

```
Active_Params = 
  Always_Active_Params 
  + (top_k / n_routed_experts) × Routed_Experts_Total_Params
```

## 결론

### 사용자 주장의 오류 요약

1. ❌ **"Expert 2개 + Shared 1개 = 3개가 돌아간다"**
   - ✅ 맞음: 3개가 활성화됨
   
2. ❌ **"3개 × 4B = 12B가 active"**
   - ❌ **완전히 틀림**: 
     - 4B는 전체 모델 파라미터
     - Expert 2개는 전체 모델의 일부만 담당
     - 각 Expert는 4B의 일부일 뿐, 4B 전체가 아님

### 올바른 이해

**4B 모델에서 top_k=2로 inference 시**:
- 활성화 파라미터 ≈ **1.0B ~ 1.5B** (전체의 25~37%)
- 전체 파라미터 4B 중에서 **일부만** 활성화됨
- Expert 2개 + Shared 1개가 활성화되지만, 이것들은 **전체 모델의 구성요소**일 뿐

### 실제 활성화 비율

일반적인 MoE 모델에서:
- **top_k=2, n_routed_experts=256**: 활성화 비율 ≈ **25~35%**
- **top_k=8, n_routed_experts=256**: 활성화 비율 ≈ **35~45%**

**절대로 전체 파라미터의 3배가 활성화되지 않습니다!**

---

## 참고: 실제 MoE 모델 사례

### Qwen3
- 전체: 235B
- 활성화: 22B (약 9.4%)
- top_k=2, n_routed_experts=256 정도로 추정

### MiniMax-01  
- 전체: 456B
- 활성화: 45.9B (약 10%)
- top_k=8, n_routed_experts=256 정도로 추정

이 사례들도 **전체의 일부만 활성화**되며, 활성화 비율은 10% 내외입니다.

