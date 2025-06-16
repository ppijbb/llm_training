# MoE 모니터링 콜백 사용 가이드

이 문서는 `sft/moe_monitoring_callback.py`에 구현된 MoE(Mixture-of-Experts) 모니터링 콜백의 사용법을 안내합니다. 이 콜백은 MoE 모델 훈련 중 발생할 수 있는 주요 문제들(Expert 로드 불균형, 라우팅 붕괴 등)을 실시간으로 추적하고, 로깅하며, 경고를 발생시키는 역할을 합니다.

## 1. 주요 기능

- **Expert 사용률 추적**: 각 Expert가 얼마나 자주, 그리고 균등하게 사용되는지 모니터링합니다.
- **라우팅 패턴 분석**: 토큰이 Expert들에게 어떻게 분배되는지 엔트로피와 같은 지표로 분석합니다.
- **실시간 로깅**: `wandb`나 `TensorBoard`와 같은 로깅 도구와 통합하여 시각적인 대시보드를 제공합니다.
- **자동 경고 시스템**: 미리 정의된 임계값을 초과하는 이상 현상(예: 심각한 불균형)이 발생하면 콘솔에 경고를 출력합니다.
- **프레임워크 호환성**: `transformers` 라이브러리의 `Trainer`와 순수 `PyTorch` 훈련 루프 모두에서 쉽게 사용할 수 있도록 설계되었습니다.

---

## 2. MoE 모니터링의 핵심 개념 및 메트릭

콜백을 사용하기 전에, MoE 모델에서 무엇을 모니터링해야 하는지 이해하는 것이 중요합니다.

### 2.1. Expert 사용률 (Expert Utilization)

가장 기본적인 지표로, 각 Expert가 얼마나 많은 토큰을 처리하는지를 나타냅니다. 이상적으로는 모든 Expert가 비슷한 수의 토큰을 처리해야 합니다.

- **문제 상황**: 특정 Expert 몇 개에만 토큰이 몰리는 현상 (로드 불균형).
- **관련 메트릭**:
    - **Expert 사용 빈도 (Usage Counts)**: 각 Expert가 처리한 토큰의 총 수.
    - **변동 계수 (Coefficient of Variation, CV)**: 사용 빈도 분포의 표준편차를 평균으로 나눈 값. 0에 가까울수록 균등합니다.
    - **미사용 Expert (Unused Experts)**: 훈련 중 한 번도 사용되지 않은 Expert의 수.

**[개념 예시 코드]**
```python
import torch

def track_expert_usage(expert_assignments, num_experts):
    """
    배치 내 Expert 사용 빈도를 계산합니다.
    
    Args:
        expert_assignments (Tensor): 각 토큰이 할당된 Expert의 인덱스.
        num_experts (int): 전체 Expert의 수.
        
    Returns:
        Tensor: 각 Expert의 사용 빈도.
    """
    return torch.bincount(expert_assignments.flatten(), minlength=num_experts)

# 예시: 100개 토큰이 8개 Expert에 할당됨
num_experts = 8
assignments = torch.randint(0, num_experts, (100,))
usage_counts = track_expert_usage(assignments, num_experts)
print(f"Expert 사용 빈도: {usage_counts}")
```

### 2.2. 라우팅 다양성 (Routing Diversity)

라우터가 토큰을 얼마나 다양한 Expert에게 보내는지를 측정합니다. 다양성이 낮으면 모델이 소수의 Expert에게 과적합될 수 있습니다.

- **문제 상황**: 대부분의 토큰이 항상 같은 Expert로 라우팅되는 현상.
- **관련 메트릭**:
    - **라우팅 엔트로피 (Routing Entropy)**: 라우터가 출력하는 확률 분포의 불확실성을 측정. 값이 높을수록 다양한 Expert에게 분산됨을 의미합니다.

**[개념 예시 코드]**
```python
import torch.nn.functional as F

def calculate_routing_entropy(routing_probs):
    """
    라우팅 확률 분포의 평균 엔트로피를 계산합니다.
    
    Args:
        routing_probs (Tensor): 라우터가 출력한 확률 분포 (batch_size, num_experts).
        
    Returns:
        float: 평균 엔트로피 값.
    """
    # 각 토큰에 대한 엔트로피 계산
    token_entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-10), dim=-1)
    # 배치 전체의 평균 엔트로피 반환
    return token_entropy.mean().item()

# 예시: 4개 토큰, 8개 Expert
probs = F.softmax(torch.randn(4, 8), dim=-1)
entropy = calculate_routing_entropy(probs)
print(f"평균 라우팅 엔트로피: {entropy:.4f}")
```

### 2.3. 로드 밸런싱 손실 (Load Balancing Loss)

많은 MoE 모델은 Expert 간의 로드 밸런스를 맞추기 위해 보조 손실(auxiliary loss)을 사용합니다. 이 손실 값을 직접 모니터링하는 것도 중요합니다.

- **문제 상황**: 로드 밸런싱 손실이 비정상적으로 높거나, 0으로 수렴하여 라우터 학습이 멈추는 경우.
- **관련 메트릭**: `aux_loss`, `load_balance_loss` 등 모델 출력에 포함된 값.

**[개념 예시 코드]**
```python
def calculate_load_balancing_loss(routing_probs, expert_assignments, num_experts):
    """
    Switch Transformer 스타일의 간단한 로드 밸런싱 손실을 계산합니다.
    
    Args:
        routing_probs (Tensor): 라우터 확률 분포.
        expert_assignments (Tensor): 할당된 Expert 인덱스.
        num_experts (int): 전체 Expert 수.
        
    Returns:
        Tensor: 로드 밸런싱 손실 값.
    """
    # 각 Expert에 할당된 토큰의 비율
    tokens_per_expert = torch.bincount(expert_assignments, minlength=num_experts).float()
    fraction_of_tokens = tokens_per_expert / (tokens_per_expert.sum() + 1e-8)
    
    # 각 Expert가 받은 라우터 확률의 평균
    mean_prob_per_expert = routing_probs.mean(dim=0)
    
    # 보조 손실 계산
    load_balance_loss = num_experts * torch.sum(fraction_of_tokens * mean_prob_per_expert)
    return load_balance_loss

# 예시
assignments = torch.randint(0, 8, (100,))
probs = F.softmax(torch.randn(100, 8), dim=-1)
aux_loss = calculate_load_balancing_loss(probs, assignments, 8)
print(f"로드 밸런싱 손실: {aux_loss.item():.4f}")
```

---

## 3. `transformers.Trainer`와 함께 사용하기

`transformers` 라이브러리를 사용하여 모델을 훈련하는 경우, `create_moe_callback_for_transformers` 헬퍼 함수를 사용하는 것이 가장 간편합니다.

### 3.1. 콜백 임포트

먼저, 훈련 스크립트에서 필요한 함수를 임포트합니다.

```python
# train_script.py

from moe_monitoring_callback import create_moe_callback_for_transformers
from transformers import Trainer, TrainingArguments
import wandb

# ... (모델, 데이터셋, 토크나이저 로딩 코드)
```

### 3.2. 로거 및 콜백 초기화

훈련을 시작하기 전에 `wandb`와 같은 로거를 초기화하고, `create_moe_callback_for_transformers` 함수를 호출하여 콜백 인스턴스를 생성합니다.

```python
# W&B 로거 초기화 (예시)
wandb.init(project="moe-sft-project", name="g3moe-run-1")

# MoE 모니터링 콜백 생성
moe_callback = create_moe_callback_for_transformers(
    log_every_n_steps=50,       # 50 스텝마다 로그 기록
    logger=wandb,               # 사용할 로거 지정 (wandb)
    log_to_console=True,        # 콘솔에도 주요 메트릭 출력
    
    # === 고급 설정 (선택사항) ===
    log_heatmap_every=500,      # 500 스텝마다 Expert 사용률 히트맵 로깅
    alert_threshold_imbalance=4.0, # 특정 Expert 사용률이 평균의 4배를 초과하면 경고
    unused_expert_threshold=0.25,  # 25% 이상의 Expert가 미사용되면 경고
    entropy_threshold=0.1,         # 라우팅 엔트로피가 0.1 미만이면 경고
    save_detailed_logs=False       # 상세 JSON 로그 저장 여부
)
```

### 3.3. `Trainer`에 콜백 추가

생성된 콜백을 `Trainer`의 `callbacks` 리스트에 추가합니다.

```python
# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    # ... (기타 훈련 인자)
    report_to="wandb"  # W&B 로깅 활성화
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[moe_callback]  # 콜백 리스트에 추가
    # ... (기타 Trainer 인자)
)
```

### 3.4. 훈련 시작

이제 `trainer.train()`을 호출하면, 훈련이 진행되는 동안 MoE 관련 지표들이 자동으로 `wandb`와 콘솔에 기록됩니다.

```python
trainer.train()
```

---

## 4. 순수 `PyTorch` 훈련 루프에서 사용하기

`transformers.Trainer`를 사용하지 않는 순수 `PyTorch` 환경에서는 `create_moe_callback_for_pytorch` 함수를 사용합니다.

### 4.1. 콜백 임포트 및 초기화

```python
# pure_pytorch_train.py

import torch
from moe_monitoring_callback import create_moe_callback_for_pytorch
import wandb

# 모델, 옵티마이저 등 초기화
model = YourMoEModel()
optimizer = torch.optim.Adam(model.parameters())

# 로거 및 콜백 초기화
wandb.init(project="moe-pytorch-project")

moe_callback = create_moe_callback_for_pytorch(
    model=model,                # **중요**: 모델 인스턴스를 직접 전달
    log_every_n_steps=100,
    logger=wandb
)
```

### 4.2. 훈련 루프에 훅(Hook) 추가

훈련 루프의 각 스텝 시작과 끝에 콜백의 `on_step_begin()`과 `on_step_end()` 메서드를 호출해야 합니다.

```python
num_epochs = 3
global_step = 0

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # --- 스텝 시작 ---
        moe_callback.on_step_begin()
        
        # 일반적인 훈련 로직
        optimizer.zero_grad()
        
        # Forward pass (이 과정에서 MoE 레이어의 hook이 자동으로 호출됨)
        outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
        loss = outputs.loss
        
        # Backward pass 및 파라미터 업데이트
        loss.backward()
        optimizer.step()
        
        # --- 스텝 종료 ---
        moe_callback.on_step_end()
        
        global_step += 1

# 훈련 종료 후 hook 정리
moe_callback.cleanup()
```

## 5. 모니터링 결과 확인

콜백이 정상적으로 작동하면 `wandb` 대시보드의 "Charts" 탭에서 다음과 같은 메트릭들을 확인할 수 있습니다.

- `moe/avg_expert_cv`: Expert 사용량의 변동 계수 (낮을수록 균등)
- `moe/avg_routing_entropy`: 라우팅의 다양성 (높을수록 좋음)
- `moe/total_unused_experts`: 모든 MoE 레이어에서 한 번도 사용되지 않은 Expert의 총 수
- `moe/{layer_name}/...`: 각 MoE 레이어별 상세 지표
- `moe/{layer_name}/usage_heatmap`: (설정 시) 특정 레이어의 시간 경과에 따른 Expert 사용 패턴 히트맵 이미지

콘솔에서는 설정된 임계값을 초과할 때 다음과 같은 경고 메시지를 볼 수 있습니다.

```
⚠️  MoE Alert at step 1500: layers.0.mlp: Severe expert imbalance (ratio: 4.52)
⚠️  MoE Alert at step 2300: layers.4.mlp: 3/8 experts unused
```

이 가이드를 통해 MoE 모델의 훈련 과정을 효과적으로 모니터링하고 안정성을 높이는 데 도움이 되기를 바랍니다. 