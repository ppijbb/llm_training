# Router Weight Tracking for SPECTRA

이 모듈은 spertra 모델의 router 가중치를 step별로 tracking하고 분석하는 기능을 제공합니다.

## 개요

SPECTRA MoE의 router는 두 가지 주요 구성 요소로 이루어져 있습니다:

1. **load_balancer (GRU)**: Sequential routing을 위한 가중치
   - `weight_ih`: Input-to-hidden weights
   - `weight_hh`: Hidden-to-hidden weights
   - `bias_ih`, `bias_hh`: Bias terms (optional)

2. **expression_projector**: Orthogonal expression projection을 위한 가중치
   - `weight`: Expression projection weights
   - `bias`: Expression projection bias (optional)

## 사용 방법

### 1. Trainer에 Callback 추가

```python
from eval.router_weight_callback import RouterWeightTrackingCallback

# Callback 생성
router_weight_callback = RouterWeightTrackingCallback(
    save_dir="./router_weight_logs",
    log_every_n_steps=100,  # 100 step마다 tracking
    save_full_weights=False,  # 통계만 저장 (메모리 효율)
    max_history=1000,  # 최근 1000 step 히스토리 유지
)

# Trainer에 추가
trainer.add_callback(router_weight_callback)
```

### 2. 수동으로 가중치 추출

```python
from eval.router_weight_tracker import extract_router_weights, compute_weight_statistics

# 모델에서 router 가중치 추출
router_weights = extract_router_weights(model)

# 각 레이어별 통계 계산
for layer_key, layer_weights in router_weights.items():
    stats = compute_weight_statistics(layer_weights['load_balancer'])
    print(f"{layer_key} - Load Balancer Stats: {stats}")
```

### 3. 가중치 변화 분석

```python
from eval.router_weight_tracker import RouterWeightTracker

# Tracker 생성
tracker = RouterWeightTracker(save_dir="./router_weight_logs")

# Step별 tracking
for step in range(0, 1000, 100):
    step_stats = tracker.track_step(model, step=step, global_step=step)
    
    # 특정 가중치의 trajectory 가져오기
    trajectory = tracker.get_weight_trajectory(
        layer_key='layer_0',
        weight_name='load_balancer.weight_ih_mean',
        start_step=0,
        end_step=step
    )
    print(f"Step {step}: weight_ih_mean trajectory length = {len(trajectory)}")
```

## 저장되는 데이터

### 통계 파일 (JSON)

각 step마다 다음 통계가 저장됩니다:

```json
{
  "step": 100,
  "global_step": 100,
  "layers": {
    "layer_0": {
      "load_balancer": {
        "weight_ih_mean": 0.001,
        "weight_ih_std": 0.002,
        "weight_ih_min": -0.005,
        "weight_ih_max": 0.005,
        "weight_ih_norm": 1.234,
        "weight_ih_shape": [128, 512],
        "weight_hh_mean": 0.001,
        ...
      },
      "expression_projector": {
        "weight_mean": 0.001,
        "weight_std": 0.002,
        ...
      },
      "load_balancer_changes": {
        "weight_ih_diff_norm": 0.0001,
        "weight_ih_diff_mean": 0.00001,
        ...
      }
    }
  }
}
```

### 전체 가중치 스냅샷 (PyTorch)

`save_full_weights=True`로 설정하면 특정 step의 전체 가중치 텐서가 저장됩니다:

```python
# 스냅샷 로드
snapshot = torch.load("router_weight_logs/router_weight_snapshot_step_1000.pt")

# 특정 레이어의 가중치 접근
layer_0_weights = snapshot['layer_0']['load_balancer']['weight_ih']
```

## WandB 통합

Callback은 자동으로 WandB에 주요 통계를 로깅합니다:

- `router_weight/{layer_key}/load_balancer/{stat_name}`
- `router_weight/{layer_key}/expression_projector/{stat_name}`
- `router_weight/{layer_key}/load_balancer_change/{change_name}`
- `router_weight/{layer_key}/expression_projector_change/{change_name}`

## 분석 예시

### 가중치 변화 시각화

```python
import matplotlib.pyplot as plt
from eval.router_weight_tracker import RouterWeightTracker

tracker = RouterWeightTracker(save_dir="./router_weight_logs")

# 특정 가중치의 trajectory 가져오기
trajectory = tracker.get_weight_trajectory(
    layer_key='layer_0',
    weight_name='load_balancer.weight_ih_mean',
)

# 시각화
plt.plot(trajectory)
plt.xlabel('Step')
plt.ylabel('Weight Mean')
plt.title('Load Balancer Weight Mean Over Time')
plt.show()
```

### 가중치 분포 분석

```python
import json
import numpy as np

# 통계 파일 로드
with open("router_weight_logs/router_weight_stats_step_1000.json", 'r') as f:
    data = json.load(f)

# 모든 레이어의 가중치 평균 추출
weight_means = []
for layer_key, layer_data in data['history'][-1]['layers'].items():
    if 'load_balancer' in layer_data:
        weight_means.append(layer_data['load_balancer']['weight_ih_mean'])

print(f"Average weight mean across layers: {np.mean(weight_means)}")
print(f"Std of weight mean across layers: {np.std(weight_means)}")
```

## 주의사항

1. **메모리 사용**: `save_full_weights=True`는 메모리를 많이 사용합니다. 대부분의 경우 통계만 저장하는 것이 좋습니다.

2. **저장 주기**: `log_every_n_steps`를 너무 작게 설정하면 디스크 I/O가 많아질 수 있습니다. 기본값 100을 권장합니다.

3. **분산 학습**: 분산 학습 환경에서는 rank 0 프로세스에서만 저장됩니다.

## 파일 구조

```
router_weight_logs/
├── router_weight_stats_step_100.json
├── router_weight_stats_step_200.json
├── ...
├── router_weight_snapshot_step_1000.pt  # save_full_weights=True인 경우만
└── router_weight_summary.json
```

## API 참조

### RouterWeightTracker

- `track_step(model, step, global_step)`: 현재 step의 router 가중치 tracking
- `get_weight_trajectory(layer_key, weight_name, start_step, end_step)`: 특정 가중치의 step별 변화 trajectory 반환
- `save_summary(output_file)`: 전체 tracking 결과 요약 저장

### RouterWeightTrackingCallback

- `on_step_end()`: 각 training step 끝에서 자동으로 router 가중치 tracking
- `on_train_end()`: Training 종료 시 최종 요약 저장
- `on_save()`: Checkpoint 저장 시 router 가중치 요약도 함께 저장

## 문제 해결

### Router 가중치를 찾을 수 없는 경우

모델 구조가 예상과 다를 수 있습니다. `extract_router_weights()` 함수가 모델을 탐색하는 방식을 확인하세요.

### 메모리 부족

- `save_full_weights=False`로 설정
- `log_every_n_steps`를 늘려서 tracking 빈도 감소
- `max_history`를 줄여서 메모리에 유지하는 히스토리 감소
