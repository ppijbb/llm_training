# DeepSpeed Custom Optimizers 사용법

이 문서는 DeepSpeed에서 custom optimizer를 사용하는 방법을 설명합니다.

## 📋 개요

DeepSpeed에서 custom optimizer를 사용하면 다음과 같은 이점이 있습니다:

- **Muon Optimizer**: Newton-Schulz iteration을 사용한 직교화 기반 최적화
- **Lion Optimizer**: 메모리 효율적이고 빠른 수렴
- **AdaFactor Optimizer**: 대용량 모델을 위한 메모리 절약
- **Sophia Optimizer**: 2차 미분 정보를 활용한 효율적인 학습

## 🚀 사용 방법

### 1. Custom Optimizer Config 설정

DeepSpeed config 파일에서 custom optimizer를 지정합니다:

```json
{
    "optimizer": {
        "type": "MuonOptimizer",  // 또는 "LionOptimizer", "AdaFactorOptimizer", "SophiaOptimizer"
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "momentum": 0.95,
            "nesterov": true,
            "ns_steps": 5,
            "adamw_betas": [0.9, 0.95],
            "adamw_eps": 1e-8
        }
    }
}
```

### 2. Training Config 설정

G3MoE training config에서 custom optimizer DeepSpeed config를 지정합니다:

```json
{
    "model_config": {
        "deepspeed_config": "sft/config/deepspeed_custom_optimizer.json"
    }
}
```

### 3. Training 실행

```bash
python sft/custom_model_sft.py --config sft/config/g3moe_custom_optimizer_config.json
```

## 🔧 Custom Optimizer 종류

### Muon Optimizer

**특징:**
- Newton-Schulz iteration을 사용한 직교화 기반 최적화
- 2D 파라미터에 대해서는 Muon, 나머지는 AdamW 사용
- 메모리 효율적이고 안정적인 학습

**권장 설정:**
```json
{
    "type": "MuonOptimizer",
    "params": {
        "lr": "auto",
        "weight_decay": "auto",
        "momentum": 0.95,
        "nesterov": true,
        "ns_steps": 5,
        "adamw_betas": [0.9, 0.95],
        "adamw_eps": 1e-8
    }
}
```

### Lion Optimizer

**특징:**
- 메모리 효율적 (momentum만 저장)
- 빠른 수렴 속도
- sign-based update

**권장 설정:**
```json
{
    "type": "LionOptimizer",
    "params": {
        "lr": "auto",
        "weight_decay": "auto",
        "beta1": 0.9,
        "beta2": 0.99
    }
}
```

### AdaFactor Optimizer

**특징:**
- 메모리 사용량 대폭 감소
- 대용량 모델에 적합
- 행/열 단위 통계 사용

**권장 설정:**
```json
{
    "type": "AdaFactorOptimizer",
    "params": {
        "lr": "auto",
        "weight_decay": "auto",
        "beta1": 0.9,
        "beta2": 0.999,
        "eps1": 1e-30,
        "eps2": 1e-3,
        "cliping_threshold": 1.0
    }
}
```

### Sophia Optimizer

**특징:**
- 2차 미분 정보 활용
- 더 정확한 parameter update
- Hessian 추정 사용

**권장 설정:**
```json
{
    "type": "SophiaOptimizer",
    "params": {
        "lr": "auto",
        "weight_decay": "auto",
        "beta1": 0.965,
        "beta2": 0.99,
        "rho": 0.01
    }
}
```

## 📊 성능 비교

| Optimizer | 메모리 효율성 | 수렴 속도 | 안정성 | 권장 용도 |
|-----------|-------------|----------|--------|----------|
| AdamW | 보통 | 보통 | 높음 | 일반적인 경우 |
| Muon | 높음 | 빠름 | 높음 | 안정적인 학습 |
| Lion | 높음 | 빠름 | 높음 | 빠른 실험 |
| AdaFactor | 매우 높음 | 보통 | 높음 | 대용량 모델 |
| Sophia | 보통 | 빠름 | 보통 | 정확도 중요 |

## ⚠️ 주의사항

1. **DeepSpeed 호환성**: Custom optimizer는 DeepSpeed의 ZeRO optimization과 호환되어야 합니다.

2. **메모리 사용량**: AdaFactor는 메모리를 절약하지만, Lion과 Sophia는 추가 메모리가 필요할 수 있습니다.

3. **학습률 조정**: Custom optimizer마다 최적 학습률이 다를 수 있으므로 실험을 통해 조정하세요.

4. **Gradient Clipping**: 일부 custom optimizer는 gradient clipping이 필요할 수 있습니다.

## 🔍 디버깅

Custom optimizer 사용 시 문제가 발생하면:

1. **로그 확인**: DeepSpeed가 custom optimizer를 인식하는지 확인
2. **메모리 사용량**: GPU 메모리 사용량 모니터링
3. **Loss 추이**: 학습이 안정적으로 진행되는지 확인

## 📝 예시 Config 파일들

- `sft/config/deepspeed_custom_optimizer.json`: Lion Optimizer 사용
- `sft/config/g3moe_custom_optimizer_config.json`: Custom optimizer와 G3MoE 조합

## 🎯 권장 사항

1. **처음 사용**: Muon Optimizer부터 시작하세요 (안정적이고 빠름)
2. **메모리 부족**: AdaFactor Optimizer 사용
3. **정확도 중요**: Sophia Optimizer 고려
4. **실험**: 여러 optimizer를 비교해보세요 