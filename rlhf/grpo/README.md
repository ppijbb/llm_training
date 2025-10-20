# GRPO (Group Relative Policy Optimization) Training

Unsloth를 사용한 GRPO 훈련을 위한 모듈입니다. HuggingFace 데이터셋을 사용하여 효율적인 GRPO 훈련을 수행할 수 있습니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install torch transformers datasets accelerate
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl peft bitsandbytes wandb
```

### 2. 기본 훈련

```bash
# 빠른 테스트 (100개 샘플, 1 에포크)
python train_grpo.py --quick-test

# 기본 훈련 (Llama 3.1 8B + UltraFeedback)
python train_grpo.py --model llama-3.1-8b --dataset ultrafeedback

# 프로덕션 훈련 (더 큰 데이터셋)
python train_grpo.py --production --model llama-3.1-8b
```

### 3. 커스텀 데이터로 훈련

```bash
# JSONL 파일 사용
python train_grpo.py --custom-data /path/to/your_data.jsonl --model llama-3.1-8b

# 설정 파일 사용
python train_grpo.py --config my_config.json
```

### 4. 보상 함수 커스터마이징

```bash
# 시스템적 보상 함수 사용
python train_grpo.py --reward-function systematic --reward-config balanced

# 그룹 상대 보상 함수 사용
python train_grpo.py --reward-function group_relative --reward-config aggressive

# 다중 목표 보상 함수 사용
python train_grpo.py --reward-function multi_objective --reward-config default

# 커스텀 보상 함수 설정 파일 사용
python train_grpo.py --custom-reward-config my_reward_config.json
```


## 📊 지원하는 모델

- `llama-3.1-8b`: Llama 3.1 8B (기본)
- `llama-3.1-70b`: Llama 3.1 70B
- `gemma-2-9b`: Gemma 2 9B
- `qwen2.5-7b`: Qwen2.5 7B

## 📦 지원하는 데이터셋

- `ultrafeedback`: HuggingFaceH4/ultrafeedback_binarized (기본)
- `ultrafeedback_large`: 더 큰 UltraFeedback 데이터셋
- `hh_rlhf`: Anthropic/hh-rlhf
- `openai_summarize`: OpenAI summarize from feedback

## ⚙️ 설정 옵션

### 명령줄 옵션

```bash
python train_grpo.py [OPTIONS]

옵션:
  --model {llama-3.1-8b,llama-3.1-70b,gemma-2-9b,qwen2.5-7b}
                        사용할 모델 (기본: llama-3.1-8b)
  --dataset {ultrafeedback,ultrafeedback_large,hh_rlhf,openai_summarize}
                        사용할 데이터셋 (기본: ultrafeedback)
  --custom-data PATH    커스텀 데이터셋 파일 경로
  --config PATH         설정 파일 경로
  --quick-test          빠른 테스트 실행
  --production          프로덕션 훈련 실행
  --max-samples N       사용할 최대 샘플 수
  --epochs N            훈련 에포크 수
  --learning-rate LR    학습률
  --batch-size N        배치 크기
  --output-dir PATH     출력 디렉토리
  --reward-function {systematic,group_relative,multi_objective}
                        보상 함수 타입 (기본: systematic)
  --reward-config {default,balanced,aggressive}
                        보상 함수 설정 (기본: default)
  --custom-reward-config PATH
                        커스텀 보상 함수 설정 파일
  --wandb-project NAME  Weights & Biases 프로젝트 이름
  --no-wandb           Weights & Biases 비활성화
  --eval-only          평가만 실행
  --resume-from-checkpoint PATH
                        체크포인트에서 재시작
```

### 설정 파일 예제

```json
{
  "model_name": "unsloth/llama-3.1-8b-bnb-4bit",
  "max_seq_length": 2048,
  "learning_rate": 5e-7,
  "num_train_epochs": 3,
  "per_device_train_batch_size": 2,
  "per_device_eval_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "max_samples": 1000,
  "output_dir": "./grpo_outputs",
  "beta": 0.1,
  "gamma": 1.0,
  "group_size": 4,
  "reward_function_type": "systematic",
  "reward_config_name": "balanced"
}
```

### 보상 함수 설정 예제

```json
{
  "consistency_weight": 0.4,
  "coherence_weight": 0.3,
  "helpfulness_weight": 0.3,
  "temperature": 0.8,
  "custom_penalty": 0.05
}
```

## 📁 데이터 형식

### UltraFeedback 형식

```json
{
  "chosen": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ],
  "rejected": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "I don't know."}
  ]
}
```

### 커스텀 데이터 형식

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "I don't know."
}
```

## 🔧 고급 사용법

### 1. 커스텀 설정 생성

```python
from config import create_default_config

# 기본 설정 생성
config = create_default_config("llama-3.1-8b", "ultrafeedback")

# 커스텀 설정 적용
config.learning_rate = 1e-6
config.num_train_epochs = 5
config.max_samples = 5000

# 설정 저장
from config import save_config_to_file
save_config_to_file(config, "my_config.json")
```

### 2. 프로그래밍 방식으로 훈련

```python
from grpo_trainer import GRPOTrainer, GRPOConfig
from data_loader import create_grpo_dataloader

# 설정 생성
config = GRPOConfig(
    model_name="unsloth/llama-3.1-8b-bnb-4bit",
    max_samples=1000,
    output_dir="./my_grpo_outputs"
)

# 데이터 로더 생성
data_loader, dataset = create_grpo_dataloader(
    model_name=config.model_name,
    dataset_name="HuggingFaceH4/ultrafeedback_binarized",
    max_samples=config.max_samples
)

# 훈련기 생성 및 훈련
trainer = GRPOTrainer(config)
trainer.load_model()
trainer.train(dataset)
trainer.save_model()
```

## 📊 모니터링

### Weights & Biases

```bash
# W&B 프로젝트 설정
python train_grpo.py --wandb-project my-grpo-project

# W&B 비활성화
python train_grpo.py --no-wandb
```

### 로그 파일

훈련 로그는 `grpo_training.log` 파일에 저장됩니다.

## 🐛 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   python train_grpo.py --batch-size 1
   
   # 그래디언트 누적 사용
   # config.py에서 gradient_accumulation_steps 증가
   ```

2. **모델 로딩 실패**
   ```bash
   # 4bit 양자화 비활성화
   # config.py에서 load_in_4bit=False로 설정
   ```

3. **데이터셋 로딩 실패**
   ```bash
   # 스트리밍 모드 사용
   # data_loader.py에서 streaming=True로 설정
   ```

### 로그 확인

```bash
# 실시간 로그 확인
tail -f grpo_training.log

# 에러 로그만 확인
grep "ERROR" grpo_training.log
```

## 📚 참고 자료

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [GRPO Paper](https://arxiv.org/abs/2406.05896)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [TRL Library](https://huggingface.co/docs/trl)

## 🤝 기여하기

1. 이슈 리포트
2. 포크 및 브랜치 생성
3. 변경사항 커밋
4. 풀 리퀘스트 생성

## 📄 라이선스

MIT License
