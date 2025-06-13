from datasets import load_dataset
from transformers import AutoProcessor
import torch
from typing import Dict, Any, List

def get_simple_sft_dataset(
    dataset_name: str = "HuggingFaceTB/smoltalk", 
    config_name: str = "all",
    tokenizer=None,
    max_length: int = 2048,
    max_samples: int = 1000,
    test_size: float = 0.1
):
    """
    간단한 SFT 데이터셋을 로드합니다.
    작은 데이터셋으로 빠른 테스트용입니다.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    print(f"📦 로딩 중: {dataset_name}")
    
    # 작은 데이터셋 로드
    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split="train", streaming=True)
        else:
            dataset = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        # 대안 데이터셋 시도
        print("🔄 대안 데이터셋 시도: microsoft/orca-agentinstruct-1M-v1")
        dataset = load_dataset("microsoft/orca-agentinstruct-1M-v1", "creative_content", split="train", streaming=True)
    
    # 제한된 샘플만 가져오기
    samples = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        
        # 데이터셋별 메시지 형식 변환
        converted = convert_sample_to_messages(sample, dataset_name)
        if converted:
            samples.append(converted)
    
    print(f"✅ {len(samples)}개 샘플 수집 완료")
    
    # 훈련/테스트 분할
    split_idx = int(len(samples) * (1 - test_size))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    # 토크나이즈 처리
    train_dataset = [process_sample(sample, tokenizer, max_length) for sample in train_samples]
    test_dataset = [process_sample(sample, tokenizer, max_length) for sample in test_samples]
    
    # None 제거
    train_dataset = [sample for sample in train_dataset if sample is not None]
    test_dataset = [sample for sample in test_dataset if sample is not None]
    
    print(f"📊 훈련: {len(train_dataset)}개, 테스트: {len(test_dataset)}개")
    
    from datasets import Dataset, DatasetDict
    return DatasetDict({
        "train": Dataset.from_list(train_dataset),
        "test": Dataset.from_list(test_dataset)
    })

def convert_sample_to_messages(sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """샘플을 messages 형식으로 변환"""
    
    if dataset_name == "HuggingFaceTB/smoltalk":
        if "messages" in sample and isinstance(sample["messages"], list):
            return {"messages": sample["messages"]}
    
    elif "orca-agentinstruct" in dataset_name:
        if "messages" in sample and isinstance(sample["messages"], list):
            return {"messages": sample["messages"]}
    
    # 기본 instruction-output 형식 처리
    elif "instruction" in sample and "output" in sample:
        messages = [
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]}
        ]
        return {"messages": messages}
    
    # conversations 형식 처리
    elif "conversations" in sample and isinstance(sample["conversations"], list):
        messages = []
        for conv in sample["conversations"]:
            if isinstance(conv, dict) and "from" in conv and "value" in conv:
                role = "user" if conv["from"] in ["human", "user"] else "assistant"
                messages.append({"role": role, "content": conv["value"]})
        if messages:
            return {"messages": messages}
    
    return None

def process_sample(sample: Dict[str, Any], tokenizer, max_length: int):
    """샘플을 토크나이즈하여 훈련 형식으로 변환"""
    try:
        if "messages" not in sample:
            return None
        
        # 채팅 템플릿 적용
        tokenized = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            truncation=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        if tokenized is None:
            return None
        
        # 텐서를 리스트로 변환하여 Dataset에 저장 가능하게 함
        result = {}
        for key, value in tokenized.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze().tolist()
            else:
                result[key] = value
        
        return result
        
    except Exception as e:
        return None

def create_simple_collate_fn(tokenizer):
    """간단한 collate function"""
    def collate_fn(examples):
        # input_ids와 attention_mask 추출
        input_ids = [torch.tensor(ex["input_ids"]) for ex in examples if "input_ids" in ex]
        attention_mask = [torch.tensor(ex["attention_mask"]) for ex in examples if "attention_mask" in ex]
        
        if not input_ids:
            return None
        
        # 패딩 처리
        from torch.nn.utils.rnn import pad_sequence
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id or tokenizer.eos_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        # labels는 input_ids와 동일 (causal LM)
        labels = input_ids.clone()
        
        # 패딩 토큰은 loss 계산에서 제외
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        labels[labels == pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    return collate_fn

# 테스트용 작은 데이터셋 빌더 함수들
def smoltalk_dataset(tokenizer, max_samples: int = 500):
    """SmolTalk 데이터셋 빌더"""
    return get_simple_sft_dataset(
        dataset_name="HuggingFaceTB/smoltalk",
        config_name="all", 
        tokenizer=tokenizer,
        max_samples=max_samples
    )

def orca_mini_dataset(tokenizer, max_samples: int = 500):
    """Orca 미니 데이터셋 빌더"""
    return get_simple_sft_dataset(
        dataset_name="microsoft/orca-agentinstruct-1M-v1",
        config_name="creative_content",
        tokenizer=tokenizer, 
        max_samples=max_samples
    )

if __name__ == "__main__":
    # 테스트
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = smoltalk_dataset(tokenizer, max_samples=100)
    print(f"데이터셋 생성 완료: {dataset}")
    print(f"샘플 예시: {dataset['train'][0]}") 