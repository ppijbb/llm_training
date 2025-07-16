import logging
from datasets import load_dataset
from transformers import AutoProcessor
import torch
from typing import Dict, Any, List, Optional

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_simple_sft_dataset(
    dataset_name: str = "HuggingFaceTB/smoltalk", 
    config_name: str = "default",
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
    
    logger.info(f"📦 로딩 중: {dataset_name}")
    logger.info(f"   - config_name: {config_name}")
    logger.info(f"   - max_samples: {max_samples}")
    
    # 작은 데이터셋 로드
    dataset = None
    try:
        if config_name:
            logger.info(f"   - 시도: load_dataset({dataset_name}, {config_name}, split='train', streaming=True)")
            dataset = load_dataset(dataset_name, config_name, split="train", streaming=True)
        else:
            logger.info(f"   - 시도: load_dataset({dataset_name}, split='train', streaming=True)")
            dataset = load_dataset(dataset_name, split="train", streaming=True)
        logger.info("   ✅ 데이터셋 로드 성공")
    except Exception as e:
        logger.error(f"❌ 데이터셋 로드 실패: {e}")
        # 대안 데이터셋 시도
        logger.info("🔄 대안 데이터셋 시도: microsoft/orca-agentinstruct-1M-v1")
        try:
            dataset = load_dataset("microsoft/orca-agentinstruct-1M-v1", "creative_content", split="train", streaming=True)
            logger.info("   ✅ 대안 데이터셋 로드 성공")
        except Exception as e2:
            logger.error(f"❌ 대안 데이터셋도 실패: {e2}")
            # 최후의 수단: 매우 작은 더미 텍스트 데이터셋 생성
            logger.info("🔄 더미 데이터셋 생성")
            dummy_data = [
                {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
                {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well, thank you!"}]},
                {"messages": [{"role": "user", "content": "What's the weather like?"}, {"role": "assistant", "content": "I don't have access to real-time weather data."}]},
                {"messages": [{"role": "user", "content": "Tell me a joke"}, {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"}]},
                {"messages": [{"role": "user", "content": "Explain AI"}, {"role": "assistant", "content": "AI is artificial intelligence, computer systems that can perform tasks typically requiring human intelligence."}]}
            ]
            # 더미 데이터를 여러 번 반복하여 최소 샘플 수 확보
            while len(dummy_data) < max_samples:
                dummy_data.extend(dummy_data[:min(len(dummy_data), max_samples - len(dummy_data))])
            
            samples = dummy_data[:max_samples]
            logger.info(f"   ✅ 더미 데이터셋 생성: {len(samples)}개 샘플")
            
            # 훈련/테스트 분할
            split_idx = int(len(samples) * (1 - test_size))
            train_samples = samples[:split_idx]
            test_samples = samples[split_idx:]
            
            # 토크나이즈 처리
            train_dataset = []
            test_dataset = []
            
            for sample in train_samples:
                processed = process_sample(sample, tokenizer, max_length)
                if processed is not None:
                    train_dataset.append(processed)
            
            for sample in test_samples:
                processed = process_sample(sample, tokenizer, max_length)
                if processed is not None:
                    test_dataset.append(processed)
            
            logger.info(f"📊 더미 데이터 - 훈련: {len(train_dataset)}개, 테스트: {len(test_dataset)}개")
            
            from datasets import Dataset, DatasetDict
            return DatasetDict({
                "train": Dataset.from_list(train_dataset),
                "test": Dataset.from_list(test_dataset)
            })
    
    if dataset is None:
        raise RuntimeError("모든 데이터셋 로딩 시도가 실패했습니다.")
    
    # 제한된 샘플만 가져오기
    samples = []
    converted_count = 0
    skipped_count = 0
    
    logger.info(f"   📊 샘플 수집 시작 (최대 {max_samples}개)")
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        
        if i % 100 == 0 and i > 0:
            logger.debug(f"      - 진행률: {i}/{max_samples}, 변환됨: {converted_count}, 건너뜀: {skipped_count}")
        
        # 데이터셋별 메시지 형식 변환
        converted = convert_sample_to_messages(sample, dataset_name)
        if converted:
            samples.append(converted)
            converted_count += 1
        else:
            skipped_count += 1
            if skipped_count <= 5:  # 처음 5개 실패한 샘플만 출력
                logger.warning(f"      ⚠️ 샘플 {i} 변환 실패: {sample}")
    
    logger.info(f"✅ {len(samples)}개 샘플 수집 완료 (변환: {converted_count}, 건너뜀: {skipped_count})")
    
    if len(samples) == 0:
        logger.error("❌ 변환된 샘플이 없습니다. 데이터셋 형식을 확인하세요.")
        logger.info("   첫 번째 원본 샘플 예시:")
        for i, sample in enumerate(dataset):
            if i >= 3:  # 처음 3개만 출력
                break
            logger.debug(f"   샘플 {i}: {sample}")
        raise ValueError("유효한 샘플이 없습니다. 데이터셋 형식을 확인하세요.")
    
    # 훈련/테스트 분할
    split_idx = int(len(samples) * (1 - test_size))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    logger.info(f"   📊 분할: 훈련 {len(train_samples)}개, 테스트 {len(test_samples)}개")
    
    # 토크나이즈 처리
    logger.info("   🔄 토크나이징 시작...")
    train_dataset = []
    test_dataset = []
    
    tokenize_failed = 0
    for i, sample in enumerate(train_samples):
        processed = process_sample(sample, tokenizer, max_length)
        if processed is not None:
            train_dataset.append(processed)
        else:
            tokenize_failed += 1
            if tokenize_failed <= 3:  # 처음 3개 실패만 출력
                logger.warning(f"      ⚠️ 훈련 샘플 {i} 토크나이징 실패")
    
    for i, sample in enumerate(test_samples):
        processed = process_sample(sample, tokenizer, max_length)
        if processed is not None:
            test_dataset.append(processed)
        else:
            tokenize_failed += 1
    
    logger.info(f"📊 최종 결과 - 훈련: {len(train_dataset)}개, 테스트: {len(test_dataset)}개 (토크나이징 실패: {tokenize_failed}개)")
    
    if len(train_dataset) == 0:
        raise ValueError("토크나이징 후 훈련 데이터가 없습니다. 토크나이저 설정을 확인하세요.")
    
    from datasets import Dataset, DatasetDict
    return DatasetDict({
        "train": Dataset.from_list(train_dataset),
        "test": Dataset.from_list(test_dataset)
    })

def convert_sample_to_messages(sample: Dict[str, Any], dataset_name: str) -> Optional[Dict[str, Any]]:
    """샘플을 messages 형식으로 변환"""
    
    if dataset_name == "HuggingFaceTB/smoltalk" or "smoltalk" in dataset_name.lower():
        if "messages" in sample and isinstance(sample["messages"], list):
            return {"messages": sample["messages"]}
    
    elif "orca-agentinstruct" in dataset_name:
        if "messages" in sample and isinstance(sample["messages"], list):
            return {"messages": sample["messages"]}
    
    # 기본 instruction-output 형식 처리
    if "instruction" in sample and "output" in sample:
        messages = [
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]}
        ]
        return {"messages": messages}
    
    # conversations 형식 처리
    if "conversations" in sample and isinstance(sample["conversations"], list):
        messages = []
        for conv in sample["conversations"]:
            if isinstance(conv, dict) and "from" in conv and "value" in conv:
                role = "user" if conv["from"] in ["human", "user"] else "assistant"
                messages.append({"role": role, "content": conv["value"]})
        if messages:
            return {"messages": messages}
    
    # text 필드만 있는 경우 (단순한 텍스트)
    if "text" in sample and isinstance(sample["text"], str):
        # 간단한 대화로 변환
        messages = [
            {"role": "user", "content": "Continue the following text:"},
            {"role": "assistant", "content": sample["text"]}
        ]
        return {"messages": messages}
    
    return None

def process_sample(sample: Dict[str, Any], tokenizer, max_length: int):
    """샘플을 토크나이즈하여 훈련 형식으로 변환"""
    try:
        if "messages" not in sample:
            return None
        
        # 메시지 유효성 검사
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            return None
        
        # 각 메시지가 올바른 형식인지 확인
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return None
            if not isinstance(msg["content"], str) or len(msg["content"].strip()) == 0:
                return None
        
        # 채팅 템플릿 적용 - AutoProcessor 호환성 개선
        try:
            # AutoProcessor인 경우 tokenizer 속성을 통해 접근
            if hasattr(tokenizer, 'tokenizer'):
                actual_tokenizer = tokenizer.tokenizer
            else:
                actual_tokenizer = tokenizer
            
            tokenized = actual_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                max_length=max_length,
                truncation=True,
                return_dict=True,
                return_tensors="pt"
            )
        except Exception as e1:
            # 대안 방법: 채팅 템플릿 없이 직접 텍스트 변환
            logger.warning(f"   ⚠️ apply_chat_template 실패, 대안 방법 시도: {e1}")
            
            # 메시지를 직접 텍스트로 변환
            text = ""
            for msg in messages:
                if msg["role"] == "user":
                    text += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
                elif msg["role"] == "assistant":
                    text += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
            
            # 토크나이징
            if hasattr(tokenizer, 'tokenizer'):
                actual_tokenizer = tokenizer.tokenizer
            else:
                actual_tokenizer = tokenizer
            
            tokenized = actual_tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
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
        
        # 최소 길이 확인 (너무 짧은 시퀀스 제외)
        if "input_ids" in result and len(result["input_ids"]) < 10:
            return None
        
        return result
        
    except Exception as e:
        # 디버깅을 위해 예외 정보 출력 (처음 몇 개만)
        logger.error(f"❌ 토크나이징 예외: {str(e)}")
        logger.error(f"   샘플: {sample}")
        logger.error(f"   토크나이저 타입: {type(tokenizer)}")
        logger.error(f"   토크나이저에 chat_template이 있는가: {hasattr(tokenizer, 'chat_template')}")
        if hasattr(tokenizer, 'chat_template'):
            logger.error(f"   chat_template 길이: {len(str(tokenizer.chat_template)) if tokenizer.chat_template else 0}")
        import traceback
        traceback.print_exc()
        return None

def create_simple_collate_fn(tokenizer):
    """간단한 collate function"""
    tokenizer = tokenizer.tokenizer
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
    logger.info(f"데이터셋 생성 완료: {dataset}")
    logger.info(f"샘플 예시: {dataset['train'][0]}") 