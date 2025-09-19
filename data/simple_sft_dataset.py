import logging
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from transformers import AutoProcessor
import torch
from typing import Dict, Any, List, Optional
import traceback
import gc
import os
import random

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_simple_sft_dataset(
    dataset_name: str = "HuggingFaceTB/smoltalk", 
    tokenizer=None,
    max_length: int = 2048,
    max_samples: int = 1000,
    test_size: float = 0.1,
    use_streaming: bool = True,
    chunk_size: int = 100
):
    """
    메모리 효율적인 SFT 데이터셋을 로드합니다.
    모든 config를 순차적으로 처리하여 메모리 사용량을 최소화합니다.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    logger.info(f"📦 메모리 효율적 로딩: {dataset_name}")
    logger.info(f"   - max_samples: {max_samples}")
    logger.info(f"   - streaming: {use_streaming}")
    logger.info(f"   - chunk_size: {chunk_size}")
    
    log_memory_usage("데이터셋 로딩 시작")
    
    # 모든 config를 순차적으로 처리
    try:
        available_configs = get_dataset_config_names(dataset_name)
        selected_configs = []
        for i in range(25):
            random.shuffle(available_configs)
            selected_configs += [random.choice(available_configs)]
        available_configs = selected_configs
        logger.info(f"   📋 사용 가능한 configs: {len(available_configs)}개")
        logger.info(f"   🎯 모든 config 사용: {len(available_configs)}개")
        
        # config당 샘플 수를 균등하게 분배
        samples_per_config = max(1, max_samples // len(available_configs))
        logger.info(f"   📊 Config당 샘플 수: {samples_per_config}개")
        
        train_samples = []
        test_samples = []
        total_processed = 0
        
        # tqdm으로 진행 상황 표시
        config_pbar = tqdm(available_configs, desc="Config 처리", unit="config")
        
        # 각 config를 순차적으로 처리 (메모리 절약)
        for i, config in enumerate(config_pbar):
            if total_processed >= max_samples:
                break
                
            try:
                config_pbar.set_description(f"Config {i+1}/{len(available_configs)}: {config[:30]}...")
                
                # 스트리밍으로 데이터셋 로드
                config_dataset = load_dataset(
                    path=dataset_name, 
                    name=config,
                    split="train", 
                    streaming=True
                )
                
                # config에서 샘플 스트리밍 처리
                config_processed = 0
                sample_pbar = tqdm(total=samples_per_config, desc=f"샘플 처리", unit="sample", leave=False)
                
                for sample in config_dataset:
                    if config_processed >= samples_per_config or total_processed >= max_samples:
                        break
                    
                    # 샘플 변환
                    converted = convert_sample_to_messages(sample, dataset_name)
                    if "images" in converted:
                        for img in converted["images"]:
                            if img is None:
                                converted = None
                    if "messages" in converted:
                        user_msg_len = len([msg for msg in converted["messages"] if msg["role"] == "user"])
                        images_len = len(["images"])
                        if user_msg_len > 1 and images_len > 1:
                            print(converted)
                            converted = None

                    if converted:
                        # 훈련/테스트 분할
                        is_train = (total_processed % int(1/test_size)) != 0
                        
                        if is_train :
                            train_samples.append(converted)
                        else:
                            test_samples.append(converted)

                        total_processed += 1
                        config_processed += 1

                        # tqdm 업데이트
                        sample_pbar.update(1)
                        memory_gb = get_memory_usage()
                        sample_pbar.set_postfix({
                            "총 처리": f"{total_processed}/{max_samples}",
                            "Train": len(train_samples),
                            "Test": len(test_samples),
                            "메모리": f"{memory_gb:.1f}GB"
                        })
                
                sample_pbar.close()
                
                # 메모리 정리
                del config_dataset
                gc.collect()
                
            except Exception as e:
                tqdm.write(f"⚠️ Config {config} 실패: {e}")
                continue
        
        config_pbar.close()
        
        logger.info(f"✅ 샘플 수집 완료: Train {len(train_samples)}개, Test {len(test_samples)}개")
        
    except Exception as e:
        logger.error(f"❌ 데이터셋 로딩 실패: {e}")
        raise Exception(f"😢 데이터셋 로딩 시도가 실패했습니다.")
    
    if len(train_samples) == 0:
        raise ValueError("변환된 훈련 샘플이 없습니다. 데이터셋 형식을 확인하세요.")
    
    # Dataset으로 변환 (최소한의 메모리 사용)
    print("📊 Dataset 변환 시작...")
    
    from datasets import Dataset, DatasetDict
    
    # tqdm으로 Dataset 변환 진행 상황 표시
    with tqdm(total=2, desc="Dataset 변환", unit="dataset") as pbar:
        pbar.set_description("Train Dataset 생성")
        train_dataset = Dataset.from_list(train_samples)
        pbar.update(1)
        
        pbar.set_description("Test Dataset 생성")
        test_dataset = Dataset.from_list(test_samples)
        pbar.update(1)
    
    # 원본 리스트 메모리 해제
    del train_samples, test_samples
    gc.collect()
    
    print("✅ 메모리 효율적 데이터셋 생성 완료")
    
    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })


def convert_sample_to_messages(sample: Dict[str, Any], dataset_name: str) -> Optional[Dict[str, Any]]:
    """샘플을 messages 형식으로 변환"""
    
    if dataset_name == "HuggingFaceTB/smoltalk" or "smoltalk" in dataset_name.lower():
        if "messages" in sample and isinstance(sample["messages"], list):
            img = sample.get("image", [])
            if not isinstance(img, list):
                img = [img]
            return {"messages": sample["messages"], "image": img}
    
    elif "orca-agentinstruct" in dataset_name:
        if "messages" in sample and isinstance(sample["messages"], list):
            img = sample.get("image", [])
            if not isinstance(img, list):
                img = [img]
            
            sample.update({"messages": sample["messages"], "images": img})
            return sample
    
    # 기본 instruction-output 형식 처리
    if "instruction" in sample and "output" in sample:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["instruction"]}]},
            {"role": "assistant", "content": sample["output"]}
        ]
        img = sample.get("image", [])
        if not isinstance(img, list):
            img = [img]
        return{"messages": messages, "images": img}
    
    # conversations 형식 처리
    if "conversations" in sample and isinstance(sample["conversations"], list):
        messages = []
        first_user_message_found = False
        for conv in sample["conversations"]:
            if isinstance(conv, dict) and "from" in conv and "value" in conv:
                role = "user" if conv["from"] in ["human", "user"] else "assistant"
                content = []
                if role == "user" and not first_user_message_found:
                    content.append({"type": "image"})
                    first_user_message_found = True
                content.append({"type": "text", "text": conv["value"]})
                messages.append({"role": role, "content": content})
        if messages:
            img = sample.get("image", [])
            if not isinstance(img, list):
                img = [img]
            return {"messages": messages, "images": img}
    
    # text 필드만 있는 경우 (단순한 텍스트)
    if "text" in sample and isinstance(sample["text"], str):
        # 간단한 대화로 변환
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Continue the following text:"}]},
            {"role": "assistant", "content": sample["text"]}
        ]
        img = sample.get("image", [])
        if not isinstance(img, list):
            img = [img]
        return {"messages": messages, "images": img}
    
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
                # max_length=max_length,
                # truncation=True,
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
                # truncation=True,
                # padding=False,
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

def create_memory_efficient_collate_fn(tokenizer, max_length: int = 2048):
    """메모리 효율적인 collate function"""
    if hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    else:
        actual_tokenizer = tokenizer
    
    def collate_fn(examples):
        # 메모리 효율적인 배치 처리
        batch_input_ids = []
        batch_attention_mask = []
        
        for ex in examples:
            if "messages" in ex:
                # 실시간 토크나이징 (메모리 효율적)
                try:
                    tokenized = actual_tokenizer.apply_chat_template(
                        ex["messages"],
                        tokenize=True,
                        add_generation_prompt=False,
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    if tokenized is not None and "input_ids" in tokenized:
                        batch_input_ids.append(tokenized["input_ids"].squeeze())
                        batch_attention_mask.append(tokenized["attention_mask"].squeeze())
                except Exception as e:
                    logger.warning(f"토크나이징 실패: {e}")
                    continue
        
        if not batch_input_ids:
            return None
        
        # 패딩 처리
        from torch.nn.utils.rnn import pad_sequence
        
        input_ids = pad_sequence(
            batch_input_ids, 
            batch_first=True, 
            padding_value=actual_tokenizer.pad_token_id or actual_tokenizer.eos_token_id
        )
        attention_mask = pad_sequence(
            batch_attention_mask, 
            batch_first=True, 
            padding_value=0
        )
        
        # labels는 input_ids와 동일 (causal LM)
        labels = input_ids.clone()
        
        # 패딩 토큰은 loss 계산에서 제외
        pad_token_id = actual_tokenizer.pad_token_id or actual_tokenizer.eos_token_id
        labels[labels == pad_token_id] = -100
        
        # 메모리 정리
        del batch_input_ids, batch_attention_mask
        gc.collect()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    return collate_fn

def create_simple_collate_fn(tokenizer):
    """간단한 collate function (기존 호환성 유지)"""
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

def get_memory_usage():
    """현재 메모리 사용량을 반환합니다."""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024 * 1024)  # GB 단위

def log_memory_usage(stage: str):
    """메모리 사용량을 로깅합니다."""
    memory_gb = get_memory_usage()
    logger.info(f"   💾 {stage} - 메모리 사용량: {memory_gb:.2f} GB")

# 테스트용 작은 데이터셋 빌더 함수들
def smoltalk_dataset(tokenizer, max_samples: int = 500, use_streaming: bool = True):
    """SmolTalk 데이터셋 빌더 (메모리 효율적)"""
    log_memory_usage("SmolTalk 데이터셋 시작")
    dataset = get_simple_sft_dataset(
        dataset_name="HuggingFaceTB/smoltalk",
        tokenizer=tokenizer,
        max_samples=max_samples,
        use_streaming=use_streaming,
        chunk_size=50
    )
    log_memory_usage("SmolTalk 데이터셋 완료")
    return dataset

def orca_mini_dataset(tokenizer, max_samples: int = 500, use_streaming: bool = True):
    """Orca 미니 데이터셋 빌더 (메모리 효율적)"""
    log_memory_usage("Orca 데이터셋 시작")
    dataset = get_simple_sft_dataset(
        dataset_name="microsoft/orca-agentinstruct-1M-v1",
        tokenizer=tokenizer, 
        max_samples=max_samples,
        use_streaming=use_streaming,
        chunk_size=50
    )
    log_memory_usage("Orca 데이터셋 완료")
    return dataset

def create_memory_efficient_dataset(
    dataset_name: str,
    tokenizer,
    max_samples: int = 1000,
    chunk_size: int = 50,
    use_streaming: bool = True
):
    """메모리 효율적인 데이터셋 생성기"""
    log_memory_usage(f"{dataset_name} 데이터셋 시작")
    
    dataset = get_simple_sft_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_samples=max_samples,
        use_streaming=use_streaming,
        chunk_size=chunk_size
    )
    
    log_memory_usage(f"{dataset_name} 데이터셋 완료")
    return dataset

def get_available_configs(dataset_name: str) -> List[str]:
    """데이터셋의 사용 가능한 config 목록을 반환합니다."""
    try:
        configs = get_dataset_config_names(dataset_name)
        logger.info(f"📋 {dataset_name} 사용 가능한 configs ({len(configs)}개):")
        for i, config in enumerate(configs[:10]):  # 처음 10개만 출력
            logger.info(f"   {i+1}. {config}")
        if len(configs) > 10:
            logger.info(f"   ... 및 {len(configs) - 10}개 더")
        return configs
    except Exception as e:
        logger.error(f"❌ config 목록 가져오기 실패: {e}")
        return []


if __name__ == "__main__":
    # 메모리 효율적 테스트
    from transformers import AutoTokenizer
    
    logger.info("🚀 메모리 효율적 데이터셋 테스트 시작")
    log_memory_usage("프로그램 시작")
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_memory_usage("토크나이저 로드 후")
    
    # Config 처리 확인을 위한 테스트
    try:
        logger.info("📋 사용 가능한 config 확인")
        configs = get_available_configs("HuggingFaceTB/smoltalk")
        logger.info(f"총 {len(configs)}개 config 발견")
    except Exception as e:
        logger.error(f"Config 확인 실패: {e}")
    
    # SmolTalk 데이터셋 테스트 (스트리밍) - 모든 config 처리
    try:
        logger.info("📦 SmolTalk 데이터셋 테스트 (스트리밍 - 모든 config)")
        dataset = smoltalk_dataset(tokenizer, max_samples=100, use_streaming=True)
        log_memory_usage("SmolTalk 데이터셋 생성 후")
        
        logger.info(f"데이터셋 생성 완료: {dataset}")
        
        # 첫 번째 샘플 확인
        try:
            first_sample = dataset['train'][0]
            logger.info(f"샘플 예시: {first_sample}")
        except Exception as e:
            logger.error(f"샘플 접근 실패: {e}")
            
    except Exception as e:
        logger.error(f"SmolTalk 데이터셋 테스트 실패: {e}")
    
    # 일반 데이터셋 테스트 (비스트리밍)
    try:
        logger.info("📦 SmolTalk 데이터셋 테스트 (일반)")
        dataset2 = smoltalk_dataset(tokenizer, max_samples=100, use_streaming=False)
        log_memory_usage("SmolTalk 일반 데이터셋 생성 후")
        
        logger.info(f"일반 데이터셋 생성 완료: {dataset2}")
            
    except Exception as e:
        logger.error(f"SmolTalk 일반 데이터셋 테스트 실패: {e}")
    
    log_memory_usage("테스트 완료")
    logger.info("✅ 메모리 효율적 테스트 완료")
