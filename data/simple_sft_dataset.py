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
import tempfile
import pathlib
import shutil
import json
import hashlib
from datetime import datetime
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset, Image as DatasetImage, Sequence, Features, Value

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def _generate_simple_cache_key(dataset_name: str, max_samples: int, test_size: float, use_streaming: bool) -> str:
    """캐시 키 생성"""
    cache_data = {
        "dataset_name": dataset_name,
        "max_samples": max_samples,
        "test_size": test_size,
        "use_streaming": use_streaming
    }
    cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
    cache_hash = hashlib.md5(cache_str.encode('utf-8')).hexdigest()
    return f"simple_{cache_hash}"

def get_simple_sft_dataset(
    dataset_name: str = "HuggingFaceTB/smoltalk", 
    tokenizer=None,
    max_length: int = 2048,
    max_samples: int = 1000,
    test_size: float = 0.1,
    use_streaming: bool = True,
    chunk_size: int = 1000,
    use_cache: bool = True
):
    """
    메모리 효율적인 SFT 데이터셋을 로드합니다.
    모든 config를 순차적으로 처리하여 메모리 사용량을 최소화합니다.
    데이터를 청크 단위로 디스크에 저장하여 메모리 초과를 방지합니다.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    # 캐시 키 생성
    cache_key = _generate_simple_cache_key(dataset_name, max_samples, test_size, use_streaming)
    base_temp_dir = "/mls/conan/tmp"
    cache_dir = os.path.join(base_temp_dir, "cache", cache_key)
    cache_train_path = os.path.join(cache_dir, "train.jsonl")
    cache_test_path = os.path.join(cache_dir, "test.jsonl")
    cache_images_dir = os.path.join(cache_dir, "images")
    cache_meta_path = os.path.join(cache_dir, "cache_meta.json")
    
    # 캐시 확인
    if use_cache and os.path.exists(cache_train_path) and os.path.exists(cache_test_path):
        # 파일 크기 확인 (빈 파일이 아닌지)
        if os.path.getsize(cache_train_path) > 0:
            logger.info(f"💾 캐시된 데이터셋 발견: {cache_key}")
            logger.info(f"   - 캐시 디렉토리: {cache_dir}")
            
            # 메타데이터 확인
            if os.path.exists(cache_meta_path):
                try:
                    with open(cache_meta_path, "r", encoding="utf-8") as f:
                        cache_meta = json.load(f)
                        logger.info(f"   - 캐시 생성 시간: {cache_meta.get('created_at', 'N/A')}")
                        logger.info(f"   - Train 샘플 수: {cache_meta.get('train_count', 'N/A')}")
                        logger.info(f"   - Test 샘플 수: {cache_meta.get('test_count', 'N/A')}")
                except Exception as e:
                    logger.warning(f"   ⚠️ 캐시 메타데이터 읽기 실패: {e}")
            
            # 캐시된 파일로부터 데이터셋 로드
            try:
                data_files = {}
                if os.path.exists(cache_train_path) and os.path.getsize(cache_train_path) > 0:
                    data_files["train"] = cache_train_path
                if os.path.exists(cache_test_path) and os.path.getsize(cache_test_path) > 0:
                    data_files["test"] = cache_test_path
                
                if not data_files:
                    logger.warning("   ⚠️ 캐시 파일이 비어있습니다. 재처리합니다.")
                    raise FileNotFoundError("Cache files are empty")
                
                logger.info("🧠 캐시된 JSONL 파일로부터 데이터셋 로딩...")
                
                # JSONL 파일을 읽어서 리스트로 만든 후 Dataset.from_list 사용
                dataset_dict = DatasetDict()
                for split_name, file_path in data_files.items():
                    logger.info(f"   📂 {split_name} split 로딩 중...")
                    
                    # JSONL 파일 읽기 및 정제
                    records = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            try:
                                record = json.loads(line.strip())
                                
                                # messages 정규화
                                if 'messages' in record and isinstance(record['messages'], list):
                                    for message in record['messages']:
                                        if not isinstance(message, dict):
                                            continue
                                        if 'content' in message and isinstance(message['content'], list):
                                            for content_item in message['content']:
                                                if not isinstance(content_item, dict):
                                                    continue
                                                # text 필드가 없거나 None이면 빈 문자열로
                                                if 'text' not in content_item or content_item.get('text') is None:
                                                    content_item['text'] = ""
                                                # type 필드가 없으면 "text"로
                                                if 'type' not in content_item:
                                                    content_item['type'] = "text"
                                
                                # images 필드 정규화 (없으면 빈 리스트, 문자열 리스트로 보장)
                                if 'images' not in record:
                                    record['images'] = []
                                elif record['images'] is None:
                                    record['images'] = []
                                elif not isinstance(record['images'], list):
                                    record['images'] = []
                                else:
                                    # 이미지 경로가 문자열인지 확인
                                    record['images'] = [str(img) if img is not None else "" for img in record['images'] if img is not None]
                                
                                records.append(record)
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.debug(f"   ⚠️ {split_name} 파일 {line_num}번째 줄 JSON 파싱 실패, 건너뜀: {e}")
                                continue
                    
                    # Dataset.from_list로 생성
                    if records:
                        dataset_dict[split_name] = Dataset.from_list(records)
                        logger.info(f"   ✅ {split_name} split 로드 완료: {len(dataset_dict[split_name])}개 샘플")
                    else:
                        logger.warning(f"   ⚠️ {split_name} split에 유효한 샘플이 없습니다.")
                        # 빈 데이터셋 생성
                        dataset_dict[split_name] = Dataset.from_list([])
                
                # 이미지 경로 처리
                logger.info("🖼️ 이미지 경로를 이미지 객체로 캐스팅 (lazy loading)...")
                for split in dataset_dict:
                    def preprocess_images(example):
                        """이미지 데이터 전처리 - 중첩 리스트 평면화"""
                        if 'images' in example and example['images']:
                            # 이미지 경로를 절대 경로로 변환
                            image_paths = example['images']
                            if isinstance(image_paths, list):
                                fixed_paths = []
                                for img_path in image_paths:
                                    if isinstance(img_path, str) and img_path.strip():
                                        if not os.path.isabs(img_path):
                                            img_path = os.path.join(cache_images_dir, os.path.basename(img_path))
                                        # 파일 존재 확인
                                        if os.path.exists(img_path):
                                            fixed_paths.append(img_path)
                                example['images'] = validate_image_data(fixed_paths)
                            else:
                                example['images'] = validate_image_data(example['images']) if example['images'] else []
                        elif 'images' not in example:
                            example['images'] = []
                        return example
                    
                    # 이미지 전처리 적용
                    dataset_dict[split] = dataset_dict[split].map(preprocess_images)
                    
                    # 이미지 필드를 DatasetImage로 캐스팅
                    current_features = dataset_dict[split].features
                    new_features = current_features.copy()
                    if 'images' in new_features:
                        new_features['images'] = Sequence(DatasetImage(decode=True))
                        dataset_dict[split] = dataset_dict[split].cast(new_features)
                
                logger.info("✅ 캐시된 데이터셋 로드 완료")
                return dataset_dict
                
            except Exception as e:
                logger.warning(f"   ⚠️ 캐시 로드 실패, 재처리합니다: {e}")
                traceback.print_exc()
                # 캐시 로드 실패 시 기존 로직으로 진행
    
    # 캐시가 없거나 사용하지 않는 경우 기존 처리 로직
    logger.info(f"📦 메모리 효율적 로딩 (V2 - JSONL): {dataset_name}")
    logger.info(f"   - max_samples: {max_samples}")
    logger.info(f"   - streaming: {use_streaming}")
    
    log_memory_usage("데이터셋 로딩 시작")
    
    # 캐시 디렉토리 사용
    os.makedirs(cache_dir, exist_ok=True)
    images_dir = cache_images_dir
    os.makedirs(images_dir, exist_ok=True)

    try:
        available_configs = get_dataset_config_names(dataset_name)
        selected_configs = []
        for i in range(25):
            random.shuffle(available_configs)
            selected_configs += [random.choice(available_configs)]
        available_configs = selected_configs
        logger.info(f"   📋 사용 가능한 configs: {len(available_configs)}개")
        logger.info(f"   🎯 모든 config 사용: {len(available_configs)}개")
        
        samples_per_config = max(1, max_samples // len(available_configs))
        logger.info(f"   📊 Config당 샘플 수: {samples_per_config}개")
        
        total_processed = 0
        image_counter = 0
        train_count, test_count = 0, 0
        
        train_jsonl_path = cache_train_path
        test_jsonl_path = cache_test_path

        with open(train_jsonl_path, "w", encoding="utf-8") as train_f, \
             open(test_jsonl_path, "w", encoding="utf-8") as test_f:
            
            config_pbar = tqdm(available_configs, desc="Config 처리", unit="config")
            
            for i, config in enumerate(config_pbar):
                if total_processed >= max_samples:
                    break
                    
                try:
                    config_pbar.set_description(f"Config {i+1}/{len(available_configs)}: {config[:30]}...")
                    
                    config_dataset = load_dataset(
                        path=dataset_name, 
                        name=config,
                        split="train", 
                        streaming=True
                    )
                    
                    sample_pbar = tqdm(total=samples_per_config, desc=f"샘플 처리", unit="sample", leave=False)
                    
                    for sample in config_dataset:
                        if total_processed >= max_samples:
                            break
                        
                        converted = convert_sample_to_messages(sample, dataset_name)
                        if not converted:
                            continue

                        # 이미지가 있는지 먼저 확인 - 이미지가 없으면 샘플 건너뜀
                        if "images" not in converted or not converted["images"]:
                            logger.debug(f"⚠️ 이미지가 없는 샘플 건너뜀: {sample}")
                            continue
                        
                        # 이미지 리스트가 중첩된 경우 평면화
                        flattened_images = validate_image_data(converted["images"])
                        if not flattened_images:
                            logger.debug(f"⚠️ 유효한 이미지가 없는 샘플 건너뜀: {sample}")
                            continue

                        # 이미지 파일로 저장하고 경로로 대체
                        image_paths = []
                        valid_sample = True
                        
                        for img_obj in flattened_images:
                            if isinstance(img_obj, Image.Image):
                                try:
                                    img_path = os.path.join(images_dir, f"{image_counter}.png")
                                    img_obj.save(img_path, "PNG")
                                    image_paths.append(img_path)
                                    image_counter += 1
                                except Exception as img_e:
                                    logger.warning(f"⚠️ 이미지 저장 실패, 샘플 건너뜀: {img_e}")
                                    valid_sample = False
                                    break
                            elif img_obj is not None:
                                # None이 아닌 다른 타입의 이미지 객체 처리
                                logger.warning(f"⚠️ 지원되지 않는 이미지 타입: {type(img_obj)}")
                                valid_sample = False
                                break
                        
                        if not valid_sample or not image_paths:
                            logger.debug(f"⚠️ 이미지 처리 실패로 샘플 건너뜀: {sample}")
                            continue
                        
                        converted["images"] = image_paths

                        is_train = (total_processed % int(1/test_size)) != 0
                        
                        if is_train:
                            train_f.write(json.dumps(converted) + "\n")
                            train_count += 1
                        else:
                            test_f.write(json.dumps(converted) + "\n")
                            test_count += 1

                        total_processed += 1
                        
                        sample_pbar.update(1)
                        memory_gb = get_memory_usage()
                        sample_pbar.set_postfix({
                            "총 처리": f"{total_processed}/{max_samples}",
                            "Train": train_count,
                            "Test": test_count,
                            "메모리": f"{memory_gb:.1f}GB"
                        })
                    
                    sample_pbar.close()
                    del config_dataset
                    gc.collect()
                    
                except Exception as e:
                    tqdm.write(f"⚠️ Config {config} 실패: {e}")
                    continue
            
            config_pbar.close()

        logger.info(f"✅ 샘플 수집 및 디스크 저장 완료: Train {train_count}개, Test {test_count}개")
        
        data_files = {}
        if train_count > 0:
            data_files["train"] = train_jsonl_path
        if test_count > 0:
            data_files["test"] = test_jsonl_path

        if not data_files:
            raise ValueError("변환된 훈련 샘플이 없습니다. 데이터셋 형식을 확인하세요.")
        
        logger.info("🧠 JSONL 파일로부터 데이터셋 로딩...")
        dataset_dict = load_dataset("json", data_files=data_files)
        
        logger.info("🖼️ 이미지 경로를 이미지 객체로 캐스팅 (lazy loading)...")
        for split in dataset_dict:
            # 새로운 Features 객체 생성
            current_features = dataset_dict[split].features
            new_features = current_features.copy()
            if 'images' in new_features and isinstance(new_features['images'], Sequence):
                # 중첩 리스트 문제를 방지하기 위해 이미지 데이터 전처리
                def preprocess_images(example):
                    """이미지 데이터 전처리 - 중첩 리스트 평면화"""
                    if 'images' in example and example['images']:
                        example['images'] = validate_image_data(example['images'])
                    return example
                
                # 이미지 전처리 적용
                dataset_dict[split] = dataset_dict[split].map(preprocess_images)
                new_features['images'] = Sequence(DatasetImage(decode=True))
                dataset_dict[split] = dataset_dict[split].cast(new_features)

        logger.info("✅ 메모리 효율적 데이터셋 생성 완료")
        
        # 처리 완료 후 메타데이터 저장
        try:
            cache_meta = {
                "created_at": datetime.now().isoformat(),
                "train_count": train_count,
                "test_count": test_count,
                "dataset_name": dataset_name,
                "max_samples": max_samples,
                "test_size": test_size,
                "use_streaming": use_streaming,
                "cache_key": cache_key
            }
            with open(cache_meta_path, "w", encoding="utf-8") as f:
                json.dump(cache_meta, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 데이터셋 캐시 저장 완료: {cache_dir}")
        except Exception as e:
            logger.warning(f"⚠️ 캐시 메타데이터 저장 실패: {e}")
        
        return dataset_dict

    except Exception as e:
        logger.error(f"❌ 데이터셋 로딩 실패: {e}")
        traceback.print_exc()
        # 실패 시에도 캐시 디렉토리는 유지 (부분적으로 처리된 데이터가 있을 수 있음)
        # shutil.rmtree(cache_dir, ignore_errors=True)
        raise Exception(f"😢 데이터셋 로딩 시도가 실패했습니다.") from e


def convert_sample_to_messages(sample: Dict[str, Any], dataset_name: str) -> Optional[Dict[str, Any]]:
    """샘플을 messages 형식으로 변환"""
    
    # safe_flatten_images 함수 사용
    
    if dataset_name == "HuggingFaceTB/smoltalk" or "smoltalk" in dataset_name.lower():
        if "messages" in sample and isinstance(sample["messages"], list):
            img = sample.get("image", [])
            if not isinstance(img, list):
                img = [img] if img is not None else []
            # 중첩 리스트 평면화 및 None 값 제거
            img = validate_image_data(img)
            # 이미지가 없으면 None 반환 (샘플 건너뜀)
            if not img:
                return None
            
            # 메시지 검증 및 최적화
            messages = validate_messages(sample["messages"])
            return {"messages": messages, "images": img}
    
    elif "orca-agentinstruct" in dataset_name:
        if "messages" in sample and isinstance(sample["messages"], list):
            img = sample.get("image", [])
            if not isinstance(img, list):
                img = [img] if img is not None else []
            # 중첩 리스트 평면화 및 None 값 제거
            img = validate_image_data(img)
            
            # 메시지 검증 및 최적화
            messages = validate_messages(sample["messages"])
            sample.update({"messages": messages, "images": img})
            return sample
    
    # 기본 instruction-output 형식 처리
    if "instruction" in sample and "output" in sample:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["instruction"]}]},
            {"role": "assistant", "content": sample["output"]}
        ]
        img = sample.get("image", [])
        if not isinstance(img, list):
            img = [img] if img is not None else []
        # 중첩 리스트 평면화 및 None 값 제거
        img = validate_image_data(img)
        # 이미지가 없으면 None 반환 (샘플 건너뜀)
        if not img:
            return None
        return {"messages": messages, "images": img}
    
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
                img = [img] if img is not None else []
            # 중첩 리스트 평면화 및 None 값 제거
            img = validate_image_data(img)
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
            img = [img] if img is not None else []
        # 중첩 리스트 평면화 및 None 값 제거
        img = validate_image_data(img)
        # 이미지가 없으면 None 반환 (샘플 건너뜀)
        if not img:
            return None
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
    
    # safe_flatten_images 함수 사용
    
    def collate_fn(examples):
        # 메모리 효율적인 배치 처리
        batch_input_ids = []
        batch_attention_mask = []
        
        for ex in examples:
            if "messages" in ex:
                # 이미지 데이터가 있는 경우 안전하게 처리
                if "images" in ex and ex["images"]:
                    # 중첩 리스트 문제 해결
                    ex["images"] = validate_image_data(ex["images"])
                
                # 실시간 토크나이징 (메모리 효율적)
                try:
                    # 먼저 텍스트로 변환
                    text = actual_tokenizer.apply_chat_template(
                        ex["messages"],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    
                    # 그 다음 토크나이징
                    tokenized = actual_tokenizer(
                        text,
                        max_length=max_length,
                        truncation=True,
                        padding=False,
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

def create_simple_collate_fn(processor, max_length: int = 2048):
    """SFTTrainer용 커스텀 data collator - 이미지 중첩 리스트 문제 해결"""
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
    
    class CustomSFTDataCollator(DataCollatorForVisionLanguageModeling):
        def __init__(self, processor, max_length: int = 2048):
            super().__init__(processor = processor, max_length = max_length)
            self.processor = processor
            self.max_length = max_length
            
        def __call__(self, features):
            # 이미지 데이터 검증 - 이미지가 없는 샘플은 오류 발생
            assert features is not None, "features is None"
            batch_images = []
            batch_messages = []

            for i, feature in enumerate(features):
                if "messages" in feature:
                    feature["messages"] = validate_messages(feature["messages"])
                    # batch_messages.append(
                    #     self.processor.apply_chat_template(
                    #         feature["messages"], 
                    #         add_generation_prompt=False, 
                    #         tokenize=False)
                    #     )
                if 'images' not in feature or not feature['images']:
                    raise ValueError(f"샘플 {i}에 이미지가 없습니다! 모든 샘플은 이미지를 포함해야 합니다.")
                
                # 중첩 리스트 문제 해결
                feature['images'] = validate_image_data(feature['images'])
                if not feature['images']:
                    raise ValueError(f"샘플 {i}의 이미지가 유효하지 않습니다!")
                # batch_images.append(feature['images'])
            # processor를 사용하여 데이터 처리
            try:
                return self.torch_call(
                    examples=features
                )

                # return self.processor(
                #     images=safe_flatten_images(batch_images),
                #     text=batch_messages)


            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"⚠️ Processor 처리 중 오류: {e}")
                print(features)
                # 오류 발생 시 기본 처리
                return features
    
    return CustomSFTDataCollator(processor)

def validate_messages(messages):
    """메시지 데이터의 유효성을 검사하고 중첩 리스트 문제를 해결합니다."""
    for message in messages:
        content = message.get("content")
        if not content or not isinstance(content, list):
            continue
            
        # 빠른 필터링: image 타입에서 text 키만 제거
        for item in content:
            if (isinstance(item, dict) and 
                item.get("type") == "image" and 
                "text" in item):
                item.pop("text", None)
    
    return messages

def safe_flatten_images(images):
    """
    이미지 리스트를 안전하게 평면화하여 중첩 리스트 문제를 해결합니다.
    transformers의 image_utils.py에서 발생하는 ValueError를 방지합니다.
    """
    if not images:
        return []
    
    flattened = []
    for img in images:
        if isinstance(img, list):
            # 중첩된 리스트인 경우 재귀적으로 평면화
            flattened.extend(safe_flatten_images(img))
        elif img is not None:
            flattened.append(img)
    
    return flattened

def validate_image_data(images):
    """
    이미지 데이터의 유효성을 검사하고 중첩 리스트 문제를 해결합니다.
    """
    if images is None:
        return []
    
    if not images:
        return []
    
    # 중첩 리스트 평면화
    flattened = safe_flatten_images(images)
    
    # None 값 제거
    valid_images = [img for img in flattened if img is not None]
    
    return valid_images

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
