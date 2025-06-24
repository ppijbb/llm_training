from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, Sequence, load_from_disk, DatasetDict
import json
from typing import List, Dict, Any, Optional, cast
from tqdm.auto import tqdm
import os
import requests
from urllib.parse import urlparse
import hashlib
import gc
import datetime
import argparse
import sys
import pandas as pd
import tempfile
import shutil

# 멀티모달 데이터셋 목록
dataset_configs = [
    ("HuggingFaceTB/smoltalk", "all"),
    ("R0k1e/UltraLink", None),
    ("PrincetonPLI/Instruct-SkillMix-SDD", None),
    ("allenai/WildChat-1M", None),
    ("nvidia/OpenCodeInstruct", None),
    ("microsoft/orca-agentinstruct-1M-v1", "default"),
    ("MaziyarPanahi/Llama-Nemotron-Post-Training-Dataset-v1-ShareGPT", "default"),
    ("nvidia/Llama-Nemotron-Post-Training-Dataset", "SFT"),
    ("open-r1/Mixture-of-Thoughts", "all"),
    ("Salesforce/blip3-kale", "core"),
    ("liuhaotian/LLaVA-Instruct-150K", None),
    ("Lin-Chen/ShareGPT4V", "ShareGPT4V")
]

def construct_image_url(image_path, dataset_name):
    """데이터셋별로 이미지 경로를 실제 URL로 변환"""
    if dataset_name == "Lin-Chen/ShareGPT4V":
        if image_path.startswith('coco/'):
            filename = os.path.basename(image_path)
            return f"http://images.cocodataset.org/train2017/{filename}"
    elif dataset_name == "liuhaotian/LLaVA-Instruct-150K":
        if not image_path.startswith('http'):
            return f"http://images.cocodataset.org/train2017/{image_path}"
    
    return None

def validate_image_url(image_url):
    """이미지 URL이 유효한지 간단히 확인"""
    try:
        parsed = urlparse(image_url)
        return bool(parsed.scheme) and bool(parsed.netloc)
    except:
        return False

def convert_to_target_format_url(sample: Dict[str, Any], dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    각 데이터셋의 샘플을 목표 형식으로 변환합니다. (URL 방식)
    이미지는 실제 다운로드하지 않고 URL만 저장합니다.
    
    목표 형식:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "질문"},
                    {"type": "image_url", "image_url": {"url": "https://..."}}
                ]
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": "답변"}]
            }
        ],
        "source_dataset": "dataset_name",
        "original_data": {...}
    }
    """
    
    result: Dict[str, Any] = {
        "messages": [],
        "source_dataset": dataset_name,
        "original_data": sample.copy()
    }
    
    try:
        # 텍스트 전용 데이터셋들 처리 (기존과 동일)
        if dataset_name == "HuggingFaceTB/smoltalk":
            if "messages" in sample and isinstance(sample["messages"], list):
                for msg in sample["messages"]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        result["messages"].append({
                            "role": msg["role"],
                            "content": [{"type": "text", "text": str(msg["content"])}]
                        })
        
        elif dataset_name == "R0k1e/UltraLink":
            if "data" in sample and isinstance(sample["data"], list) and len(sample["data"]) >= 2:
                data = sample["data"]
                for i in range(0, len(data), 2):
                    if i + 1 < len(data):
                        result["messages"].extend([
                            {"role": "user", "content": [{"type": "text", "text": str(data[i])}]},
                            {"role": "assistant", "content": [{"type": "text", "text": str(data[i + 1])}]}
                        ])
        
        # ... 다른 텍스트 데이터셋들도 동일 ...
        
        # 멀티모달 데이터셋들 처리 (URL 방식으로 수정)
        elif dataset_name in ["Lin-Chen/ShareGPT4V", "liuhaotian/LLaVA-Instruct-150K"]:
            # 이미지 URL 추출
            image_url = None
            if "image" in sample and sample["image"] is not None:
                if isinstance(sample["image"], str):
                    if sample["image"].startswith('http'):
                        image_url = sample["image"]
                    else:
                        image_url = construct_image_url(sample["image"], dataset_name)
            elif "images" in sample and sample["images"] is not None:
                if isinstance(sample["images"], list) and len(sample["images"]) > 0:
                    img_path = sample["images"][0]
                    if isinstance(img_path, str):
                        if img_path.startswith('http'):
                            image_url = img_path
                        else:
                            image_url = construct_image_url(img_path, dataset_name)
                elif isinstance(sample["images"], str):
                    if sample["images"].startswith('http'):
                        image_url = sample["images"]
                    else:
                        image_url = construct_image_url(sample["images"], dataset_name)
            
            # URL 유효성 검사
            if image_url and not validate_image_url(image_url):
                image_url = None
            
            # conversations 처리
            if "conversations" in sample and isinstance(sample["conversations"], list):
                for i, conv in enumerate(sample["conversations"]):
                    if not isinstance(conv, dict):
                        continue
                        
                    # role 결정
                    role = "assistant"
                    if "from" in conv:
                        if conv["from"] in ["human", "user"]:
                            role = "user"
                        elif conv["from"] in ["gpt", "assistant"]:
                            role = "assistant"
                    
                    # content 생성
                    content_list = []
                    text_content = str(conv.get("value", ""))
                    
                    if text_content:
                        # <image> 태그 제거
                        text_content = text_content.replace("<image>", "").strip()
                        
                        if text_content:
                            content_list.append({
                                "type": "text",
                                "text": text_content
                            })
                    
                    # 첫 번째 user 메시지에 이미지 URL 추가
                    if role == "user" and i == 0 and image_url:
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })
                    
                    if content_list:
                        result["messages"].append({
                            "role": role,
                            "content": content_list
                        })
        
        elif dataset_name == "Salesforce/blip3-kale":
            # 이미지 URL 추출
            image_url = None
            if "url" in sample and sample["url"]:
                if validate_image_url(sample["url"]):
                    image_url = sample["url"]
            elif "image" in sample and sample["image"]:
                if isinstance(sample["image"], str) and validate_image_url(sample["image"]):
                    image_url = sample["image"]
            
            # caption을 대화 형식으로 변환
            caption = str(sample.get("caption", "")).strip()
            if not caption:
                caption = str(sample.get("cogvlm_caption", "")).strip()
            
            if caption:
                user_content: List[Dict[str, Any]] = [{"type": "text", "text": "Describe this image."}]
                if image_url:
                    user_content.append({
                        "type": "image_url", 
                        "image_url": {"url": image_url}
                    })
                
                result["messages"] = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": caption}]}
                ]
        
        # 빈 messages인 경우 None 반환
        if not result["messages"]:
            return None
            
        return result
        
    except Exception as e:
        print(f"샘플 변환 중 오류 (건너뛰기): {dataset_name} - {str(e)}")
        return None

def process_dataset_url(dataset_name: str, config_name: Optional[str] = None, max_samples: Optional[int] = None):
    """데이터셋을 처리하여 목표 형식으로 변환합니다. (URL 방식 - 빠른 처리)"""
    try:
        # 데이터셋 로드 로직은 동일
        if dataset_name == "microsoft/orca-agentinstruct-1M-v1":
            split = "creative_content"
        elif dataset_name == "MaziyarPanahi/Llama-Nemotron-Post-Training-Dataset-v1-ShareGPT":
            split = "chat"
        elif dataset_name == "nvidia/Llama-Nemotron-Post-Training-Dataset":
            split = "chat"
        else:
            split = "train"
        
        try:
            if config_name:
                full_dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
            else:
                full_dataset = load_dataset(dataset_name, split=split, streaming=True)
        except Exception as e:
            print(f"❌ 데이터셋 로드 실패: {e}")
            return

        success_count = 0
        total_count = 0
        
        desc = f"{dataset_name.split('/')[-1]}"
        if config_name:
            desc += f"({config_name})"
        
        progress_bar = tqdm(desc=desc, unit="samples", leave=False)
        
        batch = []
        batch_size = 1000  # URL 방식은 빠르므로 더 큰 배치 사용
        
        for sample in full_dataset:
            if max_samples and total_count >= max_samples:
                break
            
            # URL 방식은 단일 스레드로도 충분히 빠름
            converted = convert_to_target_format_url(sample, dataset_name)
            if converted:
                batch.append(converted)
                success_count += 1
            
            total_count += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "processed": f"{success_count}/{total_count}",
                "success_rate": f"{success_count/total_count*100:.1f}%"
            })
            
            # 배치가 찼으면 yield
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # 남은 배치 처리
        if batch:
            yield batch
        
        progress_bar.close()
        yield f"✅ {dataset_name}: {success_count}/{total_count} 샘플 변환 완료 (성공률: {success_count/total_count*100:.1f}%)" if total_count > 0 else f"ℹ️ {dataset_name}: 처리할 샘플 없음"

    except Exception as e:
        yield f"❌ {dataset_name} 처리 중 오류: {str(e)}"

def merge_and_create_dataset_url(
    output_name: str = "unified-multimodal-sft-url", 
    max_samples_per_dataset: Optional[int] = None,
    local_path: str = "./"
):
    """URL 방식으로 데이터셋 병합 (훨씬 빠르고 메모리 효율적)"""
    print(f"🚀 URL 방식 멀티모달 데이터셋 병합 시작...")
    
    staging_dir = f"{local_path}/{output_name}_staging".replace("//", "/")
    os.makedirs(staging_dir, exist_ok=True)
    jsonl_path = os.path.join(staging_dir, "data.jsonl")
    
    total_samples = 0
    completion_messages = []

    # JSON 직렬화를 위한 datetime 핸들러
    def datetime_handler(x):
        if isinstance(x, datetime.datetime):
            return x.isoformat()
        raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

    # JSONL 파일에 직접 저장
    with open(jsonl_path, "w", encoding="utf-8") as f:
        dataset_progress = tqdm(dataset_configs, desc="데이터셋 처리", unit="dataset")

        for dataset_name, config_name in dataset_progress:
            dataset_progress.set_description(f"처리중: {dataset_name.split('/')[-1]}")
            try:
                for result in process_dataset_url(dataset_name, config_name, max_samples_per_dataset):
                    if isinstance(result, list):  # 배치 결과
                        for sample in result:
                            # original_data를 JSON 문자열로 변환
                            try:
                                sample["original_data"] = json.dumps(sample["original_data"], ensure_ascii=False, default=str)
                            except (TypeError, OverflowError):
                                sample["original_data"] = "{}"

                            f.write(json.dumps(
                                sample, 
                                ensure_ascii=False, 
                                default=datetime_handler
                            ) + "\n")
                            total_samples += 1
                        
                        dataset_progress.set_postfix({"총 샘플": total_samples})

                    elif isinstance(result, str):  # 완료 메시지
                        completion_messages.append(result)
            except Exception as e:
                tqdm.write(f"❌ {dataset_name} 처리 실패: {str(e)}")
                continue

    dataset_progress.close()

    # 처리 결과 출력
    tqdm.write("\n" + "="*20 + " 처리 결과 요약 " + "="*20)
    for msg in completion_messages:
        tqdm.write(msg)
    tqdm.write("="*55)

    if total_samples == 0:
        print("❌ 변환된 샘플이 없습니다.")
        return None
    
    tqdm.write(f"\n🎯 총 {total_samples}개 샘플 변환 완료")
    
    # Dataset 객체 생성
    tqdm.write("📦 Dataset 객체 생성 중...")

    # URL 방식의 스키마
    features = Features({
        'messages': Sequence(
            Features({
                'role': Value('string'),
                'content': Sequence(
                    Features({
                        'type': Value('string'),
                        'text': Value('string'),
                        'image_url': Features({
                            'url': Value('string')
                        })
                    })
                )
            })
        ),
        'source_dataset': Value('string'),
        'original_data': Value('string')
    })
    
    def data_generator():
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    # content 필드 정규화
                    if 'messages' in record:
                        for message in record['messages']:
                            if 'content' in message:
                                for content_item in message['content']:
                                    # text 필드가 없으면 빈 문자열 추가
                                    if 'text' not in content_item:
                                        content_item['text'] = ""
                                    # image_url이 없으면 빈 구조 추가
                                    if 'image_url' not in content_item:
                                        content_item['image_url'] = {'url': ''}
                    yield record
                except json.JSONDecodeError:
                    continue
    
    # Dataset 생성
    dataset = Dataset.from_generator(data_generator, features=features)
    
    # 로컬 저장
    final_save_path = f"{local_path}/{output_name}".replace("//", "/")
    dataset.save_to_disk(final_save_path)
    
    tqdm.write(f"✅ 최종 데이터셋 저장 완료: {final_save_path}")
    
    # 임시 파일 정리
    shutil.rmtree(staging_dir, ignore_errors=True)
    
    return final_save_path

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="URL 방식 멀티모달 통합 데이터셋 처리 스크립트")
    parser.add_argument("--output_name", type=str, default="unified-multimodal-sft-url", help="생성할 데이터셋의 로컬 폴더 이름")
    parser.add_argument("--max_samples", type=int, default=None, help="데이터셋별 최대 샘플 수")
    parser.add_argument("--local_path", type=str, default="./", help="데이터셋을 저장할 로컬 경로")

    args = parser.parse_args()

    print(f"🎯 URL 방식 처리 시작")
    print(f"🎯 타겟 로컬 경로: {os.path.join(args.local_path, args.output_name)}")
    
    final_path = merge_and_create_dataset_url(
        output_name=args.output_name,
        max_samples_per_dataset=args.max_samples,
        local_path=args.local_path
    )
    
    if final_path:
        print("\n🎉 URL 방식 병합 완료!")
        print(f"✅ 데이터셋이 '{final_path}'에 저장되었습니다.")
        print(f"📝 이미지는 URL로 저장되어 실제 학습 시 다운로드됩니다.")

if __name__ == "__main__":
    main() 