from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, Sequence, Image as ImageFeature, load_from_disk, DatasetDict
from huggingface_hub import HfApi
import json
from typing import List, Dict, Any, Optional, cast
from tqdm.auto import tqdm
import os
import requests
from PIL import Image
from io import BytesIO
from huggingface_hub.utils.tqdm import disable_progress_bars
import concurrent.futures
from functools import partial
import threading
import time
from urllib.parse import urlparse
import hashlib
import gc
import datetime
import argparse
import sys
import pandas as pd
import tempfile
import shutil

# PNG 텍스트 청크 크기 제한 증가 (Decompressed data too large 오류 해결)
# 일부 이미지에 매우 큰 메타데이터가 포함된 경우를 처리하기 위함
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024 # 100MB로 제한 증가

disable_progress_bars()  # 진행 표시줄 비활성화

# 이미지 캐시 및 세션 설정
image_cache = {}
cache_lock = threading.Lock()

# 세션 풀 생성 (재사용 가능한 연결)
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

# 데이터셋별 모드 선택 매핑 (reasoning 모드와 instruction 모드 구분)
DATASET_MODE_MAPPING = {
    # Reasoning 모드가 적합한 데이터셋들 (복잡한 추론과 논리적 사고가 필요한 데이터셋)
    "HuggingFaceTB/smoltalk": "reasoning",
    "PrincetonPLI/Instruct-SkillMix-SDD": "reasoning", 
    "nvidia/OpenCodeInstruct": "reasoning",
    "microsoft/orca-agentinstruct-1M-v1": "reasoning",
    "open-r1/Mixture-of-Thoughts": "reasoning",
    "NousResearch/Hermes-3-Dataset": "reasoning",
    
    # Instruction 모드가 적합한 데이터셋들 (명확한 지시사항 수행과 대화에 특화된 데이터셋)
    "R0k1e/UltraLink": "instruction",
    "allenai/WildChat-1M": "instruction",
    "MaziyarPanahi/Llama-Nemotron-Post-Training-Dataset-v1-ShareGPT": "instruction",
    "nvidia/Llama-Nemotron-Post-Training-Dataset": "instruction",
    "Salesforce/blip3-kale": "instruction",
    "liuhaotian/LLaVA-Instruct-150K": "instruction",
    "Lin-Chen/ShareGPT4V": "instruction",
    "nvidia/Llama-Nemotron-VLM-Dataset-v1": "instruction"
}

# 통합 시스템 프롬프트 - 간결하고 명확한 버전
UNIFIED_SYSTEM_PROMPT = """You are an AI assistant. You have two modes:

**REASONING MODE**: Use "Let me think through this step by step..." for complex problems, coding, math, or when user requests deep thinking.

**INSTRUCTION MODE**: Follow instructions directly for simple tasks, questions, or conversations.

Choose mode based on user request. For images, analyze carefully and choose appropriate mode."""

# 데이터셋별 모드 선택 매핑
DATASET_MODE_MAPPING = {
    # Reasoning 모드가 적합한 데이터셋들
    "HuggingFaceTB/smoltalk": "reasoning",
    "PrincetonPLI/Instruct-SkillMix-SDD": "reasoning", 
    "nvidia/OpenCodeInstruct": "reasoning",
    "microsoft/orca-agentinstruct-1M-v1": "reasoning",
    "open-r1/Mixture-of-Thoughts": "reasoning",
    "NousResearch/Hermes-3-Dataset": "reasoning",
    
    # Instruction 모드가 적합한 데이터셋들
    "R0k1e/UltraLink": "instruction",
    "allenai/WildChat-1M": "instruction",
    "MaziyarPanahi/Llama-Nemotron-Post-Training-Dataset-v1-ShareGPT": "instruction",
    "nvidia/Llama-Nemotron-Post-Training-Dataset": "instruction",
    "Salesforce/blip3-kale": "instruction",
    "liuhaotian/LLaVA-Instruct-150K": "instruction",
    "Lin-Chen/ShareGPT4V": "instruction",
    "nvidia/Llama-Nemotron-VLM-Dataset-v1": "instruction"
}

# 멀티모달 데이터셋 목록
dataset_configs = [
    ("HuggingFaceTB/smoltalk", "all"),
    ("R0k1e/UltraLink", None),
    ("PrincetonPLI/Instruct-SkillMix-SDD", None),
    ("allenai/WildChat-1M", None),
    ("nvidia/OpenCodeInstruct", None),
    ("microsoft/orca-agentinstruct-1M-v1", "default"),  # default config 사용
    ("MaziyarPanahi/Llama-Nemotron-Post-Training-Dataset-v1-ShareGPT", "default"),  # default config 사용
    ("nvidia/Llama-Nemotron-Post-Training-Dataset", "SFT"),  # SFT config 사용
    ("open-r1/Mixture-of-Thoughts", "all"),
    ("Salesforce/blip3-kale", "core"),
    ("liuhaotian/LLaVA-Instruct-150K", None),
    ("Lin-Chen/ShareGPT4V", "ShareGPT4V"),
    # 추가: Hermes-3, Nemotron VLM v1
    ("NousResearch/Hermes-3-Dataset", None),
    ("nvidia/Llama-Nemotron-VLM-Dataset-v1", None)
]

def get_system_prompt_for_dataset(dataset_name: str) -> str:
    """데이터셋에 적절한 통합 시스템 프롬프트를 반환합니다."""
    return UNIFIED_SYSTEM_PROMPT

def get_dataset_mode(dataset_name: str) -> str:
    """데이터셋의 기본 모드를 반환합니다."""
    return DATASET_MODE_MAPPING.get(dataset_name, "instruction")

def add_system_prompt_to_messages(messages: List[Dict[str, Any]], system_prompt: str, dataset_name: str) -> List[Dict[str, Any]]:
    """메시지 리스트의 맨 앞에 시스템 프롬프트와 모드 선택 지시를 추가합니다. 기존 시스템 프롬프트가 있다면 병합합니다."""
    if not messages:
        return messages
    
    # 데이터셋의 기본 모드 가져오기
    default_mode = get_dataset_mode(dataset_name)
    
    # 기존 시스템 프롬프트가 있는지 확인하고 추출
    existing_system_prompt = ""
    non_system_messages = []
    
    for message in messages:
        if message.get("role") == "system":
            # 기존 시스템 프롬프트 내용 추출
            content = message.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                existing_system_prompt = str(content[0].get("text", ""))
            elif isinstance(content, str):
                existing_system_prompt = content
        else:
            non_system_messages.append(message)
    
    # 모드 선택 지시를 포함한 시스템 프롬프트 생성
    mode_instruction = f"\n\n**Current Context:** This dataset typically requires {default_mode.upper()} MODE responses. However, adapt the mode based on the specific user request as outlined in the system prompt above."
    
    # 기존 시스템 프롬프트와 새로운 시스템 프롬프트 병합
    if existing_system_prompt:
        # 기존 시스템 프롬프트가 있으면 병합
        combined_system_prompt = f"{existing_system_prompt}\n\n{system_prompt}{mode_instruction}"
    else:
        # 기존 시스템 프롬프트가 없으면 새로운 것만 사용
        combined_system_prompt = system_prompt + mode_instruction
    
    # 병합된 시스템 프롬프트를 첫 번째 메시지로 추가
    system_message = {
        "role": "system",
        "content": [{"type": "text", "text": combined_system_prompt}]
    }
    
    return [system_message] + non_system_messages

def construct_image_url(
    image_path,
    dataset_name
):
    """데이터셋별로 이미지 경로를 실제 URL로 변환"""
    if dataset_name == "Lin-Chen/ShareGPT4V":
        # ShareGPT4V는 COCO 이미지를 사용
        if image_path.startswith('coco/'):
            # coco/train2017/000000000009.jpg -> COCO 이미지 URL
            filename = os.path.basename(image_path)
            return f"http://images.cocodataset.org/train2017/{filename}"
    elif dataset_name == "liuhaotian/LLaVA-Instruct-150K":
        # LLaVA도 COCO 이미지를 주로 사용
        if not image_path.startswith('http'):
            # 파일명만 있는 경우 COCO URL 구성 시도
            return f"http://images.cocodataset.org/train2017/{image_path}"
    
    return None

def get_image_cache_key(image_url):
    """이미지 URL에서 캐시 키 생성"""
    return hashlib.md5(image_url.encode()).hexdigest()

def download_image_with_retry(image_url, max_retries=3, timeout=10):
    """재시도 로직이 있는 이미지 다운로드"""
    for attempt in range(max_retries):
        try:
            response = session.get(image_url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(0.5 * (2 ** attempt))  # 지수 백오프
    return None

def load_image_from_url_or_path(
    image_source,
    dataset_name=None
):
    """
    URL이나 경로에서 실제 이미지를 로드합니다. (캐시 및 최적화 포함)
    """
    try:
        # 이미 PIL Image 객체인 경우
        if hasattr(image_source, 'size') and hasattr(image_source, 'convert'):
            return image_source.convert('RGB')
        
        # 문자열인 경우 (URL 또는 파일명)
        if isinstance(image_source, str):
            # HTTP/HTTPS URL인 경우
            if image_source.startswith('http://') or image_source.startswith('https://'):
                # 캐시 확인
                cache_key = get_image_cache_key(image_source)
                with cache_lock:
                    if cache_key in image_cache:
                        return image_cache[cache_key]
                
                # 이미지 다운로드
                image_data = download_image_with_retry(image_source)
                if image_data:
                    image = Image.open(BytesIO(image_data))
                    image = image.convert('RGB')
                    
                    # 메모리 사용량 제한을 위한 이미지 크기 조정
                    max_size = (1024, 1024)
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    # 캐시에 저장 (최근 100개만 유지)
                    with cache_lock:
                        if len(image_cache) >= 100:
                            # 가장 오래된 항목 제거
                            oldest_key = next(iter(image_cache))
                            del image_cache[oldest_key]
                        image_cache[cache_key] = image
                    
                    return image
                return None
            
            # 로컬 파일 경로인 경우
            elif os.path.exists(image_source):
                image = Image.open(image_source)
                image = image.convert('RGB')
                # 크기 조정
                max_size = (1024, 1024)
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                return image
            
            # 파일명만 있는 경우 - URL 구성 시도
            else:
                if dataset_name:
                    constructed_url = construct_image_url(image_source, dataset_name)
                    if constructed_url:
                        return load_image_from_url_or_path(constructed_url, dataset_name)
                return None
        
        # bytes 데이터인 경우
        elif isinstance(image_source, bytes):
            image = Image.open(BytesIO(image_source))
            image = image.convert('RGB')
            # 크기 조정
            max_size = (1024, 1024)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image
            
        else:
            return None
        
    except Exception as e:
        return None

def process_image_batch(
    image_sources_with_info,
    max_workers=8
):
    """이미지 배치를 병렬로 처리"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 이미지 로드 작업을 동시에 시작
        future_to_info = {
            executor.submit(load_image_from_url_or_path, img_source, dataset_name): (img_source, dataset_name, idx)
            for idx, (img_source, dataset_name) in enumerate(image_sources_with_info)
        }
        
        # 결과 수집
        for future in concurrent.futures.as_completed(future_to_info):
            img_source, dataset_name, idx = future_to_info[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                results.append((idx, None))
    
    # 원래 순서대로 정렬
    results.sort(key=lambda x: x[0])
    return [result for _, result in results]

def convert_to_target_format(
    sample: Dict[str, Any], 
    dataset_name: str
) -> Optional[Dict[str, Any]]:
    """
    각 데이터셋의 샘플을 목표 형식으로 변환합니다.
    - messages: List[{ role: string, content: List[{type: "text"|"image", text: string}] }]
    - images: [PIL.Image] (멀티모달에서만)
    """

    result: Dict[str, Any] = {
        "messages": [],
        "images": [],
        "source_dataset": dataset_name,
        "original_data": sample.copy()
    }
    
    # 데이터셋에 적절한 시스템 프롬프트 가져오기
    system_prompt = get_system_prompt_for_dataset(dataset_name)

    try:
        # 텍스트 전용 데이터셋들 처리
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

        elif dataset_name == "PrincetonPLI/Instruct-SkillMix-SDD":
            if "instruction" in sample and "output" in sample:
                user_content_str = str(sample["instruction"])
                if "input" in sample and sample["input"] and str(sample["input"]).strip():
                    user_content_str += f"\n\nInput: {sample['input']}"

                result["messages"] = [
                    {"role": "user", "content": [{"type": "text", "text": user_content_str}]},
                    {"role": "assistant", "content": [{"type": "text", "text": str(sample["output"]) }]}
                ]

        elif dataset_name == "allenai/WildChat-1M":
            if "conversation" in sample and isinstance(sample["conversation"], list):
                for conv in sample["conversation"]:
                    if isinstance(conv, dict) and "role" in conv and "content" in conv:
                        result["messages"].append({
                            "role": conv["role"],
                            "content": [{"type": "text", "text": str(conv["content"]) }]
                        })

        elif dataset_name == "nvidia/OpenCodeInstruct":
            if "input" in sample and "output" in sample:
                result["messages"] = [
                    {"role": "user", "content": [{"type": "text", "text": str(sample["input"]) }]},
                    {"role": "assistant", "content": [{"type": "text", "text": str(sample["output"]) }]}
                ]

        elif dataset_name == "microsoft/orca-agentinstruct-1M-v1":
            if "messages" in sample and isinstance(sample["messages"], list):
                for msg in sample["messages"]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        result["messages"].append({
                            "role": msg["role"],
                            "content": [{"type": "text", "text": str(msg["content"]) }]
                        })

        elif "Nemotron" in dataset_name:
            if "conversations" in sample and isinstance(sample["conversations"], list):
                for conv in sample["conversations"]:
                    if isinstance(conv, dict) and "from" in conv and "value" in conv:
                        role = "user" if conv["from"] in ["human", "user"] else "assistant"
                        result["messages"].append({
                            "role": role,
                            "content": [{"type": "text", "text": str(conv["value"]) }]
                        })

        elif dataset_name == "NousResearch/Hermes-3-Dataset":
            # Hermes-3는 conversations 배열을 그대로 사용
            convs = sample.get("conversations") or sample.get("messages")
            if isinstance(convs, list):
                for conv in convs:
                    if not isinstance(conv, dict):
                        continue
                    frm = conv.get("from") or conv.get("role")
                    val = str(conv.get("value") or conv.get("content") or "")
                    if not frm:
                        continue
                    role = "user" if frm in ["human", "user"] else ("assistant" if frm in ["gpt", "assistant"] else frm)
                    result["messages"].append({
                        "role": role,
                        "content": [{"type": "text", "text": val }]
                    })

        elif dataset_name == "nvidia/Llama-Nemotron-VLM-Dataset-v1":
            # VLM v1: 다양한 서브스플릿, captioning 류는 이미지 + 캡션 구조
            # 예시 필드: image_url/image_path, caption/response 등을 가정하고 매핑
            image_obj = None
            for k in ["image", "image_path", "image_url", "url"]:
                if k in sample and sample[k]:
                    image_obj = load_image_from_url_or_path(sample[k])
                    if image_obj is not None:
                        result["images"].append(image_obj)
                        break
            caption = str(sample.get("caption") or sample.get("response") or sample.get("value") or "").strip()
            if caption:
                user_content: List[Dict[str, Any]] = [{"type": "text", "text": "Describe this image.", "image": ""}]
                if result["images"]:
                    user_content.append({"type": "image", "text": "", "image": ""})
                result["messages"] = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": caption, "image": ""}]}
                ]

        elif dataset_name == "open-r1/Mixture-of-Thoughts":
            if "messages" in sample and isinstance(sample["messages"], list):
                for msg in sample["messages"]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        result["messages"].append({
                            "role": msg["role"],
                            "content": [{"type": "text", "text": str(msg["content"]) }]
                        })

        # 멀티모달 데이터셋들 처리
        elif dataset_name in ["Lin-Chen/ShareGPT4V", "liuhaotian/LLaVA-Instruct-150K"]:
            # 이미지 추출 및 로드
            image_obj = None
            if "image" in sample and sample["image"] is not None:
                image_obj = load_image_from_url_or_path(sample["image"], dataset_name)
            elif "images" in sample and sample["images"] is not None:
                if isinstance(sample["images"], list) and len(sample["images"]) > 0:
                    image_obj = load_image_from_url_or_path(sample["images"][0], dataset_name)
                else:
                    image_obj = load_image_from_url_or_path(sample["images"], dataset_name)

            if image_obj is not None:
                result["images"].append(image_obj)

            # conversations 처리 (content: List[{type,text}] + 이미지 토큰은 별도 item로 표현)
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
                    content_list: List[Dict[str, Any]] = []
                    text_content = str(conv.get("value", "")).strip()
                    if text_content:
                        # 원 데이터에 <image> 토큰이 포함되어 있으면, 분리하여 image item으로 보존
                        if "<image>" in text_content:
                            # 각 토큰 앞뒤 텍스트도 보존
                            segments = [seg for seg in text_content.split("<image>")]
                            for sidx, seg in enumerate(segments):
                                seg = seg.strip()
                                if seg:
                                    content_list.append({"type": "text", "text": seg})
                                if sidx != len(segments) - 1:
                                    content_list.append({"type": "image", "text": None})
                        else:
                            content_list.append({"type": "text", "text": text_content})
                    # 첫 turn에 이미지가 존재하지만 텍스트에 <image> 토큰이 전혀 없는 경우, image item을 추가
                    if role == "user" and i == 0 and result["images"] and not any(it.get('type') == 'image' for it in content_list):
                        content_list.append({"type": "image", "text": None})

                    if content_list:
                        result["messages"].append({
                            "role": role,
                            "content": content_list
                        })

        elif dataset_name == "Salesforce/blip3-kale":
            # 이미지 로드
            image_obj = None
            if "url" in sample:
                image_obj = load_image_from_url_or_path(sample["url"], dataset_name)
            elif "image" in sample:
                image_obj = load_image_from_url_or_path(sample["image"], dataset_name)

            if image_obj is not None:
                result["images"].append(image_obj)

            # caption을 대화 형식으로 변환 (content 리스트 구조)
            caption = str(sample.get("caption", "")).strip()
            if not caption:
                caption = str(sample.get("cogvlm_caption", "")).strip()

            if caption:
                user_content: List[Dict[str, Any]] = [{"type": "text", "text": "Describe this image.", "image": ""}]
                if result["images"]:
                    # content.image에 상대 경로를 넣을 수 있으나, 이 시점에서는 이미지 파일명이 아직 없으므로 빈 문자열 유지
                    user_content.append({"type": "image", "text": "", "image": ""})

                result["messages"] = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": caption, "image": ""}]}
                ]

        # 빈 messages인 경우 None 반환
        if not result["messages"]:
            return None
        
        # 시스템 프롬프트를 메시지 맨 앞에 추가
        result["messages"] = add_system_prompt_to_messages(result["messages"], system_prompt, dataset_name)
            
        return result

    except Exception as e:
        # 오류가 발생하면 None 반환하여 건너뛰기
        print(f"샘플 변환 중 오류 (건너뛰기): {dataset_name} - {str(e)}")
        return None

def process_samples_batch(
    samples_batch, 
    dataset_name, 
    max_workers=8
):
    """샘플 배치를 병렬로 처리"""
    # 이미지가 있는 샘플들을 먼저 식별
    image_samples = []
    non_image_samples = []
    
    for i, sample in enumerate(samples_batch):
        has_image = False
        
        # 이미지가 있는 데이터셋인지 확인
        if dataset_name in ["Lin-Chen/ShareGPT4V", "liuhaotian/LLaVA-Instruct-150K", "Salesforce/blip3-kale"]:
            if ("image" in sample and sample["image"]) or ("images" in sample and sample["images"]) or ("url" in sample and sample["url"]):
                has_image = True
        
        if has_image:
            image_samples.append((i, sample))
        else:
            non_image_samples.append((i, sample))
    
    # 이미지가 없는 샘플들을 먼저 빠르게 처리
    results: List[Optional[Dict[str, Any]]] = [None] * len(samples_batch)
    
    for i, sample in non_image_samples:
        converted = convert_to_target_format(sample, dataset_name)
        results[i] = converted
    
    # 이미지가 있는 샘플들을 병렬로 처리
    if image_samples:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(convert_to_target_format, sample, dataset_name): i
                for i, sample in image_samples
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    results[idx] = None
    
    return [r for r in results if r is not None]

def process_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 8
):
    """데이터셋을 처리하여 목표 형식으로 변환합니다. (병렬 처리 추가)"""
    try:
        # 특정 데이터셋들의 split 설정
        if dataset_name == "microsoft/orca-agentinstruct-1M-v1":
            split_candidates = ["creative_content", "train"]
        elif dataset_name == "MaziyarPanahi/Llama-Nemotron-Post-Training-Dataset-v1-ShareGPT":
            split_candidates = ["chat", "train"]
        elif dataset_name == "nvidia/Llama-Nemotron-Post-Training-Dataset":
            split_candidates = ["chat", "train"]
        elif dataset_name == "nvidia/Llama-Nemotron-VLM-Dataset-v1":
            # 다양한 서브스플릿 존재: captioning_x, ocr_x, vqa_x 등
            # 이 데이터셋은 split 정보 불일치 문제가 있어서 특별 처리
            split_candidates = ["train"]
        else:
            split_candidates = ["train"]
        
        # 데이터셋 로드 (여러 후보 split 순회)
        full_dataset = None
        last_err = None
        
        # 특정 데이터셋들은 streaming=False로 처리 (split 정보 불일치 문제 해결)
        use_streaming = True
        special_handling = False
        
        if dataset_name == "nvidia/Llama-Nemotron-VLM-Dataset-v1":
            use_streaming = False
            special_handling = True
            print(f"ℹ️ {dataset_name}: split 정보 불일치 문제로 인해 특별 처리 모드 활성화")
        
        for split in split_candidates:
            try:
                if config_name:
                    full_dataset = load_dataset(dataset_name, config_name, split=split, streaming=use_streaming)
                else:
                    full_dataset = load_dataset(dataset_name, split=split, streaming=use_streaming)
                break
            except Exception as e:
                last_err = e
                print(f"⚠️ split '{split}' 로드 실패: {e}")
                continue
                
        if full_dataset is None:
            # Nemotron VLM 처럼 다양한 split이 있을 경우 전체 split을 불러와 순회
            try:
                ds_all = load_dataset(dataset_name, config_name) if config_name else load_dataset(dataset_name)
                # 가능한 첫 split 선택
                for split_name in ds_all.keys():
                    try:
                        full_dataset = load_dataset(dataset_name, config_name, split=split_name, streaming=use_streaming) if config_name else load_dataset(dataset_name, split=split_name, streaming=use_streaming)
                        print(f"✅ split '{split_name}'로 진행")
                        break
                    except Exception as e:
                        print(f"⚠️ split '{split_name}' 로드 실패: {e}")
                        continue
            except Exception as e2:
                print(f"❌ 데이터셋 split 탐색 실패: {e2}")
                # split 정보 불일치 문제가 있는 경우 전체 데이터셋을 한 번에 로드 시도
                if "split" in str(e2).lower() or "expected" in str(e2).lower():
                    print(f"🔄 split 정보 불일치 문제 감지. 전체 데이터셋을 한 번에 로드 시도...")
                    
                            # 특별 처리 모드인 경우 huggingface_hub에서 split 정보를 먼저 확인
        if special_handling:
            print(f"🔄 특별 처리 모드: huggingface_hub에서 split 정보 확인 중...")
            try:
                api = HfApi()
                # 데이터셋의 split 정보를 먼저 확인
                dataset_info = api.dataset_info(dataset_name)
                available_splits = list(dataset_info.splits.keys())
                print(f"✅ 사용 가능한 split: {available_splits}")
                
                # 사용 가능한 split 중에서 첫 번째를 선택하여 로드
                if available_splits:
                    selected_split = available_splits[0]
                    print(f"🔄 선택된 split: {selected_split}")
                    
                    try:
                        if config_name:
                            full_dataset = load_dataset(dataset_name, config_name, split=selected_split, streaming=use_streaming)
                        else:
                            full_dataset = load_dataset(dataset_name, split=selected_split, streaming=use_streaming)
                        print(f"✅ {selected_split} split 로드 성공!")
                    except Exception as split_e:
                        print(f"⚠️ {selected_split} split 로드 실패: {split_e}")
                        # 다른 split들도 시도
                        for split_name in available_splits[1:]:
                            try:
                                print(f"🔄 {split_name} split 시도 중...")
                                if config_name:
                                    full_dataset = load_dataset(dataset_name, config_name, split=split_name, streaming=use_streaming)
                                else:
                                    full_dataset = load_dataset(dataset_name, split=split_name, streaming=use_streaming)
                                print(f"✅ {split_name} split 로드 성공!")
                                break
                            except Exception as other_split_e:
                                print(f"⚠️ {split_name} split 실패: {other_split_e}")
                                continue
                        else:
                            print(f"❌ 모든 split 로드 실패")
                            return
                else:
                    print(f"❌ 사용 가능한 split이 없습니다")
                    return
                    
            except Exception as e3:
                print(f"⚠️ huggingface_hub에서 split 정보 가져오기 실패: {e3}")
                # fallback: 기존 방식으로 시도
                try:
                    if config_name:
                        full_dataset = load_dataset(dataset_name, config_name, trust_remote_code=True, streaming=use_streaming)
                    else:
                        full_dataset = load_dataset(dataset_name, trust_remote_code=True, streaming=use_streaming)
                    print(f"✅ trust_remote_code=True로 데이터셋 로드 성공")
                except Exception as e4:
                    print(f"❌ fallback 방식도 실패: {e4}")
                    return
            else:
                # 일반적인 fallback 시도
                try:
                    if config_name:
                        full_dataset = load_dataset(dataset_name, config_name, streaming=use_streaming)
                    else:
                        full_dataset = load_dataset(dataset_name, streaming=use_streaming)
                    print(f"✅ 전체 데이터셋 로드 성공")
                except Exception as e3:
                    print(f"❌ 전체 데이터셋 로드도 실패: {e3}")
                    return
                    
        if full_dataset is None:
            print(f"❌ 데이터셋 로드 실패(모든 split): {last_err}")
            return

        success_count = 0
        total_count = 0
        batch_size = max(8, num_workers)  # 배치 크기를 워커 수에 맞춤
        current_batch = []
        
        # 진행 상황 표시를 위한 tqdm 설정
        desc = f"{dataset_name.split('/')[-1]}"
        if config_name:
            desc += f"({config_name})"
        
        # leave=False를 추가하여 완료 후 진행 막대가 사라지도록 함
        progress_bar = tqdm(desc=desc, unit="samples", leave=False)
        
        # 데이터 처리 (streaming 여부에 따라 다르게 처리)
        if use_streaming:
            # 스트리밍 데이터 처리
            for sample in full_dataset:
                if max_samples and total_count >= max_samples:
                    break
                
                current_batch.append(sample)
                total_count += 1
                
                # 배치가 찼거나 마지막 샘플인 경우 처리
                if len(current_batch) >= batch_size or (max_samples and total_count >= max_samples):
                    # 배치 처리
                    batch_results = process_samples_batch(current_batch, dataset_name, num_workers)
                    
                    if batch_results:
                        success_count += len(batch_results)
                        yield batch_results
                    
                    progress_bar.update(len(current_batch))
                    progress_bar.set_postfix({
                        "processed": f"{success_count}/{total_count}",
                        "success_rate": f"{success_count/total_count*100:.1f}%"
                    })
                    
                    current_batch = []
            
            # 남은 배치 처리
            if current_batch:
                batch_results = process_samples_batch(current_batch, dataset_name, num_workers)
                if batch_results:
                    success_count += len(batch_results)
                    yield batch_results
                progress_bar.update(len(current_batch))
        else:
            # 일반 Dataset 객체 처리 (전체 데이터를 한 번에 처리)
            total_samples_in_dataset = len(full_dataset)
            if max_samples:
                total_samples_in_dataset = min(total_samples_in_dataset, max_samples)
            
            print(f"📊 전체 {total_samples_in_dataset}개 샘플을 배치로 처리 중...")
            
            # 전체 데이터를 배치 단위로 처리
            for i in range(0, total_samples_in_dataset, batch_size):
                end_idx = min(i + batch_size, total_samples_in_dataset)
                batch_samples = list(full_dataset.select(range(i, end_idx)))
                
                # 배치 처리
                batch_results = process_samples_batch(batch_samples, dataset_name, num_workers)
                
                if batch_results:
                    success_count += len(batch_results)
                    yield batch_results
                
                total_count = end_idx
                progress_bar.update(len(batch_samples))
                progress_bar.set_postfix({
                    "processed": f"{success_count}/{total_count}",
                    "success_rate": f"{success_count/total_count*100:.1f}%"
                })
        
        progress_bar.close()
        
        # 완료 메시지를 반환값으로 변경하여 나중에 한 번에 출력하도록 함
        yield f"✅ {dataset_name}: {success_count}/{total_count} 샘플 변환 완료 (성공률: {success_count/total_count*100:.1f}%)" if total_count > 0 else f"ℹ️ {dataset_name}: 처리할 샘플 없음"

    except Exception as e:
        yield f"❌ {dataset_name} 처리 중 오류: {str(e)}"

def generate_cleaned_records(file_path: str):
    """
    Reads a JSONL file line-by-line, cleans the data, and yields records.
    This generator approach is highly memory-efficient.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # tqdm will show processing speed (it/s) without a total count,
        # which avoids reading the file twice.
        for line in tqdm(f, desc="Streaming and cleaning records"):
            try:
                record = json.loads(line)
                
                # Clean the 'messages' field in-place for efficiency
                # messages -> conversations로 이행. 기존 messages는 무시하고 conversations만 유지
                if 'messages' in record and isinstance(record['messages'], list):
                    conversations = []
                    for m in record['messages']:
                        role = m.get('role', '')
                        # 시스템 메시지는 conversations에서 제외 (시스템 프롬프트는 별도로 처리)
                        if role == 'system':
                            continue
                        frm = 'human' if role == 'user' else ('gpt' if role == 'assistant' else role)
                        parts: List[str] = []
                        for it in m.get('content', []) or []:
                            if not isinstance(it, dict):
                                continue
                            if it.get('type') == 'text':
                                txt = str(it.get('text') or '')
                                if txt:
                                    parts.append(txt)
                            elif it.get('type') == 'image':
                                img_ref = it.get('image') or ''
                                parts.append(f"<image:{img_ref}>" if img_ref else "<image>")
                        conversations.append({'from': frm, 'value': '\n'.join(parts)})
                    record['conversations'] = conversations
                    del record['messages']

                yield record

            except (json.JSONDecodeError, TypeError):
                print(f"Skipping malformed line: {line.strip()}")

def merge_and_create_dataset(
    output_name: str = "unified-multimodal-sft", 
    max_samples_per_dataset: Optional[int] = None, 
    num_workers: int = 8,  # 더 적은 워커 수로 메모리 절약
    local_path: str = "./",
):
    """
    모든 멀티모달 데이터셋을 병합하고 목표 형식으로 생성합니다.
    메모리 문제를 해결하기 위해 중간 결과를 디스크에 저장하는 방식을 사용합니다.
    """
    print(f"🚀 멀티모달 데이터셋 병합 시작... (워커 수: {num_workers})")
    
    # 1. 임시 저장 공간 설정
    staging_dir = f"{local_path}/{output_name}_staging".replace("//", "/")
    images_dir = os.path.join(staging_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    jsonl_path = os.path.join(staging_dir, "data.jsonl")
    
    tqdm.write(f"📂 임시 저장 경로: {staging_dir}")

    # JSON 직렬화를 위한 datetime 핸들러
    def datetime_handler(x):
        if isinstance(x, datetime.datetime):
            return x.isoformat()
        raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

    total_samples = 0
    image_counter = 0
    completion_messages = []

    # 2. 데이터를 JSONL과 이미지 파일로 디스크에 저장 (이미지 경로 보존)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        dataset_progress = tqdm(dataset_configs, desc="데이터셋 처리", unit="dataset")

        for dataset_name, config_name in dataset_progress:
            dataset_progress.set_description(f"처리중: {dataset_name.split('/')[-1]}")
            try:
                for result in process_dataset(dataset_name, config_name, max_samples_per_dataset, num_workers):
                    if isinstance(result, list): # 배치 결과
                        for sample in result:
                            # 이미지 처리: PIL 객체를 파일로 저장하되 상대경로만 저장
                            image_paths = []
                            if sample.get("images"):
                                for img in sample["images"]:
                                    if hasattr(img, 'save'):
                                        image_filename = f"{image_counter:08d}.png"
                                        # 이미지 파일 저장
                                        img_save_path = os.path.join(images_dir, image_filename)
                                        img.save(img_save_path, "PNG")
                                        # 상대 경로만 저장 (staging_dir 기준)
                                        image_paths.append(os.path.join("images", image_filename))
                                        image_counter += 1
                            
                            sample["images"] = image_paths

                            # messages.content 내 image 아이템에 경로 매핑 (순서대로 할당)
                            try:
                                if sample.get("messages"):
                                    img_idx = 0
                                    for m in sample["messages"]:
                                        content_list = m.get("content")
                                        if isinstance(content_list, list):
                                            for item in content_list:
                                                if isinstance(item, dict) and item.get("type") == "image":
                                                    if img_idx < len(image_paths):
                                                        item["image"] = image_paths[img_idx]
                                                        img_idx += 1
                                                    else:
                                                        # 남는 경로가 없으면 None 유지
                                                        item.setdefault("image", None)
                            except Exception:
                                pass

                            # Hermes 스타일 'conversations' 필드 동시 생성 (viewer 호환)
                            try:
                                conversations = []
                                for m in sample.get("messages", []):
                                    role = m.get("role", "")
                                    frm = "human" if role == "user" else ("gpt" if role == "assistant" else role)
                                    parts: List[str] = []
                                    for it in m.get("content", []) or []:
                                        if not isinstance(it, dict):
                                            continue
                                        if it.get("type") == "text":
                                            txt = str(it.get("text") or "")
                                            if txt:
                                                parts.append(txt)
                                        elif it.get("type") == "image":
                                            img_ref = it.get("image") or ""
                                            if img_ref:
                                                parts.append(f"<image:{img_ref}>")
                                            else:
                                                parts.append("<image>")
                                    conversations.append({"from": frm, "value": "\n".join(parts)})
                                sample["conversations"] = conversations
                            except Exception:
                                pass
                            
                            # original_data를 안전하게 JSON 문자열로 변환
                            try:
                                sample["original_data"] = json.dumps(sample["original_data"], ensure_ascii=False, default=str)
                            except (TypeError, OverflowError):
                                sample["original_data"] = "{}" # 변환 실패 시 빈 객체로

                            f.write(json.dumps(
                                sample, 
                                ensure_ascii=False, 
                                default=datetime_handler
                            ) + "\n")
                            total_samples += 1
                        
                        dataset_progress.set_postfix({"총 샘플": total_samples})

                    elif isinstance(result, str): # 완료 메시지
                        completion_messages.append(result)
            except Exception as e:
                tqdm.write(f"❌ {dataset_name} 처리 실패: {str(e)}")
                continue

            # 데이터셋 처리 후 메모리 최적화
            with cache_lock:
                image_cache.clear()
            gc.collect()
            tqdm.write(f"🧠 {dataset_name.split('/')[-1]} 처리 후 메모리 최적화 완료.")

    dataset_progress.close()

    tqdm.write("\n" + "="*20 + " 처리 결과 요약 " + "="*20)
    for msg in completion_messages:
        tqdm.write(msg)
    tqdm.write("="*55)

    if total_samples == 0:
        print("❌ 변환된 샘플이 없습니다.")
        return None
    
    tqdm.write(f"\n🎯 총 {total_samples}개 샘플 변환 완료 및 임시 저장 완료")
    
    # 3. 디스크에 저장된 데이터를 메모리 효율적으로 로드
    tqdm.write("📦 임시 파일로부터 Dataset 객체 생성 중...")

    # 데이터셋의 최종 스키마(구조) 정의
    features = Features({
        'conversations': Sequence(
            Features({
                'from': Value('string'),
                'value': Value('string')
            })
        ),
        'images': Sequence(Value('string')), # 먼저 문자열 경로로 로드
        'source_dataset': Value('string'),
        'original_data': Value('string')
    })
    
    # 로컬 저장
    tqdm.write("💾 로컬 저장 중 (최종 Arrow 포맷)...")
    final_save_path = f"{local_path}/{output_name}".replace("//", "/")

    # 1. 제너레이터를 사용하여 스트리밍 방식으로 데이터 로드 및 정제
    tqdm.write("   - 스트리밍 방식으로 데이터 로드 및 정제 중...")
    iterable_dataset = Dataset.from_generator(
        generate_cleaned_records,
        features=features,
        gen_kwargs={"file_path": jsonl_path},
    )
    # IterableDataset을 일반 Dataset으로 변환하여 .map()과 .filter() 사용
    dataset = Dataset.from_list(list(tqdm(iterable_dataset, desc="Converting to standard dataset")))

    # 2. 이미지 경로를 실제 이미지 객체로 변환 (상대 경로 기준 설정)
    staging_dir_abs = os.path.abspath(staging_dir)
    def resolve_and_load_images(example):
        if example['images']:
            # 절대 경로로 변환하고 이미지 로드
            loaded_images = []
            for img_path in example['images']:
                full_path = os.path.join(staging_dir_abs, img_path)
                if os.path.exists(full_path):
                    try:
                        img = Image.open(full_path)
                        loaded_images.append(img.convert('RGB'))
                    except Exception as e:
                        print(f"이미지 로드 실패: {full_path} - {e}")
                        loaded_images.append(None)
                else:
                    loaded_images.append(None)
            example['images'] = loaded_images
        # 메시지 내 image 경로는 그대로 두고(문자열), 이후 캐스팅으로 처리하거나 업로드에서 ImageFeature로 처리
        return example

    # 이미지 경로를 변환하고, None인 이미지를 필터링 (다중 처리로 가속)
    tqdm.write(f"   - 이미지 로드 중 (워커: {num_workers})...")
    dataset = dataset.map(resolve_and_load_images, num_proc=num_workers)
    dataset = dataset.filter(lambda example: not (example.get('images') and None in example['images']), num_proc=num_workers)

    # 최종적으로 Image Feature로 캐스팅
    tqdm.write("   - 이미지 데이터 타입 변환 중...")
    unified_dataset = dataset.cast_column("images", Sequence(ImageFeature()))

    # 캐시 정리
    with cache_lock:
        image_cache.clear()
    
    unified_dataset.save_to_disk(final_save_path)
    tqdm.write(f"   - 최종 데이터셋 경로: {final_save_path}")
    
    return final_save_path

def upload_dataset_to_hub(
    dataset_path: str,
    repo_id: str,
    private: bool = False,
    num_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    single_repo: bool = False,
    start_chunk_num: int = 0
):
    """
    로컬에 저장된 데이터셋을 허깅페이스 허브에 업로드합니다.
    메모리 효율적인 스트리밍 방식 사용
    
    Args:
        single_repo: True면 하나의 리포지토리에 순차적으로 추가, False면 청크별 리포지토리 생성
    """
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 4)  # 워커 수 제한
    
    if chunk_size is None:
        chunk_size = min(200, num_workers * 25)  # 더 작은 청크 크기
        
    jsonl_path = os.path.join(dataset_path, "data.jsonl")
    # Fallback: 최종 폴더에 JSONL이 없다면 staging 경로를 자동 검색
    if not os.path.exists(jsonl_path):
        candidate = f"{dataset_path}_staging"
        alt_jsonl = os.path.join(candidate, "data.jsonl")
        if os.path.exists(alt_jsonl):
            print(f"ℹ️ data.jsonl이 최종 경로에 없어 staging 경로로 대체: {alt_jsonl}")
            jsonl_path = alt_jsonl
        else:
            print(f"❌ JSONL 파일 없음: {jsonl_path}")
            return False

    print(f"🚀 데이터셋 업로드: {repo_id}")
    print(f"📊 청크 크기: {chunk_size}, 워커 수: {num_workers}")
    
    try:
        # 이미지를 포함한 데이터셋 생성
        def data_generator():
            staging_dir_abs = os.path.dirname(jsonl_path)
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        record = json.loads(line.strip())
                        
                        # conversations 필드가 이미 올바른 형태로 되어 있으므로 그대로 사용
                        # messages 필드가 있다면 conversations로 변환
                        if 'messages' in record and isinstance(record['messages'], list) and 'conversations' not in record:
                            conversations = []
                            for message in record['messages']:
                                role = message.get('role', '')
                                if role == 'system':
                                    continue  # 시스템 메시지는 제외
                                frm = 'human' if role == 'user' else ('gpt' if role == 'assistant' else role)
                                content_value = message.get('content', [])
                                if isinstance(content_value, list):
                                    parts = []
                                    for item in content_value:
                                        if isinstance(item, dict):
                                            if item.get('type') == 'text':
                                                txt = str(item.get('text') or '')
                                                if txt:
                                                    parts.append(txt)
                                            elif item.get('type') == 'image':
                                                img_ref = item.get('image') or ''
                                                parts.append(f"<image:{img_ref}>" if img_ref else "<image>")
                                    conversations.append({'from': frm, 'value': '\n'.join(parts)})
                                elif isinstance(content_value, str):
                                    conversations.append({'from': frm, 'value': content_value})
                            record['conversations'] = conversations

                        # 이미지 경로를 실제 이미지로 변환
                        if 'images' in record and isinstance(record['images'], list):
                            loaded_images = []
                            for img_path in record['images']:
                                if img_path and isinstance(img_path, str):
                                    full_path = os.path.join(staging_dir_abs, img_path)
                                    if os.path.exists(full_path):
                                        try:
                                            img = Image.open(full_path)
                                            loaded_images.append(img.convert('RGB'))
                                        except Exception as e:
                                            print(f"이미지 로드 실패: {full_path} - {e}")
                                            continue
                            record['images'] = loaded_images
                        
                        # 시스템 프롬프트 및 모드 정보 추출 및 추가
                        system_prompt = ""
                        dataset_mode = "instruction"  # 기본값
                        
                        # messages 필드에서 시스템 프롬프트 추출
                        if 'messages' in record and isinstance(record['messages'], list):
                            for message in record['messages']:
                                if message.get('role') == 'system':
                                    content = message.get('content', [])
                                    if isinstance(content, list) and len(content) > 0:
                                        system_prompt = str(content[0].get('text', ''))
                                        break
                        
                        # 소스 데이터셋에서 모드 정보 추출
                        if 'source_dataset' in record:
                            dataset_mode = get_dataset_mode(record['source_dataset'])
                        
                        record['system_prompt'] = system_prompt
                        record['dataset_mode'] = dataset_mode
                        
                        yield record
                        
                        # 메모리 정리를 위해 주기적으로 가비지 컬렉션
                        if line_num % 500 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        print(f"   라인 {line_num} 건너뛰기: {e}")
                        continue
        
        # 이미지를 포함한 데이터셋 스키마 (시스템 프롬프트 및 모드 정보 포함)
        from datasets import Features, Value, Sequence, Image as ImageFeature
        features = Features({
            'conversations': Sequence(
                Features({
                    'from': Value('string'),
                    'value': Value('string')
                })
            ),
            'images': Sequence(ImageFeature()),
            'source_dataset': Value('string'),
            'original_data': Value('string'),
            'system_prompt': Value('string'),  # 시스템 프롬프트
            'dataset_mode': Value('string')   # 데이터셋 기본 모드
        })
        
        print("📦 스트리밍 방식으로 데이터셋 생성 중...")
        # 스트리밍 방식으로 데이터셋 생성 (메모리 효율적)
        iterable_dataset = Dataset.from_generator(
            data_generator,
            features=features
        )
        
        # 메모리 효율적인 청크 단위 처리
        print("📦 청크 단위로 데이터셋 변환 및 업로드 중...")
        
        CHUNK_SIZE = 10000  # 1만 개씩 처리
        chunk_datasets = []
        current_chunk = []
        chunk_num = 0
        # 청크 저장 디렉토리 보장 (이미지 포함 저장 시 경로 필요)
        temp_chunk_dir = "/mnt/disks/data/tmp"
        os.makedirs(temp_chunk_dir, exist_ok=True)
        
        for record in tqdm(iterable_dataset, desc="Processing records"):
            current_chunk.append(record)
            
            if len(current_chunk) >= CHUNK_SIZE:
                if chunk_num < start_chunk_num:
                    print(f"   건너뛰기: 청크 {chunk_num} (시작 번호: {start_chunk_num})")
                    current_chunk = []
                    chunk_num += 1
                    continue

                # 청크를 Dataset으로 변환하고 임시 저장
                chunk_dataset = Dataset.from_list(current_chunk, features=features)
                temp_chunk_path = f"{temp_chunk_dir}/chunk_{chunk_num}"
                chunk_dataset.save_to_disk(temp_chunk_path)
                chunk_datasets.append(temp_chunk_path)
                
                print(f"   청크 {chunk_num}: {len(current_chunk)}개 저장 완료")
                current_chunk = []
                chunk_num += 1
                
                # 메모리 정리
                del chunk_dataset
                gc.collect()
        
        # 마지막 청크 처리
        if current_chunk:
            if chunk_num >= start_chunk_num:
                chunk_dataset = Dataset.from_list(current_chunk, features=features)
                temp_chunk_path = f"{temp_chunk_dir}/chunk_{chunk_num}"
                chunk_dataset.save_to_disk(temp_chunk_path)
                chunk_datasets.append(temp_chunk_path)
                print(f"   청크 {chunk_num}: {len(current_chunk)}개 저장 완료")
            del chunk_dataset
            gc.collect()
        
        # 하나의 리포지토리에 청크들을 순차적으로 이어서 추가
        print(f"📤 총 {len(chunk_datasets)}개 신규 청크와 기존 청크를 하나의 리포지토리에 순차 추가...")
        
        all_chunk_paths_to_process = []
        if start_chunk_num > 0:
            print(f"기존 청크 (0 ~ {start_chunk_num - 1})를 처리 목록에 추가 중...")
            for i in range(start_chunk_num):
                path = os.path.join(temp_chunk_dir, f"chunk_{i}")
                if os.path.exists(path):
                    all_chunk_paths_to_process.append(path)
                else:
                    print(f"  [경고] 기존 청크를 찾을 수 없어 건너뜁니다: {path}")

        all_chunk_paths_to_process.extend(chunk_datasets)
        print(f"✅ 총 {len(all_chunk_paths_to_process)}개의 청크를 처리합니다.")

        accumulated_dataset = None
        successful_chunks = 0
        failed_uploads = []
        
        for i, chunk_path in enumerate(all_chunk_paths_to_process):
            chunk_dataset = Dataset.load_from_disk(chunk_path)
            
            print(f"   청크 {i+1}/{len(all_chunk_paths_to_process)} 처리 중... ({os.path.basename(chunk_path)})")
            
            try:
                # 첫 번째 청크이거나 accumulated_dataset이 None인 경우
                if accumulated_dataset is None:
                    accumulated_dataset = chunk_dataset
                    print(f"     첫 번째 청크 ({len(chunk_dataset)}개 샘플) 준비")
                else:
                    # 기존 데이터에 새 청크 추가
                    print(f"     기존 {len(accumulated_dataset)}개에 {len(chunk_dataset)}개 추가 중...")
                    from datasets import concatenate_datasets
                    accumulated_dataset = concatenate_datasets([accumulated_dataset, chunk_dataset])
                    print(f"     총 {len(accumulated_dataset)}개 샘플로 확장")
                
                successful_chunks += 1
                
                # 10개 청크마다 또는 마지막 청크일 때 업로드
                should_upload = (i + 1) % 10 == 0 or i == len(all_chunk_paths_to_process) - 1
                
                if should_upload:
                    print(f"     📤 중간 업로드 ({successful_chunks}개 청크, {len(accumulated_dataset)}개 샘플)...")
                    
                    # 재시도 로직으로 업로드
                    upload_success = False
                    for attempt in range(3):
                        try:
                            if attempt > 0:
                                wait_time = 30 * (2 ** attempt)
                                print(f"       재시도 {attempt+1}/3... {wait_time}초 대기 후")
                                time.sleep(wait_time)
                            
                            accumulated_dataset.push_to_hub(
                                repo_id,  # 항상 같은 리포지토리에 업로드
                                private=private,
                                max_shard_size="100MB",
                                commit_message=f"Add chunks 1-{successful_chunks}: {len(accumulated_dataset)} total samples with images",
                                embed_external_files=True  # 이미지 포함
                            )
                            print(f"       ✅ 업로드 성공! (총 {len(accumulated_dataset)}개 샘플)")
                            upload_success = True
                            break
                            
                        except Exception as e:
                            print(f"       시도 {attempt+1} 실패: {e}")
                            if "429" in str(e) or "Too Many Requests" in str(e):
                                if attempt < 2:
                                    print(f"       429 에러 - 추가 대기...")
                                    time.sleep(60 * (attempt + 1))
                                continue
                    
                    if not upload_success:
                        print(f"       ❌ 업로드 최종 실패")
                        failed_uploads.append((i, f"chunks_1_to_{successful_chunks}"))
                        break  # 업로드 실패 시 중단
                
            except Exception as e:
                print(f"   ❌ 청크 {i+1} 처리 실패: {e}")
                failed_uploads.append((i, f"chunk_{i+1}", chunk_path))
                # 실패한 청크는 건너뛰고 계속 진행
            
            # 메모리 정리
            del chunk_dataset
            gc.collect()
            
            # 성공한 청크 임시 파일 정리
            shutil.rmtree(chunk_path, ignore_errors=True)
            
            # API 제한 회피를 위한 대기
            if i < len(all_chunk_paths_to_process) - 1:
                wait_time = 5  # 5초 대기
                print(f"     다음 청크까지 {wait_time}초 대기...")
                time.sleep(wait_time)
        
        # 최종 결과 리포트
        print(f"\n🎉 업로드 완료!")
        if accumulated_dataset:
            print(f"✅ 최종 데이터셋: {len(accumulated_dataset):,}개 샘플")
            print(f"📋 리포지토리: https://huggingface.co/datasets/{repo_id}")
        
        print(f"✅ 처리된 청크: {successful_chunks}/{len(all_chunk_paths_to_process)}개")
        print(f"❌ 실패: {len(failed_uploads)}개")
        
        if failed_uploads:
            print(f"\n🔄 실패한 항목들:")
            for chunk_idx, description, *extra in failed_uploads:
                print(f"   - {description}")
        
        return len(failed_uploads) == 0  # 모든 청크가 성공했으면 True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 업로드 실패: {e}")
        return False
    finally:
        # 최종 메모리 정리
        gc.collect()

def inspect_dataset(
    dataset_path: str = "./unified-multimodal-sft"
):
    """생성된 데이터셋 검사"""
    try:
        print(f"🔍 데이터셋 검사: {dataset_path}")
        
        # DatasetDict인지 Dataset인지 확인
        if os.path.exists(os.path.join(dataset_path, "dataset_dict.json")):
            # DatasetDict 형태로 저장된 경우
            from datasets import DatasetDict
            dataset_dict = DatasetDict.load_from_disk(dataset_path)
            if "train" in dataset_dict:
                dataset = cast(Dataset, dataset_dict["train"])
                print(f"📊 DatasetDict에서 train split 로드 - 총 샘플 수: {len(dataset)}")
            else:
                # 첫 번째 split 사용
                split_name = list(dataset_dict.keys())[0]
                dataset = cast(Dataset, dataset_dict[split_name])
                print(f"📊 DatasetDict에서 '{split_name}' split 로드 - 총 샘플 수: {len(dataset)}")
        else:
            # 일반 Dataset 형태로 저장된 경우
            dataset = cast(Dataset, Dataset.load_from_disk(dataset_path))
            print(f"📊 총 샘플 수: {len(dataset)}")
        
        # 구조 검사
        sample_with_image = None
        sample_without_image = None
        
        for sample_any in dataset:
            sample = cast(Dict[str, Any], sample_any)
            if sample.get("images") and not sample_with_image:
                sample_with_image = sample
            elif not sample.get("images") and not sample_without_image:
                sample_without_image = sample
            
            if sample_with_image and sample_without_image:
                break
        
        # 이미지 포함 샘플 예시
        if sample_with_image:
            print(f"\n🖼️ 이미지 포함 샘플 예시:")
            print(f"   이미지 수: {len(sample_with_image['images'])}")
            print(f"   메시지 수: {len(sample_with_image['messages'])}")
            
            # 첫 번째 메시지 구조 확인 (멀티모달 스키마)
            if sample_with_image['messages']:
                first_msg = sample_with_image['messages'][0]
                print(f"   첫 번째 메시지 role: {first_msg.get('role')}")
                content_list = first_msg.get('content', [])
                print(f"   첫 번째 메시지 content 수: {len(content_list)}")
                for j, content in enumerate(content_list[:3]):
                    ctype = content.get('type')
                    if ctype == 'text':
                        text_preview = (content.get('text') or '')
                        if len(text_preview) > 80:
                            text_preview = text_preview[:80] + '...'
                        print(f"     Content {j+1}: text - '{text_preview}'")
                    else:
                        print(f"     Content {j+1}: {ctype}")
        
        # 통계
        image_count = sum(1 for s in dataset if cast(Dict[str, Any], s).get("images"))
        print(f"\n📈 이미지 포함 샘플: {image_count}/{len(dataset)} ({image_count/len(dataset)*100:.1f}%)")
        
        # 원본 데이터셋별 통계
        source_stats: Dict[str, int] = {}
        for s in dataset:
            sample = cast(Dict[str, Any], s)
            source = sample.get("source_dataset", "unknown")
            source_stats[source] = source_stats.get(source, 0) + 1
        
        print(f"\n📊 원본 데이터셋별 분포:")
        for source, count in sorted(source_stats.items()):
            print(f"   {source}: {count}개 ({count/len(dataset)*100:.1f}%)")
        
        # 원본 데이터 보존 확인
        original_data_count = sum(1 for s in dataset if cast(Dict[str, Any], s).get("original_data"))
        print(f"\n💾 원본 데이터 보존: {original_data_count}/{len(dataset)} ({original_data_count/len(dataset)*100:.1f}%)")
        
        # 시스템 프롬프트 및 모드 통계
        system_prompt_count = sum(1 for s in dataset if cast(Dict[str, Any], s).get("system_prompt"))
        print(f"\n🤖 시스템 프롬프트 포함: {system_prompt_count}/{len(dataset)} ({system_prompt_count/len(dataset)*100:.1f}%)")
        
        # 모드별 통계
        mode_stats = {}
        for s in dataset:
            sample = cast(Dict[str, Any], s)
            mode = sample.get("dataset_mode", "unknown")
            mode_stats[mode] = mode_stats.get(mode, 0) + 1
        
        print(f"\n🎯 모드별 분포:")
        for mode, count in sorted(mode_stats.items()):
            print(f"   {mode.upper()} 모드: {count}개 ({count/len(dataset)*100:.1f}%)")
        
        # 시스템 프롬프트 예시 (첫 번째 샘플)
        if len(dataset) > 0:
            first_sample = cast(Dict[str, Any], dataset[0])
            if first_sample.get("system_prompt"):
                print(f"\n📝 시스템 프롬프트 예시 (첫 번째 샘플):")
                system_prompt = first_sample["system_prompt"]
                if len(system_prompt) > 100:
                    print(f"   {system_prompt[:100]}...")
                else:
                    print(f"   {system_prompt}")
                
                # 모드 정보도 표시
                if first_sample.get("dataset_mode"):
                    print(f"   기본 모드: {first_sample['dataset_mode'].upper()}")
        
        # 원본 데이터 예시 (첫 번째 샘플)
        if len(dataset) > 0:
            first_sample = cast(Dict[str, Any], dataset[0])
            if first_sample.get("original_data"):
                print(f"\n🔍 원본 데이터 예시 (첫 번째 샘플):")
                try:
                    original_str = first_sample["original_data"]
                    original = json.loads(original_str)
                    print(f"   원본 데이터 키: {list(original.keys())}")
                    for key, value in list(original.items())[:3]:  # 처음 3개 키만 표시
                        if isinstance(value, str) and len(value) > 50:
                            print(f"   {key}: {value[:50]}...")
                        else:
                            print(f"   {key}: {value}")
                except (json.JSONDecodeError, TypeError):
                     print(f"   원본 데이터 (raw): {first_sample['original_data'][:100]}...")

        return dataset
        
    except Exception as e:
        print(f"❌ 검사 중 오류: {str(e)}")
        return None


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="텍스트 + 멀티모달 통합 데이터셋 처리 및 업로드 스크립트")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # merge 명령어
    parser_merge = subparsers.add_parser("merge", help="여러 데이터셋을 병합하여 로컬에 저장합니다.")
    parser_merge.add_argument("--output_name", type=str, help="생성할 데이터셋의 로컬 폴더 이름")
    parser_merge.add_argument("--max_samples", type=int, default=None, help="데이터셋별 최대 샘플 수")
    parser_merge.add_argument("--num_workers", type=int, default=16, help="데이터 처리 워커 수")
    parser_merge.add_argument("--local_path", type=str, default="./", help="데이터셋을 저장할 로컬 경로")

    # upload 명령어
    parser_upload = subparsers.add_parser("upload", help="로컬에 저장된 데이터셋을 허깅페이스 허브에 업로드합니다.")
    parser_upload.add_argument("--dataset_path", type=str, help="업로드할 로컬 데이터셋 경로")
    parser_upload.add_argument("--repo_id", type=str, help="허깅페이스 허브 리포지토리 ID (예: username/repo-name)")
    parser_upload.add_argument("--private", action="store_true", help="리포지토리를 비공개로 설정")
    parser_upload.add_argument("--num_workers", type=int, default=None, help="처리 워커 수 (기본값: CPU 코어 수)")
    parser_upload.add_argument("--chunk_size", type=int, default=None, help="메모리 처리 청크 크기 (기본값: 동적 계산)")
    parser_upload.add_argument("--single_repo", action="store_true", help="하나의 리포지토리에 순차적으로 추가")
    parser_upload.add_argument("--start_chunk_num", type=int, default=0, help="업로드를 시작할 청크 번호. 이 번호 이전의 청크는 생성/처리를 건너뜁니다.")

    # inspect 명령어
    parser_inspect = subparsers.add_parser("inspect", help="로컬 데이터셋의 정보를 확인합니다.")
    parser_inspect.add_argument("dataset_path", nargs="?", default="./unified-multimodal-sft", help="검사할 데이터셋 경로")

    args = parser.parse_args()

    if args.command == "merge":
        print(f"🎯 타겟 로컬 경로: {os.path.join(args.local_path, args.output_name)}")
        print(f"🔧 워커 수: {args.num_workers}")
        final_path = merge_and_create_dataset(
            output_name=args.output_name,
            max_samples_per_dataset=args.max_samples,
            num_workers=args.num_workers,
            local_path=args.local_path
        )
        if final_path:
            print("\n🎉 병합 완료!")
            print(f"✅ 데이터셋이 '{final_path}'에 저장되었습니다.")
            print(f"\n👉 이제 다음 명령어로 허브에 업로드할 수 있습니다:")
            print(f"   python {sys.argv[0]} upload {final_path} <your_hf_username>/{args.output_name}")

    elif args.command == "upload":
        upload_dataset_to_hub(
            dataset_path=args.dataset_path,
            repo_id=args.repo_id,
            private=args.private,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            single_repo=args.single_repo,
            start_chunk_num=args.start_chunk_num
        )

    elif args.command == "inspect":
        inspect_dataset(args.dataset_path)

if __name__ == "__main__":
    main()
    