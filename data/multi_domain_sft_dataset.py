import logging
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names, concatenate_datasets, load_dataset_builder, Features, Sequence, Value
from transformers import AutoProcessor
import torch
from typing import Dict, Any, List, Optional, Tuple
import traceback
import gc
import os
import sys
import random
import tempfile
import pathlib
import shutil
import json
import hashlib
from datetime import datetime
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset, Image as DatasetImage, Sequence, Features, Value
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
import threading

# simple_sft_dataset의 유틸리티 함수들 import
from data.simple_sft_dataset import (
    validate_image_data,
    validate_messages,
    safe_flatten_images,
    get_memory_usage,
    log_memory_usage
)

# ============================================================
# Monkey Patch for datasets library compatibility
# ============================================================
try:
    import datasets.features.features
    # Check if 'List' type is missing (datasets < 4.0.0)
    if "List" not in datasets.features.features._FEATURE_TYPES:
        print("🛠️ Monkey-patching: Registering 'List' feature type as alias for 'Sequence'")
        datasets.features.features._FEATURE_TYPES["List"] = datasets.features.features.Sequence
except Exception as e:
    print(f"⚠️ Failed to apply monkey patch for datasets library: {e}")

def ensure_string(value: Any) -> str:
    """
    값을 문자열로 변환합니다. None이면 빈 문자열을 반환합니다.
    """
    if value is None:
        return ""
    return str(value)

SFT_JSON_FEATURES = Features({
    "messages": [
        {
            "role": Value("string"),
            "content": [
                {
                    "type": Value("string"),
                    "text": Value("string")
                }
            ]
        }
    ],
    "images": [Value("string")],
    "domain": Value("string"),
    "source": Value("string")
})

def _preprocess_images_for_mapping(example, cache_images_dir=None):
    """
    Dataset.map()에서 사용할 전역 이미지 전처리 함수.
    picklable해야 하므로 최상위 레벨에 정의합니다.
    """
    if 'images' in example and example['images']:
        image_paths = example['images']
        if isinstance(image_paths, list):
            fixed_paths = []
            for img_path in image_paths:
                if isinstance(img_path, str) and img_path.strip():
                    if not os.path.isabs(img_path) and cache_images_dir:
                        img_path = os.path.join(cache_images_dir, os.path.basename(img_path))
                    if os.path.exists(img_path):
                        fixed_paths.append(img_path)
            example['images'] = validate_image_data(fixed_paths)
        else:
            example['images'] = validate_image_data(example['images']) if example['images'] else []
    elif 'images' not in example:
        example['images'] = []
    
    # 텍스트 정규화 추가 (ImportError 방지를 위해 로딩 시 수행하던 로직을 여기로 이전)
    if 'messages' in example and isinstance(example['messages'], list):
        for message in example['messages']:
            if not isinstance(message, dict):
                continue
            if 'content' in message and isinstance(message['content'], list):
                for content_item in message['content']:
                    if not isinstance(content_item, dict):
                        continue
                    if 'text' not in content_item or content_item.get('text') is None:
                        content_item['text'] = ""
                    if 'type' not in content_item:
                        content_item['type'] = "text"
    
    return example

def ensure_messages_text_strings(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    messages의 모든 텍스트 필드를 문자열로 보장합니다.
    content는 항상 리스트 형태의 객체 배열이어야 합니다.
    """
    result = []
    for msg in messages:
        if not isinstance(msg, dict) or "content" not in msg:
            continue
        
        new_msg = msg.copy()
        content = msg.get("content", [])
        
        # content가 문자열인 경우 리스트로 변환
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        # content가 리스트가 아닌 경우 리스트로 변환
        elif not isinstance(content, list):
            content = [content] if content else []
        
        new_content = []
        for content_item in content:
            if isinstance(content_item, dict):
                new_item = content_item.copy()
                if "text" in new_item:
                    new_item["text"] = ensure_string(new_item["text"])
                new_content.append(new_item)
            elif isinstance(content_item, str):
                # 문자열인 경우 객체로 변환
                new_content.append({"type": "text", "text": ensure_string(content_item)})
            else:
                # 기타 타입은 문자열로 변환
                new_content.append({"type": "text", "text": ensure_string(content_item)})
        
        new_msg["content"] = new_content
        result.append(new_msg)
    
    return result

def ensure_vlm_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    모든 샘플을 VLM 형식으로 변환합니다.
    이미지가 있는 경우에만 이미지 placeholder를 추가합니다.
    Qwen3-VL-MoE는 이미지 토큰과 이미지 features의 개수가 일치해야 하므로,
    실제 이미지가 있을 때만 이미지 placeholder를 추가합니다.
    """
    if not isinstance(sample, dict):
        return sample
    
    # messages가 없으면 변환 불가
    if "messages" not in sample or not isinstance(sample["messages"], list):
        return sample
    
    messages = sample["messages"].copy()
    images = sample.get("images", [])
    
    # images가 리스트가 아니면 리스트로 변환
    if not isinstance(images, list):
        images = [images] if images else []
    
    # 이미지가 문자열 경로인지 확인 (실제 이미지 파일이 있는지)
    has_images = False
    for img in images:
        if isinstance(img, str) and img.strip():
            # 파일 경로가 존재하는지 확인
            if os.path.exists(img):
                has_images = True
                break
    
    # 이미지가 있는 경우에만 이미지 placeholder 추가
    if has_images:
        first_user_msg_found = False
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content] if content else []
            
            # 이미지 placeholder가 있는지 확인
            has_image_placeholder = False
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    has_image_placeholder = True
                    break
            
            # 첫 번째 user 메시지에 이미지 placeholder 추가 (없는 경우에만)
            if not first_user_msg_found and not has_image_placeholder:
                content.insert(0, {"type": "image"})
                msg["content"] = content
                first_user_msg_found = True
                break
    
    # messages 정규화
    messages = ensure_messages_text_strings(messages)
    
    result = sample.copy()
    result["messages"] = messages
    result["images"] = images if images else []
    
    return result

def sanitize_sample_for_json(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    샘플을 JSON 직렬화 가능한 형태로 정리합니다.
    PIL Image 객체나 직렬화 불가능한 객체를 제거합니다.
    """
    if not isinstance(sample, dict):
        return sample
    
    result = {}
    for key, value in sample.items():
        if key == "images":
            # images는 문자열 경로 리스트만 유지
            if isinstance(value, list):
                sanitized_images = []
                for img in value:
                    if isinstance(img, str) and img.strip():
                        sanitized_images.append(img)
                    # PIL Image나 다른 객체는 무시 (이미 파일로 저장되어 있어야 함)
                result[key] = sanitized_images
            else:
                result[key] = []
        else:
            # 다른 필드는 그대로 복사 (재귀적으로 처리)
            if isinstance(value, dict):
                result[key] = sanitize_sample_for_json(value)
            elif isinstance(value, list):
                result[key] = [
                    sanitize_sample_for_json(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                # PIL Image나 다른 직렬화 불가능한 객체는 무시
                try:
                    json.dumps(value)
                    result[key] = value
                except (TypeError, ValueError):
                    # 직렬화 불가능한 객체는 문자열로 변환 시도
                    result[key] = str(value)
    
    return result

def dataset_exists(dataset_name: str) -> bool:
    """
    주어진 데이터셋이 Hugging Face Hub에 존재하는지 간단히 확인합니다.
    존재하지 않거나 접근 불가하면 False를 반환합니다.
    """
    try:
        _ = get_dataset_config_names(dataset_name)
        return True
    except Exception:
        logger.warning(f"⚠️ 데이터셋이 존재하지 않거나 접근할 수 없습니다: {dataset_name} (건너뜀)")
        return False

# ============================================================
# 데이터셋별 전용 변환 프로세서 함수들
# ============================================================

def process_rstar_coder(dataset, dataset_name: str, max_samples: int, log_detail: bool = False) -> List[Dict[str, Any]]:
    """
    microsoft/rStar-Coder 데이터셋 전용 프로세서
    데이터셋 전체를 처리하여 messages 형식 리스트 반환
    """
    results = []
    sample_count = 0
    # failed_count removed
    
    logger.debug(f"   🔧 rStar-Coder 전용 프로세서 시작 (최대 {max_samples}개 샘플)")
    
    for idx, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break
        
        try:
            # rStar-Coder의 실제 컬럼: question, seed_question, seed_source, response, code
            question = sample.get("question", "")
            seed_question = sample.get("seed_question", "")
            response = sample.get("response", "")
            code = sample.get("code", "")
            
            # 디버깅: 처음 몇 개 샘플 로깅 (DEBUG 레벨)
            if log_detail and idx < 2:
                logger.debug(f"   🔍 rStar-Coder 샘플 {idx}: keys={list(sample.keys())}")
            
            # question 또는 seed_question 중 하나는 있어야 함
            user_prompt = question.strip() if question and question.strip() else (seed_question.strip() if seed_question and seed_question.strip() else "")
            if not user_prompt:
                raise RuntimeError(f"Sample processing failed")
            
            # response와 code 중 하나는 있어야 함
            assistant_content_parts = []
            if response and response.strip():
                assistant_content_parts.append(response.strip())
            if code and code.strip():
                if assistant_content_parts:
                    assistant_content_parts.append(f"\n\n```python\n{code.strip()}\n```")
                else:
                    assistant_content_parts.append(code.strip())
            
            if not assistant_content_parts:
                raise RuntimeError(f"Sample processing failed")
            
            assistant_text = "\n".join(assistant_content_parts)
            
            messages = [
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
            ]
            
            # Revert to text-only (no dummy image)
            results.append({"messages": messages, "images": []})
            sample_count += 1
            
        except Exception as e:
            logger.error(f"   ❌ rStar-Coder 샘플 {idx} 처리 실패: {e}")
            raise RuntimeError(f"Sample processing failed")
    

    return results

def process_metamath(dataset, dataset_name: str, max_samples: int, log_detail: bool = False) -> List[Dict[str, Any]]:
    """
    meta-math/MetaMathQA 데이터셋 전용 프로세서
    """
    results = []
    sample_count = 0
    # failed_count removed
    
    logger.debug(f"   🔧 MetaMath 전용 프로세서 시작 (최대 {max_samples}개 샘플)")
    
    for idx, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break
        
        try:
            query = sample.get("query", "")
            response = sample.get("response", "")
            
            if not query or not response:
                raise RuntimeError(f"Sample processing failed")
            
            messages = [
                {"role": "user", "content": [{"type": "text", "text": query}]},
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            ]
            
            results.append({"messages": messages, "images": []})
            sample_count += 1
            
        except Exception as e:
            logger.error(f"   ❌ MetaMath 샘플 {idx} 처리 실패: {e}")
            raise RuntimeError(f"Sample processing failed")
    

    return results

def process_math_python_reasoning(dataset, dataset_name: str, max_samples: int, log_detail: bool = False) -> List[Dict[str, Any]]:
    """
    sdiazlor/math-python-reasoning-dataset 전용 프로세서
    """
    results = []
    sample_count = 0
    # failed_count removed
    
    logger.debug(f"   🔧 Math-Python-Reasoning 전용 프로세서 시작 (최대 {max_samples}개 샘플)")
    
    for idx, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break
        
        try:
            prompt = sample.get("prompt", "")
            completion = sample.get("completion", "")
            system_prompt = sample.get("system_prompt", "")
            
            # instruction/output 형식도 지원 (하위 호환성)
            if not prompt:
                prompt = sample.get("instruction", "")
            if not completion:
                completion = sample.get("output", "")
            
            if not prompt or not completion:
                raise RuntimeError(f"Sample processing failed")
            
            messages = []
            
            # system_prompt가 있으면 system 메시지로 추가
            if system_prompt:
                messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
            
            messages.extend([
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": completion}]}
            ])
            
            results.append({"messages": messages, "images": []})
            sample_count += 1
            
        except Exception as e:
            logger.error(f"   ❌ Math-Python-Reasoning 샘플 {idx} 처리 실패: {e}")
            raise RuntimeError(f"Sample processing failed")
    

    return results

def process_llava_onevision(dataset, dataset_name: str, max_samples: int, log_detail: bool = False) -> List[Dict[str, Any]]:
    """
    lmms-lab/LLaVA-OneVision-Data 전용 프로세서 (multimodal)
    """
    results = []
    sample_count = 0
    # failed_count removed
    
    logger.debug(f"   🔧 LLaVA-OneVision 전용 프로세서 시작 (최대 {max_samples}개 샘플)")
    
    for idx, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break
        
        try:
            # conversations + images 형식
            conversations = sample.get("conversations", [])
            images = sample.get("images", [])
            
            if not conversations:
                raise RuntimeError(f"Sample processing failed")
            
            messages = []
            for conv in conversations:
                role = conv.get("from", "")
                value = conv.get("value", "")
                
                if role == "human":
                    messages.append({"role": "user", "content": [{"type": "text", "text": value}]})
                elif role == "gpt":
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": value}]})
            
            if not messages:
                raise RuntimeError(f"Sample processing failed")
            
            # 이미지 처리
            image_list = []
            if images:
                flattened_images = validate_image_data(images)
                image_list = flattened_images if flattened_images else []
            
            results.append({"messages": messages, "images": image_list})
            sample_count += 1
            
        except Exception as e:
            logger.error(f"   ❌ LLaVA-OneVision 샘플 {idx} 처리 실패: {e}")
            raise RuntimeError(f"Sample processing failed")
    

    return results

def process_olmocr(dataset, dataset_name: str, max_samples: int, log_detail: bool = False) -> List[Dict[str, Any]]:
    """
    allenai/olmOCR-mix-1025 전용 프로세서 (OCR, multimodal)
    
    데이터 구조:
    - natural_text: OCR된 텍스트 (ground truth)
    - pdf_relpath: PDF 파일 경로 (tar.gz 내부)
    - url: 원본 PDF URL
    - image: PDF 페이지 이미지 (있는 경우)
    
    VLM용 instruction: "이 문서 페이지의 텍스트를 추출하세요"
    """
    results = []
    sample_count = 0
    # failed_count removed
    
    logger.debug(f"   🔧 olmOCR 전용 프로세서 시작 (최대 {max_samples}개 샘플)")
    
    for idx, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break
        
        try:
            # OCR된 텍스트 (ground truth)
            natural_text = sample.get("natural_text", "")
            if not natural_text or not natural_text.strip():
                if log_detail and idx < 5:
                    logger.warning(f"   ⚠️ olmOCR 샘플 {idx}: natural_text가 비어있음")
                raise RuntimeError(f"Sample processing failed")
            
            # 이미지 처리 (PDF 페이지 이미지)
            image_list = []
            
            # 1. image 필드 확인 (DatasetImage 타입)
            if "image" in sample:
                img = sample["image"]
                if img is not None:
                    flattened_images = validate_image_data([img])
                    if flattened_images:
                        image_list = flattened_images
            
            # 2. images 필드 확인 (리스트)
            if not image_list and "images" in sample:
                images = sample["images"]
                if images:
                    flattened_images = validate_image_data(images)
                    if flattened_images:
                        image_list = flattened_images
            
            # 이미지가 없으면 OCR 태스크가 불가능하므로 건너뛰기
            if not image_list:
                raise RuntimeError(f"Sample processing failed")
            
            # Instruction: OCR 태스크
            instruction = "이 문서 페이지의 텍스트를 추출하세요. 이미지에서 보이는 모든 텍스트를 정확하게 읽어주세요."
            
            # Messages 구성 (이미지 필수)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": natural_text.strip()}]
                }
            ]
            
            results.append({"messages": messages, "images": image_list})
            sample_count += 1
            
        except Exception as e:
            logger.debug(f"   ❌ olmOCR 샘플 {idx} 처리 실패: {e}")
            raise RuntimeError(f"Sample processing failed")
    

    
    return results

def process_cord(dataset, dataset_name: str, max_samples: int, log_detail: bool = False) -> List[Dict[str, Any]]:
    """
    naver-clova-ix/cord-v2 전용 프로세서 (OCR, multimodal)
    
    데이터 구조 (웹사이트 참고):
    - image: 영수증 이미지 (DatasetImage 타입)
    - ground_truth: OCR된 텍스트 (ground truth, JSON 형식일 수 있음)
    
    VLM용 instruction: "이 영수증 이미지에서 텍스트를 추출하세요"
    """
    results = []
    sample_count = 0
    # failed_count removed
    
    logger.debug(f"   🔧 CORD-v2 전용 프로세서 시작 (최대 {max_samples}개 샘플)")
    
    for idx, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break
        
        try:
            # OCR된 텍스트 (ground truth)
            ground_truth = sample.get("ground_truth", "")
            if not ground_truth or not str(ground_truth).strip():
                if log_detail and idx < 5:
                    logger.warning(f"   ⚠️ CORD-v2 샘플 {idx}: ground_truth가 비어있음")
                raise RuntimeError(f"Sample processing failed")
            
            # ground_truth가 JSON 형식일 수 있으므로 문자열로 변환
            if isinstance(ground_truth, dict):
                import json
                ground_truth = json.dumps(ground_truth, ensure_ascii=False)
            else:
                ground_truth = str(ground_truth).strip()
            
            # 이미지 처리
            image_list = []
            
            # image 필드 확인 (DatasetImage 타입)
            if "image" in sample:
                img = sample["image"]
                if img is not None:
                    flattened_images = validate_image_data([img])
                    if flattened_images:
                        image_list = flattened_images
            
            # 이미지가 없으면 OCR 태스크가 불가능하므로 건너뛰기
            if not image_list:
                raise RuntimeError(f"Sample processing failed")
            
            # Instruction: OCR 태스크 (영수증 특화)
            instruction = "이 영수증 이미지에서 텍스트를 추출하세요. 이미지에서 보이는 모든 텍스트를 정확하게 읽어주세요."
            
            # Messages 구성 (이미지 필수)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ground_truth}]
                }
            ]
            
            results.append({"messages": messages, "images": image_list})
            sample_count += 1
            
        except Exception as e:
            logger.debug(f"   ❌ CORD-v2 샘플 {idx} 처리 실패: {e}")
            raise RuntimeError(f"Sample processing failed")
    

    
    return results

def process_ask_science_qg(dataset, dataset_name: str, max_samples: int, log_detail: bool = False) -> List[Dict[str, Any]]:
    """
    dhmeltzer/ask-science-qg 전용 프로세서 (Science Q&A, 텍스트 전용)
    
    데이터 구조 (웹사이트 참고):
    - title: 질문 제목
    - selftext: 질문 본문 (선택적, 비어있을 수 있음)
    - answers.text: 답변 텍스트 (sequence/리스트)
    - answers.score: 답변 점수 (sequence/리스트)
    
    VLM용 instruction: title + selftext를 질문으로, answers.text를 답변으로
    """
    results = []
    sample_count = 0
    # failed_count removed
    
    logger.debug(f"   🔧 ask-science-qg 전용 프로세서 시작 (최대 {max_samples}개 샘플)")
    
    # 처음 몇 개 샘플의 실제 구조를 확인하기 위한 디버깅 (DEBUG 레벨)
    debug_samples_checked = 0
    max_debug_samples = 3  # 10 -> 3으로 감소
    
    for idx, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break
        
        try:
            # 처음 몇 개 샘플의 전체 구조 로깅 (DEBUG 레벨)
            if debug_samples_checked < max_debug_samples and log_detail:
                logger.debug(f"   🔍 ask-science-qg 샘플 {idx} 구조: keys={list(sample.keys())}")
                debug_samples_checked += 1
            
            # 질문 구성: title + selftext
            title = sample.get("title", "")
            selftext = sample.get("selftext", "")
            
            # title은 필수
            if not title or not str(title).strip():
                raise RuntimeError(f"Sample processing failed")
            
            # 질문 텍스트 구성
            question_parts = [str(title).strip()]
            if selftext and str(selftext).strip():
                question_parts.append(str(selftext).strip())
            question = "\n\n".join(question_parts)
            
            # 답변 추출 - 웹사이트 구조: answers.text (sequence), answers.score (sequence)
            # HuggingFace datasets에서 sequence는 평탄화될 수 있음: answers.text -> 최상위 레벨
            answer_text = ""
            
            # Case 1: answers.text가 평탄화되어 최상위 레벨에 있는 경우
            answer_texts = sample.get("answers.text", None)
            answer_scores = sample.get("answers.score", None)
            
            # Case 2: answers가 dict이고 answers.text가 있는 경우
            if answer_texts is None:
                answers = sample.get("answers", {})
                if isinstance(answers, dict) and len(answers) > 0:
                    answer_texts = answers.get("text", None)
                    answer_scores = answers.get("score", None)
            
            # Case 3: answers가 리스트인 경우
            if answer_texts is None:
                answers = sample.get("answers", {})
                if isinstance(answers, (list, tuple)) and len(answers) > 0:
                    first_answer = answers[0]
                    if isinstance(first_answer, dict):
                        answer_texts = first_answer.get("text", None)
                    else:
                        answer_texts = first_answer
            
            # Case 4: answers가 직접 문자열인 경우
            if answer_texts is None:
                answers = sample.get("answers", "")
                if isinstance(answers, str):
                    answer_texts = answers
            
            # answer_texts 처리
            if answer_texts is not None:
                # sequence를 리스트로 변환 (이미 리스트일 수도 있음)
                if not isinstance(answer_texts, (list, tuple)):
                    # 단일 값인 경우 리스트로 변환
                    answer_texts = [answer_texts]
                
                if len(answer_texts) > 0:
                    # scores가 있고 길이가 같으면 가장 높은 점수 선택
                    if answer_scores is not None:
                        if not isinstance(answer_scores, (list, tuple)):
                            answer_scores = [answer_scores]
                        
                        if len(answer_scores) == len(answer_texts) and len(answer_scores) > 0:
                            try:
                                # 가장 높은 점수의 답변 선택
                                best_idx = max(range(len(answer_scores)), key=lambda i: answer_scores[i] if isinstance(answer_scores[i], (int, float)) else 0)
                                answer_text = str(answer_texts[best_idx]).strip()
                            except Exception as e:
                                # 점수 선택 실패 시 첫 번째 답변 사용
                                answer_text = str(answer_texts[0]).strip()
                        else:
                            # scores 길이가 다르면 첫 번째 답변 사용
                            answer_text = str(answer_texts[0]).strip()
                    else:
                        # scores가 없으면 첫 번째 답변 사용
                        answer_text = str(answer_texts[0]).strip()
            
            # 답변이 비어있으면 건너뛰기
            if not answer_text:
                raise RuntimeError(f"Sample processing failed")
            
            # Messages 구성
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text}]
                }
            ]
            
            results.append({"messages": messages, "images": []})
            sample_count += 1
            
        except Exception as e:
            logger.debug(f"   ❌ ask-science-qg 샘플 {idx} 처리 실패: {e}")
            raise RuntimeError(f"Sample processing failed")
    

    
    return results

def process_ocr_vqa(dataset, dataset_name: str, max_samples: int, log_detail: bool = False) -> List[Dict[str, Any]]:
    """
    howard-hou/OCR-VQA 전용 프로세서 (OCR VQA, multimodal)
    
    데이터 구조 (웹사이트 참고):
    - image: 이미지 (DatasetImage 타입)
    - questions: 질문 리스트 (sequence)
    - answers: 답변 리스트 (sequence)
    - ocr_tokens: OCR 토큰 (sequence, 선택적)
    - ocr_info: OCR 정보 (list, 선택적)
    
    하나의 이미지에 여러 질문-답변 쌍이 있으므로, 각 쌍을 별도 샘플로 생성
    """
    results = []
    sample_count = 0
    # failed_count removed
    
    logger.debug(f"   🔧 OCR-VQA 전용 프로세서 시작 (최대 {max_samples}개 샘플)")
    
    debug_samples_checked = 0
    max_debug_samples = 2  # 5 -> 2로 감소
    
    for idx, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break
        
        try:
            # 처음 몇 개 샘플의 구조 확인 (DEBUG 레벨)
            if debug_samples_checked < max_debug_samples and log_detail:
                logger.debug(f"   🔍 OCR-VQA 샘플 {idx} 구조: keys={list(sample.keys())}")
                debug_samples_checked += 1
            
            # 이미지 처리
            image = sample.get("image", None)
            if image is None:
                raise RuntimeError(f"Sample processing failed")
            
            # 이미지 검증
            image_list = validate_image_data([image])
            if not image_list:
                raise RuntimeError(f"Sample processing failed")
            
            # questions와 answers 추출
            questions = sample.get("questions", None)
            answers = sample.get("answers", None)
            
            # questions와 answers가 리스트인지 확인
            if not isinstance(questions, (list, tuple)) or not isinstance(answers, (list, tuple)):
                raise RuntimeError(f"Sample processing failed")
            
            if len(questions) == 0 or len(answers) == 0:
                raise RuntimeError(f"Sample processing failed")
            
            # 질문과 답변의 개수가 다를 수 있으므로, 최소 개수만큼만 처리
            num_pairs = min(len(questions), len(answers))
            
            # 각 질문-답변 쌍을 별도 샘플로 생성
            for qa_idx in range(num_pairs):
                if sample_count >= max_samples:
                    break
                
                question = str(questions[qa_idx]).strip()
                answer = str(answers[qa_idx]).strip()
                
                if not question or not answer:
                    continue
                
                # Messages 구성
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}]
                    }
                ]
                
                results.append({"messages": messages, "images": image_list})
                sample_count += 1
            
        except Exception as e:
            logger.debug(f"   ❌ OCR-VQA 샘플 {idx} 처리 실패: {e}")
            raise RuntimeError(f"Sample processing failed")
    

    
    return results

def process_generic_instruction(dataset, dataset_name: str, max_samples: int, log_detail: bool = False) -> List[Dict[str, Any]]:
    """
    범용 instruction-output 형식 프로세서
    """
    results = []
    sample_count = 0
    # failed_count removed
    
    logger.debug(f"   🔧 범용 Instruction 프로세서 시작 ({dataset_name}, 최대 {max_samples}개 샘플)")
    
    for idx, sample in enumerate(dataset):
        if sample_count >= max_samples:
            break
        
        try:
            # messages 형식이 이미 있는 경우
            if "messages" in sample and sample["messages"]:
                messages = validate_messages(sample["messages"])
                images = sample.get("images", [])
                results.append({"messages": messages, "images": images if images else []})
                sample_count += 1
                continue
            
            # trajectory 형식 (UltraInteract_sft)
            if "trajectory" in sample and sample["trajectory"]:
                trajectory = sample["trajectory"]
                if isinstance(trajectory, list):
                    messages = []
                    for turn in trajectory:
                        if isinstance(turn, dict):
                            role = turn.get("role", "")
                            content = turn.get("content", "")
                            if role and content:
                                if role in ["user", "human"]:
                                    messages.append({"role": "user", "content": [{"type": "text", "text": str(content)}]})
                                elif role in ["assistant", "gpt"]:
                                    messages.append({"role": "assistant", "content": [{"type": "text", "text": str(content)}]})
                    if messages:
                        results.append({"messages": messages, "images": []})
                        sample_count += 1
                        continue
            
            # conversations 형식
            if "conversations" in sample:
                conversations = sample["conversations"]
                messages = []
                for conv in conversations:
                    role = conv.get("from", conv.get("role", ""))
                    value = conv.get("value", conv.get("content", ""))
                    
                    if role in ["human", "user"]:
                        messages.append({"role": "user", "content": [{"type": "text", "text": value}]})
                    elif role in ["gpt", "assistant"]:
                        messages.append({"role": "assistant", "content": [{"type": "text", "text": value}]})
                
                if messages:
                    results.append({"messages": messages, "images": []})
                    sample_count += 1
                    continue
            
            # instruction-output 형식
            if "instruction" in sample:
                instruction = sample.get("instruction", "")
                output = sample.get("output", sample.get("response", ""))
                
                if instruction and output:
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text": instruction}]},
                        {"role": "assistant", "content": [{"type": "text", "text": output}]}
                    ]
                    results.append({"messages": messages, "images": []})
                    sample_count += 1
                    continue
            
            # prompt-response 형식
            if "prompt" in sample:
                prompt = sample.get("prompt", "")
                response = sample.get("response", sample.get("completion", sample.get("output", "")))
                
                if prompt and response:
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text": str(prompt)}]},
                        {"role": "assistant", "content": [{"type": "text", "text": str(response)}]}
                    ]
                    results.append({"messages": messages, "images": []})
                    sample_count += 1
                    continue
            
            # question-answer 형식
            if "question" in sample:
                question = sample.get("question", "")
                answer = sample.get("answer", sample.get("response", ""))
                
                # ScienceQA 형식 지원 (choices, hint, solution 포함)
                if "choices" in sample and "answer" in sample:
                    choices = sample["choices"]
                    hint = sample.get("hint", "")
                    solution = sample.get("solution", "")
                    
                    # 질문 구성
                    query_text = question
                    if hint:
                        query_text = f"Hint: {hint}\n{query_text}"
                    
                    if choices:
                        query_text += "\nChoices:\n"
                        for i, choice in enumerate(choices):
                            query_text += f"({i}) {choice}\n"
                    
                    # 정답 구성
                    answer_idx = sample["answer"]
                    try:
                        # answer가 정수 인덱스인 경우
                        if isinstance(answer_idx, int):
                            answer_text = choices[answer_idx]
                        else:
                            answer_text = str(answer_idx)
                    except:
                        answer_text = str(answer_idx)
                        
                    if solution:
                        answer_text += f"\n\nExplanation: {solution}"
                    
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text": query_text}]},
                        {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
                    ]
                    
                    # 이미지 처리
                    images = []
                    if "image" in sample and sample["image"]:
                         # 이미지가 1개라고 가정 (ScienceQA는 1개) <= No this is fallback code. Use Every Image in datasetss
                         if isinstance(sample["image"], list):
                             images = sample["image"]
                         else:
                             images = [sample["image"]]
                    
                    results.append({"messages": messages, "images": images})
                    sample_count += 1
                    continue
                
                # 일반 QA
                if question and answer:
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text": question}]},
                        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
                    ]
                    results.append({"messages": messages, "images": []})
                    sample_count += 1
                    continue
            
            # 변환 실패 - 엔거하게 에러 발생
            sample_keys = list(sample.keys()) if isinstance(sample, dict) else type(sample).__name__
            raise RuntimeError(f"[{dataset_name}] 샘플 {idx} 변환 실패. 키: {sample_keys}")
            
        except Exception as e:
            raise RuntimeError(f"[{dataset_name}] 샘플 {idx} 처리 중 오류: {e}")
    
    return results

# 데이터셋별 프로세서 매핑
DATASET_PROCESSORS = {
    "microsoft/rStar-Coder": process_rstar_coder,
    "meta-math/MetaMathQA": process_metamath,
    "sdiazlor/math-python-reasoning-dataset": process_math_python_reasoning,
    "lmms-lab/LLaVA-OneVision-Data": process_llava_onevision,
    "allenai/olmOCR-mix-1025": process_olmocr,
    "naver-clova-ix/cord-v2": process_cord,
    "dhmeltzer/ask-science-qg": process_ask_science_qg,
    "howard-hou/OCR-VQA": process_ocr_vqa,
}

def get_processor_for_dataset(dataset_name: str):
    """
    데이터셋 이름에 따라 적절한 프로세서 반환
    """
    # 정확한 이름 매칭
    if dataset_name in DATASET_PROCESSORS:
        return DATASET_PROCESSORS[dataset_name]
    
    # 부분 매칭 (소문자 변환)
    dataset_name_lower = dataset_name.lower()
    for key, processor in DATASET_PROCESSORS.items():
        if key.lower() in dataset_name_lower:
            return processor
    
    # 기본 프로세서
    return process_generic_instruction


# ============================================================
# 하위 호환성: 기존 convert_sample_to_messages 함수 (deprecated)
# ============================================================

def convert_sample_to_messages(sample: Dict[str, Any], dataset_name: str, log_failure: bool = False) -> Optional[Dict[str, Any]]:
    """
    샘플을 messages 형식으로 변환 (하위 호환성용 - deprecated)
    
    이 함수는 하위 호환성을 위해 남겨둡니다.
    새로운 코드에서는 데이터셋별 프로세서를 직접 사용하세요.
    
    Args:
        sample: 변환할 샘플
        dataset_name: 데이터셋 이름
        log_failure: 변환 실패 시 로그를 남길지 여부
    """
    # rStar-Coder 형식 처리 (가장 먼저 체크 - 구체적인 데이터셋)
    if "rstar-coder" in dataset_name.lower() or "rstar_coder" in dataset_name.lower():
        # 실제 keys 확인 및 처리
        sample_keys = list(sample.keys()) if isinstance(sample, dict) else []
        
        # rStar-Coder의 실제 컬럼: question, seed_question, seed_source, response, code
        question = sample.get("question", "")
        seed_question = sample.get("seed_question", "")
        seed_source = sample.get("seed_source", "")
        response = sample.get("response", "")
        code = sample.get("code", "")
        
        # 디버깅: 처음 몇 개 샘플의 keys 로깅
        if log_failure:
            logger.debug(f"   🔍 rStar-Coder 샘플 keys: {sample_keys}")
            logger.debug(f"   🔍 question: {bool(question)}, seed_question: {bool(seed_question)}, response: {bool(response)}, code: {bool(code)}")
            if question:
                logger.debug(f"   🔍 question preview: {question[:100]}")
            if seed_question:
                logger.debug(f"   🔍 seed_question preview: {seed_question[:100]}")
            if response:
                logger.debug(f"   🔍 response preview: {response[:100]}")
            if code:
                logger.debug(f"   🔍 code preview: {code[:100]}")
        
        # question 또는 seed_question 중 하나는 있어야 함
        user_prompt = question.strip() if question and question.strip() else (seed_question.strip() if seed_question and seed_question.strip() else "")
        if not user_prompt:
            if log_failure:
                logger.warning(f"   ⚠️ rStar-Coder: question과 seed_question이 모두 비어있음. keys: {sample_keys}")
            return None
        
        # response와 code 중 하나는 있어야 함
        assistant_content_parts = []
        if response and response.strip():
            assistant_content_parts.append(response.strip())
        if code and code.strip():
            if assistant_content_parts:
                assistant_content_parts.append(f"\n\n```python\n{code.strip()}\n```")
            else:
                assistant_content_parts.append(code.strip())
        
        if not assistant_content_parts:
            if log_failure:
                logger.warning(f"   ⚠️ rStar-Coder: response와 code가 모두 비어있음. keys: {sample_keys}")
            return None
        
        assistant_text = "\n".join(assistant_content_parts)
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]
        return {"messages": messages, "images": []}
    
    # ask-science-qg 형식 처리 (answers.text 사용)
    if "ask-science-qg" in dataset_name.lower():
        question = sample.get("question", "")
        answers = sample.get("answers", {})
        
        # answers.text 필드 추출 (숫자일 수 있으므로 문자열로 변환)
        if isinstance(answers, dict):
            answer_text = answers.get("text", "")
        elif isinstance(answers, list) and len(answers) > 0:
            # 리스트인 경우 첫 번째 항목의 text 사용
            if isinstance(answers[0], dict):
                answer_text = answers[0].get("text", "")
            else:
                answer_text = str(answers[0]) if answers[0] else ""
        else:
            answer_text = ""
        
        # 문자열로 변환 보장
        question = ensure_string(question)
        answer_text = ensure_string(answer_text)
        
        if not question or not answer_text:
            if log_failure:
                sample_keys_str = list(sample.keys()) if isinstance(sample, dict) else 'N/A'
                logger.debug(f"[{dataset_name}] ask-science-qg: 빈 질문 또는 답변 - question: {bool(question)}, answer: {bool(answer_text)}, sample_keys: {sample_keys_str}")
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
        ]
        
        # 텍스트 필드 문자열 보장
        messages = ensure_messages_text_strings(messages)
        
        return {"messages": messages, "images": []}
    
    # ScienceQA 형식 처리
    if "ScienceQA" in dataset_name or "scienceqa" in dataset_name.lower():
        question = sample.get("question", "")
        choices = sample.get("choices", [])
        answer = sample.get("answer", "")
        explanation = sample.get("explanation", "")
        
        # 질문과 선택지 구성
        question_text = question
        if choices:
            choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            question_text = f"{question}\n\n{choices_text}"
        
        # 답변 구성
        answer_text = answer
        if explanation:
            answer_text = f"{answer}\n\nExplanation: {explanation}"
        
        # 이미지 처리
        img = sample.get("image", [])
        if not isinstance(img, list):
            img = [img] if img is not None else []
        img = validate_image_data(img)
        
        # 이미지가 있으면 멀티모달, 없으면 텍스트 전용
        if img:
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
            ]
        else:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": question_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
            ]
        
        return {"messages": messages, "images": img if img else []}
    
    # LLaVA-OneVision-Data 형식 처리
    if "llava-onevision" in dataset_name.lower() or "onevision" in dataset_name.lower():
        # LLaVA 형식: conversations 또는 messages 필드 사용
        if "conversations" in sample:
            messages = []
            img = sample.get("image", [])
            if not isinstance(img, list):
                img = [img] if img is not None else []
            img = validate_image_data(img)
            
            first_user = True
            for conv in sample["conversations"]:
                if isinstance(conv, dict):
                    role = conv.get("from", "").lower()
                    value = conv.get("value", "")
                    
                    if role in ["human", "user"]:
                        content = []
                        if first_user and img:
                            content.append({"type": "image"})
                            first_user = False
                        if value:
                            content.append({"type": "text", "text": str(value)})
                        if content:
                            messages.append({"role": "user", "content": content})
                    elif role in ["gpt", "assistant"]:
                        if value:
                            messages.append({"role": "assistant", "content": [{"type": "text", "text": str(value)}]})
            
            if messages and img:
                return {"messages": messages, "images": img}
            elif messages:
                # 이미지가 없어도 처리
                return {"messages": messages, "images": []}
        
        # UltraFeedback / Binarized 처리
        if "ultrafeedback" in dataset_name.lower():
            # 1. messages 필드 우선 확인
            if "messages" in sample and isinstance(sample["messages"], list):
                try:
                    messages = validate_messages(sample["messages"])
                    return {"messages": messages, "images": []}
                except:
                    pass
            
            # 2. prompt/chosen 확인
            if "prompt" in sample and "chosen" in sample:
                prompt = sample["prompt"]
                chosen = sample["chosen"]
                
                response = ""
                if isinstance(chosen, list):
                    # 리스트인 경우 assistant 메시지 찾기
                    for m in chosen:
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            # content가 리스트일 수도 문자열일 수도 있음
                            content = m.get("content", "")
                            if isinstance(content, list):
                                parts = [x.get("text", "") for x in content if x.get("type") == "text"]
                                response = "\n".join(parts)
                            else:
                                response = str(content)
                            break
                    # 못 찾았으면 마지막 항목 사용
                    if not response and chosen:
                        m = chosen[-1]
                        if isinstance(m, dict):
                            content = m.get("content", "")
                            response = str(content) if not isinstance(content, list) else "\n".join([x.get("text","") for x in content if x.get("type")=="text"])
                else:
                    response = str(chosen)
                
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                    {"role": "assistant", "content": [{"type": "text", "text": response}]}
                ]
                return {"messages": messages, "images": []}

        # UltraInteract 처리
        if "ultrainteract" in dataset_name.lower():
            instruction = sample.get("instruction", "")
            response = sample.get("response", "")
            if not response and "trajectory" in sample:
                 response = str(sample["trajectory"])
            
            if instruction and response:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": instruction}]},
                    {"role": "assistant", "content": [{"type": "text", "text": response}]}
                ]
                return {"messages": messages, "images": []}

        
        # messages 형식 직접 지원
        if "messages" in sample and isinstance(sample["messages"], list):
            img = sample.get("image", [])
            if not isinstance(img, list):
                img = [img] if img is not None else []
            img = validate_image_data(img)
            
            messages = validate_messages(sample["messages"])
            return {"messages": messages, "images": img if img else []}
        
        # instruction-output 형식
        if "instruction" in sample and "output" in sample:
            img = sample.get("image", [])
            if not isinstance(img, list):
                img = [img] if img is not None else []
            img = validate_image_data(img)
            
            if img:
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["instruction"]}]},
                    {"role": "assistant", "content": [{"type": "text", "text": sample["output"]}]}
                ]
            else:
                # Input 필드가 있는 경우 처리
                user_text = sample["instruction"]
                if "input" in sample and sample["input"]:
                    user_text += f"\n\nInput:\n{sample['input']}"
                
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": user_text}]},
                    {"role": "assistant", "content": [{"type": "text", "text": sample["output"]}]}
                ]
            
            return {"messages": messages, "images": img if img else []}
    
    # VQA 형식 처리 (VQAv2) - 하위 호환성
    if "VQA" in dataset_name or "vqa" in dataset_name.lower():
        question = sample.get("question", "")
        answers = sample.get("answers", [])
        if isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict):
                answer = answers[0].get("answer", "")
            else:
                answer = str(answers[0])
        else:
            answer = sample.get("answer", "")
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        
        img = sample.get("image", [])
        if not isinstance(img, list):
            img = [img] if img is not None else []
        img = validate_image_data(img)
        if not img:
            return None
        
        return {"messages": messages, "images": img}
    
    # Flickr30k 형식 처리 - 하위 호환성
    if "flickr30k" in dataset_name.lower():
        captions = sample.get("caption", [])
        if not isinstance(captions, list):
            captions = [captions] if captions else []
        
        if not captions:
            return None
        
        caption = str(captions[0])
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]}
        ]
        
        img = sample.get("image", [])
        if not isinstance(img, list):
            img = [img] if img is not None else []
        img = validate_image_data(img)
        if not img:
            return None
        
        return {"messages": messages, "images": img}
    
    # CORD (OCR) 형식 처리
    if "cord" in dataset_name.lower():
        # CORD는 문서 이미지와 텍스트를 포함
        text = sample.get("text", "")
        if not text:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Extract and read the text from this document."}]},
            {"role": "assistant", "content": [{"type": "text", "text": text}]}
        ]
        
        img = sample.get("image", [])
        if not isinstance(img, list):
            img = [img] if img is not None else []
        img = validate_image_data(img)
        if not img:
            return None
        
        return {"messages": messages, "images": img}
    
    # FUNSD (OCR) 형식 처리
    if "funsd" in dataset_name.lower() or "layoutlmv3" in dataset_name.lower():
        words = sample.get("words", [])
        bboxes = sample.get("bboxes", [])
        
        # 단어들을 텍스트로 결합
        text = " ".join([str(word) for word in words]) if words else ""
        if not text:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Extract and read the text from this document."}]},
            {"role": "assistant", "content": [{"type": "text", "text": text}]}
        ]
        
        img = sample.get("image", [])
        if not isinstance(img, list):
            img = [img] if img is not None else []
        img = validate_image_data(img)
        if not img:
            return None
        
        return {"messages": messages, "images": img}
    
    # SciAlpaca / Camel-AI Science 형식 처리 (텍스트 전용)
    if "scialpaca" in dataset_name.lower() or "camel-ai/science" in dataset_name.lower():
        # 두 데이터셋 모두 instruction-output 형식을 따름
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        
        # Camel-AI Science는 message_1, message_2 형식을 사용할 수 있음
        if not instruction and "message_1" in sample and "message_2" in sample:
            instruction = sample["message_1"]
            output = sample["message_2"]

        if not instruction or not output:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": output}]}
        ]
        return {"messages": messages, "images": []}

    # SciTLDR 형식 처리
    if "scitldr" in dataset_name.lower():
        # source (abstract) -> target (summary)
        source_text = " ".join(sample.get("source", []))
        target_text = " ".join(sample.get("target", []))
        
        if not source_text or not target_text:
            return None

        instruction = f"Summarize the following scientific text in one or two sentences:\n\n{source_text}"
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
        ]
        return {"messages": messages, "images": []}

    # SROIE (OCR) 형식 처리
    if "sroie" in dataset_name.lower():
        text = sample.get("text", "")
        if not text:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Extract and read the text from this document."}]},
            {"role": "assistant", "content": [{"type": "text", "text": text}]}
        ]
        
        img = sample.get("image", [])
        if not isinstance(img, list):
            img = [img] if img is not None else []
        img = validate_image_data(img)
        if not img:
            return None
        
        return {"messages": messages, "images": img}
    
    # Evol-CodeAlpaca 형식 처리 (텍스트 전용)
    if "evol-codealpaca" in dataset_name.lower():
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        _input = sample.get("input", "")
        if not instruction or not output:
            return None
        user_text = instruction if not _input else f"{instruction}\n\nInput:\n{_input}"
        messages = [
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": output}]}
        ]
        return {"messages": messages, "images": []}
    
    # OCR-VQA 계열 (하위 호환성용 - 실제로는 process_ocr_vqa 사용)
    if "ocr-vqa" in dataset_name.lower() or "ocrvqa" in dataset_name.lower():
        # OCR-VQA는 questions (복수, 리스트)와 answers (복수, 리스트)를 사용
        # 하나의 이미지에 여러 질문-답변 쌍이 있으므로 첫 번째 쌍만 사용 (하위 호환성)
        questions = sample.get("questions", [])
        answers = sample.get("answers", [])
        
        # questions/answers가 리스트인지 확인
        if not isinstance(questions, (list, tuple)) or not isinstance(answers, (list, tuple)):
            return None
        
        if len(questions) == 0 or len(answers) == 0:
            return None
        
        # 첫 번째 질문-답변 쌍 사용
        question = str(questions[0]).strip()
        answer = str(answers[0]).strip()
        
        if not question or not answer:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        
        img = sample.get("image", None)
        if img is None:
            return None
        
        img_list = validate_image_data([img])
        if not img_list:
            return None
        
        return {"messages": messages, "images": img_list}
    
    # MetaMathQA 형식 처리 (학습용 수학 instruction)
    if "metamathqa" in dataset_name.lower() or "meta-math" in dataset_name.lower():
        query = sample.get("query", "")
        response = sample.get("response", "")
        if not query or not response:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": query}]},
            {"role": "assistant", "content": [{"type": "text", "text": response}]}
        ]
        return {"messages": messages, "images": []}
    
    # Math-Python-Reasoning 형식 처리 (학습용 수학 Python 추론)
    if "math-python-reasoning" in dataset_name.lower():
        # prompt/completion/system_prompt 형식 지원
        prompt = sample.get("prompt", "")
        completion = sample.get("completion", "")
        system_prompt = sample.get("system_prompt", "")
        
        # instruction/output 형식도 지원 (하위 호환성)
        if not prompt:
            prompt = sample.get("instruction", "")
        if not completion:
            completion = sample.get("output", "")
        
        if not prompt or not completion:
            return None
        
        messages = []
        
        # system_prompt가 있으면 system 메시지로 추가
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        
        messages.extend([
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": completion}]}
        ])
        
        return {"messages": messages, "images": []}
    
    # UltraInteract 형식 처리 (학습용 논리 추론 instruction)
    if "ultrainteract" in dataset_name.lower() or "ultra-interact" in dataset_name.lower():
        # UltraInteract는 다양한 형식이 있을 수 있음
        if "messages" in sample:
            messages = validate_messages(sample["messages"])
            return {"messages": messages, "images": []}
        elif "instruction" in sample and "output" in sample:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": sample["instruction"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": sample["output"]}]}
            ]
            return {"messages": messages, "images": []}
        elif "question" in sample and "answer" in sample:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": sample["question"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]}
            ]
            return {"messages": messages, "images": []}
    
    # UltraFeedback 형식 처리 (학습용 추론 instruction)
    if "ultrafeedback" in dataset_name.lower():
        # UltraFeedback은 다양한 형식이 있을 수 있음
        if "messages" in sample:
            messages = validate_messages(sample["messages"])
            return {"messages": messages, "images": []}
        elif "instruction" in sample and "output" in sample:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": sample["instruction"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": sample["output"]}]}
            ]
            return {"messages": messages, "images": []}
        elif "prompt" in sample and "response" in sample:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": sample["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": sample["response"]}]}
            ]
            return {"messages": messages, "images": []}
    
    # GSM8K 형식 처리 (텍스트 전용) - 벤치마크용, 하위 호환성
    if "gsm8k" in dataset_name.lower():
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        if not question or not answer:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        return {"messages": messages, "images": []}
    
    # MATH 형식 처리 (텍스트 전용) - 벤치마크용, 하위 호환성
    if "competition_math" in dataset_name.lower() or "hendrycks/math" in dataset_name.lower():
        problem = sample.get("problem", "")
        solution = sample.get("solution", "")
        if not problem or not solution:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": problem}]},
            {"role": "assistant", "content": [{"type": "text", "text": solution}]}
        ]
        return {"messages": messages, "images": []}
    
    # PubMedQA 형식 처리 (텍스트 전용) - 제거됨, 하위 호환성
    if "pubmed_qa" in dataset_name.lower():
        question = sample.get("question", "")
        long_answer = sample.get("long_answer", "")
        final_decision = sample.get("final_decision", "")
        
        if not question:
            return None
        
        answer_text = long_answer if long_answer else final_decision
        if not answer_text:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
        ]
        return {"messages": messages, "images": []}
    
    # CodeSearchNet 형식 처리 (텍스트 전용)
    if "code_search_net" in dataset_name.lower() or "codesearchnet" in dataset_name.lower():
        code = sample.get("code", "")
        docstring = sample.get("docstring", "")
        func_name = sample.get("func_name", "")
        
        if not code:
            return None
        
        # 코드와 설명을 instruction-output 형식으로 변환
        instruction = f"Write code for: {docstring}" if docstring else f"Write code for function: {func_name}" if func_name else "Write the following code:"
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": code}]}
        ]
        return {"messages": messages, "images": []}
    
    # CoNaLa 형식 처리 (텍스트 전용)
    if "conala" in dataset_name.lower():
        intent = sample.get("intent", "")
        snippet = sample.get("snippet", "")
        
        if not intent or not snippet:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": intent}]},
            {"role": "assistant", "content": [{"type": "text", "text": snippet}]}
        ]
        return {"messages": messages, "images": []}
    
    # rStar-Coder 형식 처리 (텍스트 전용)
    if "rstar-coder" in dataset_name.lower() or "rstar_coder" in dataset_name.lower():
        # 실제 keys 확인 및 처리
        sample_keys = list(sample.keys()) if isinstance(sample, dict) else []
        
        # rStar-Coder의 실제 컬럼: question, seed_question, seed_source, response, code
        question = sample.get("question", "")
        seed_question = sample.get("seed_question", "")
        seed_source = sample.get("seed_source", "")
        response = sample.get("response", "")
        code = sample.get("code", "")
        
        # 디버깅: 처음 몇 개 샘플의 keys 로깅
        if log_failure:
            logger.debug(f"   🔍 rStar-Coder 샘플 keys: {sample_keys}")
            logger.debug(f"   🔍 question: {bool(question)}, seed_question: {bool(seed_question)}, response: {bool(response)}, code: {bool(code)}")
        
        # question 또는 seed_question 중 하나는 있어야 함
        user_prompt = question if question else seed_question
        if not user_prompt:
            if log_failure:
                logger.warning(f"   ⚠️ rStar-Coder: question과 seed_question이 모두 비어있음. keys: {sample_keys}")
            return None
        
        # response와 code 중 하나는 있어야 함
        assistant_content_parts = []
        if response and response.strip():
            assistant_content_parts.append(response.strip())
        if code and code.strip():
            if assistant_content_parts:
                assistant_content_parts.append(f"\n\n```python\n{code.strip()}\n```")
            else:
                assistant_content_parts.append(code.strip())
        
        if not assistant_content_parts:
            if log_failure:
                logger.warning(f"   ⚠️ rStar-Coder: response와 code가 모두 비어있음. keys: {sample_keys}")
            return None
        
        assistant_text = "\n".join(assistant_content_parts)
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]
        return {"messages": messages, "images": []}
    
    # The Stack / StarCoderData 형식 처리 (텍스트 전용) - 하위 호환성
    if "the-stack" in dataset_name.lower() or "starcoderdata" in dataset_name.lower():
        content = sample.get("content", "")
        if not content:
            return None
        
        # 코드 데이터셋은 instruction-output 형식으로 변환
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Write the following code:"}]},
            {"role": "assistant", "content": [{"type": "text", "text": content}]}
        ]
        return {"messages": messages, "images": []}
    
    # LogiQA 형식 처리 (텍스트 전용) - 벤치마크용, 하위 호환성
    if "logiqa" in dataset_name.lower():
        question = sample.get("question", "")
        options = sample.get("options", [])
        answer = sample.get("answer", "")
        
        if not question or not answer:
            return None
        
        question_text = question
        if options:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            question_text = f"{question}\n\n{options_text}"
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": question_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        return {"messages": messages, "images": []}
    
    # ReClor 형식 처리 (텍스트 전용) - 벤치마크용, 하위 호환성
    if "reclor" in dataset_name.lower():
        question = sample.get("question", "")
        answers = sample.get("answers", [])
        label = sample.get("label", -1)
        
        if not question:
            return None
        
        question_text = question
        if answers and isinstance(answers, list):
            options_text = "\n".join([f"{chr(65+i)}. {ans}" for i, ans in enumerate(answers)])
            question_text = f"{question}\n\n{options_text}"
        
        answer_text = answers[label] if label >= 0 and label < len(answers) else (answers[0] if answers else "")
        if not answer_text:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": question_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
        ]
        return {"messages": messages, "images": []}
    
    # OpenOrca 형식 처리 (텍스트 전용)
    if "openorca" in dataset_name.lower() or "open-orca" in dataset_name.lower():
        # OpenOrca는 conversations 형식일 가능성이 높음
        if "conversations" in sample:
            messages = []
            for conv in sample["conversations"]:
                if isinstance(conv, dict):
                    role = conv.get("from", "user")
                    value = conv.get("value", "")
                    if value:
                        role_mapped = "user" if role in ["human", "user"] else "assistant"
                        messages.append({
                            "role": role_mapped,
                            "content": [{"type": "text", "text": value}]
                        })
            if messages:
                return {"messages": messages, "images": []}
        
        # instruction-output 형식
        if "instruction" in sample and "response" in sample:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": sample["instruction"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": sample["response"]}]}
            ]
            return {"messages": messages, "images": []}
    
    # smoltalk 데이터셋 처리
    if "smoltalk" in dataset_name.lower():
        if "messages" in sample and isinstance(sample["messages"], list):
            # messages가 이미 올바른 형식인 경우
            messages = validate_messages(sample["messages"])
            # content 필드 정규화
            messages = ensure_messages_text_strings(messages)
            img = sample.get("image", [])
            if not isinstance(img, list):
                img = [img] if img is not None else []
            img = validate_image_data(img)
            # smoltalk은 이미지가 없어도 텍스트만으로 처리 가능
            return {"messages": messages, "images": img if img else []}
    
    # simple_sft_dataset의 기본 변환 로직 사용
    from data.simple_sft_dataset import convert_sample_to_messages as base_convert
    result = base_convert(sample, dataset_name)
    
    # base_convert가 None을 반환하거나 이미지가 없는 경우, 텍스트 전용으로 처리 시도
    if result is None:
        # instruction-output 형식 재시도
        if "instruction" in sample and "output" in sample:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": sample["instruction"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": sample["output"]}]}
            ]
            messages = ensure_messages_text_strings(messages)
            return {"messages": messages, "images": []}
        
        # 모든 변환 시도 실패
        if log_failure:
            sample_keys = list(sample.keys()) if isinstance(sample, dict) else "N/A"
            sample_preview = str(sample)[:200] if isinstance(sample, dict) else str(sample)[:200]
            logger.debug(f"[{dataset_name}] 샘플 변환 실패 - 지원되지 않는 형식. 샘플 키: {sample_keys}, 미리보기: {sample_preview}...")
    
    # base_convert 결과 정규화
    if result:
        # messages 정규화
        if "messages" in result:
            result["messages"] = ensure_messages_text_strings(result["messages"])
        # 이미지가 없으면 빈 리스트로 설정
        if "images" in result:
            if not result["images"]:
                result["images"] = []
    
    return result

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 도메인별 데이터셋 설정
# 각 도메인별로 텍스트 전용 또는 멀티모달 데이터셋을 지정합니다.
# 텍스트 전용 데이터셋도 허용하며, 최종적으로 messages 형식으로 통합됩니다.
DOMAIN_DATASETS = {
    "math": [
        "meta-math/MetaMathQA",  # MetaMathQA: 수학 instruction 데이터셋 (학습용)
        "sdiazlor/math-python-reasoning-dataset",  # Math-Python-Reasoning: 수학 Python 추론 (학습용)
    ],
    "science": [
        "derek-thomas/ScienceQA",  # SciTLDR: 과학 논문 요약 (학습용)
        "dhmeltzer/ask-science-qg"
    ],
    "code": [
        "theblackcat102/evol-codealpaca-v1", # Evol-CodeAlpaca: 코드 instruction (학습용)
        "microsoft/rStar-Coder",  # rStar-Coder: 코드 데이터셋
    ],
    "puzzle": [
        "openbmb/UltraInteract_sft",  # UltraInteract_sft: 논리 추론 instruction 데이터셋 (학습용)
        "HuggingFaceH4/ultrafeedback_binarized",  # UltraFeedback 캐시 문제 해결 후 복원
    ],
    "vision": [
        "lmms-lab/LLaVA-OneVision-Data",  # LLaVA-OneVision-Data: 다양한 비전 태스크 (멀티모달)
        # "textvqa",  # TextVQA: 존재하지 않음, 대체 필요
    ],
    "ocr": [
        "howard-hou/OCR-VQA",  # OCR-VQA: OCR 질의응답 데이터셋
        "naver-clova-ix/cord-v2",  # CORD-v2: 영수증 OCR 데이터셋 (이미지 + ground_truth)
        # "allenai/olmOCR-mix-1025",  # olmOCR-mix: 별도 전처리 필요 (olmocr 툴킷), HF에서 직접 로드 시 이미지 미제공
    ],
    "chat": [
        "HuggingFaceTB/smoltalk",  # SmolTalk: 일반 채팅 (멀티모달 가능)
        "Open-Orca/OpenOrca",  # OpenOrca: 일반 대화 (텍스트 전용)
    ]
}

def _generate_cache_key(domain_configs: Dict[str, List[str]], max_samples_per_domain: int, 
                       test_size: float, use_streaming: bool, max_workers: int) -> str:
    """캐시 키 생성"""
    # 도메인 설정을 정렬하여 일관성 보장
    sorted_domains = sorted(domain_configs.items())
    cache_data = {
        "domain_configs": sorted_domains,
        "max_samples_per_domain": max_samples_per_domain,
        "test_size": test_size,
        "use_streaming": use_streaming,
        "max_workers": max_workers
    }
    cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
    cache_hash = hashlib.md5(cache_str.encode('utf-8')).hexdigest()
    return f"multi_domain_{cache_hash}"

def get_domain_from_config(config_name: str, dataset_name: str) -> Optional[str]:
    """
    Config 이름과 데이터셋 이름을 기반으로 도메인을 추론합니다.
    
    Args:
        config_name: 데이터셋 config 이름
        dataset_name: 데이터셋 이름
    
    Returns:
        추론된 도메인 이름 또는 None
    """
    config_lower = config_name.lower()
    dataset_lower = dataset_name.lower()
    
    # 키워드 기반 도메인 매칭 (우선순위 순)
    math_keywords = ["math", "mathematical", "algebra", "geometry", "calculus", "arithmetic", "equation"]
    science_keywords = ["science", "physics", "chemistry", "biology", "scientific", "astronomy", "geology"]
    code_keywords = ["code", "programming", "python", "javascript", "coding", "software", "algorithm", "function"]
    puzzle_keywords = ["puzzle", "logic", "reasoning", "riddle", "brain", "challenge", "problem"]
    vision_keywords = ["vision", "visual", "image", "photo", "picture", "camera", "see", "look"]
    ocr_keywords = ["ocr", "text", "document", "scan", "recognition", "read", "extract", "textual"]
    
    if any(keyword in config_lower for keyword in math_keywords):
        return "math"
    elif any(keyword in config_lower for keyword in science_keywords):
        return "science"
    elif any(keyword in config_lower for keyword in code_keywords):
        return "code"
    elif any(keyword in config_lower for keyword in puzzle_keywords):
        return "puzzle"
    elif any(keyword in config_lower for keyword in vision_keywords):
        return "vision"
    elif any(keyword in config_lower for keyword in ocr_keywords):
        return "ocr"
    
    # 데이터셋 이름 기반 매칭
    if any(keyword in dataset_lower for keyword in math_keywords):
        return "math"
    elif any(keyword in dataset_lower for keyword in science_keywords):
        return "science"
    elif any(keyword in dataset_lower for keyword in code_keywords):
        return "code"
    elif any(keyword in dataset_lower for keyword in puzzle_keywords):
        return "puzzle"
    elif any(keyword in dataset_lower for keyword in vision_keywords):
        return "vision"
    elif any(keyword in dataset_lower for keyword in ocr_keywords):
        return "ocr"
    
    return None

def _process_dataset_config_split(
    domain: str,
    dataset_name: str,
    config: str,
    train_split: str,
    test_split: Optional[str],
    train_path: str,
    test_path: str,
    image_counter_lock: threading.Lock,
    shared_counters: Dict[str, Any],
    images_dir: str,
    samples_per_config: int,
    test_size: float,
    use_streaming: bool,
    domain_processed_lock: threading.Lock,
    domain_processed_dict: Dict[str, int],
    max_samples_per_domain: int
) -> Dict[str, Any]:
    """
    단일 데이터셋의 단일 config와 split을 처리하는 함수 (병렬 처리용)
    데이터셋별 전용 프로세서를 사용하여 messages 형식으로 변환
    """
    result = {
        "train_count": 0,
        "test_count": 0,
    }
    
    try:
        # 데이터셋에 맞는 프로세서 선택
        processor = get_processor_for_dataset(dataset_name)
        logger.debug(f"   🎯 [{dataset_name}] 프로세서 선택: {processor.__name__}")
        
        # Train split 처리
        try:
            logger.debug(f"   🔍 [{dataset_name}] Config {config} Train split {train_split} 로딩...")
            

            # Load Args 구성 (복구됨)
            load_kwargs = {
                "path": dataset_name,
                "split": train_split,
                "streaming": use_streaming,
                "trust_remote_code": True,
            }
            if config != "default":
                load_kwargs["name"] = config

            # [Dynamic Load] 메타데이터 오류 우회 및 동적 Split/File 매핑
            broken_metadata_datasets = ["lmms-lab/LLaVA-OneVision-Data", "howard-hou/OCR-VQA", "HuggingFaceH4/ultrafeedback_binarized", "HuggingFaceTB/smoltalk"]
            
            if any(broken in dataset_name for broken in broken_metadata_datasets):
                logger.warning(f"   🛡️ [{dataset_name}] 메타데이터 오류 회피 -> Parquet 동적 로딩 시작")
                try:
                    # 1. 빌더 로드 (메타데이터 확보)
                    builder = load_dataset_builder(dataset_name, name=config if config != "default" else None, trust_remote_code=True)
                    
                    # 2. 사용 가능한 Split 확인 및 동적 매핑
                    available_splits = list(builder.info.splits.keys()) if builder.info.splits else []
                    target_split = train_split
                    
                    if train_split not in available_splits:
                        candidates = [s for s in available_splits if "train" in s]
                        if candidates:
                            target_split = candidates[0]
                            logger.warning(f"   ⚠️ Split '{train_split}' 부재 -> '{target_split}' 자동 매핑")
                        else:
                            logger.error(f"   ❌ Split 매핑 실패. 요청: {train_split}, 가용: {available_splits}")
                            raise ValueError(f"Split '{train_split}'을 찾을 수 없습니다.")
                    
                    # 3. 파일 매핑 (Split 이름 -> Parquet 파일 경로)
                    data_files = builder.config.data_files
                    files = None
                    if isinstance(data_files, dict):
                        files = data_files.get(target_split)
                        if not files:
                            if "train" in data_files:
                                files = data_files["train"]
                            else:
                                for k in data_files.keys():
                                    if target_split in k or k in target_split:
                                        files = data_files.get(k)
                                        break
                    else:
                        files = data_files
                    
                    if not files:
                        raise ValueError(f"Files not found for split {target_split}")

                    # 4. Parquet 엔진으로 로드
                    train_dataset = load_dataset(
                        "parquet", 
                        data_files={target_split: files}, 
                        split=target_split, 
                        streaming=use_streaming
                    )
                    logger.debug(f"   ✅ Parquet 동적 로딩 성공: Split '{target_split}'")
                    
                except Exception as pq_e:
                    logger.error(f"   ❌ Parquet 로딩 중 치명적 오류: {pq_e}")
                    raise pq_e 
            else:
                # 일반 데이터셋 로딩
                train_dataset = load_dataset(**load_kwargs)

            
            # 데이터셋 정보 확인
            dataset_size = None
            if not use_streaming:
                dataset_size = len(train_dataset) if hasattr(train_dataset, '__len__') else None
                if dataset_size == 0:
                    error_msg = f"❌ [{dataset_name}] Config {config} Train split {train_split}이 비어있습니다."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            
            train_samples_per_config = samples_per_config
            if test_split:
                train_samples_per_config = int(samples_per_config * (1 - test_size))
            
            # 도메인별 처리량 체크
            with domain_processed_lock:
                current_domain_processed = domain_processed_dict.get(domain, 0)
                remaining_domain_capacity = max_samples_per_domain - current_domain_processed
                effective_max_samples = min(train_samples_per_config, remaining_domain_capacity)
            
            if effective_max_samples <= 0:
                logger.debug(f"   ⚠️ [{dataset_name}] Config {config}: 도메인 처리량 한계 도달, 건너뜀")
                return result
            
            # ============================================================
            # 데이터셋별 프로세서 호출 (전체 데이터셋 처리)
            # ============================================================
            # 디버깅이 필요한 데이터셋은 항상 상세 로그 출력
            log_detail = any(keyword in dataset_name.lower() for keyword in ["rstar", "ask-science"])
            converted_results = processor(train_dataset, dataset_name, effective_max_samples, log_detail=log_detail)
            
            logger.debug(f"   ✅ [{dataset_name}] Config {config} 프로세서 완료: {len(converted_results)}개 샘플 변환됨")
            
            # 변환 결과가 없으면 에러
            if not converted_results:
                error_msg = f"❌ [{dataset_name}] Config {config}: 프로세서에서 변환된 샘플이 없습니다."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # ============================================================
            # 변환된 결과를 파일에 저장
            # ============================================================
            train_processed = 0
            for idx, converted in enumerate(converted_results):
                try:
                    # messages 텍스트 문자열 보장
                    if "messages" in converted:
                        converted["messages"] = ensure_messages_text_strings(converted["messages"])
                    
                    # 이미지 처리 (PIL Image 객체를 파일로 저장)
                    image_paths = []
                    if "images" in converted and converted["images"]:
                        for img_obj in converted["images"]:
                            if isinstance(img_obj, Image.Image):
                                try:
                                    with image_counter_lock:
                                        current_counter = shared_counters["image_counter"]
                                        img_path = os.path.join(images_dir, f"{current_counter}.png")
                                        img_obj.save(img_path, "PNG")
                                        image_paths.append(img_path)
                                        shared_counters["image_counter"] += 1
                                except Exception as img_e:
                                    error_msg = f"❌ [{dataset_name}] 이미지 저장 실패: {img_e}"
                                    logger.error(error_msg)
                                    raise RuntimeError(error_msg) from img_e
                    
                    # 최종 데이터 구성
                    converted["images"] = image_paths
                    converted["domain"] = domain
                    
                    # VLM 형식으로 변환 (모든 데이터를 VLM 형식으로 통일)
                    converted = ensure_vlm_format(converted)
                    
                    # JSON 직렬화 가능한 형태로 정리
                    converted = sanitize_sample_for_json(converted)
                    
                    # 파일 쓰기 (도메인별 파일에 append)
                    try:
                        json_str = json.dumps(converted, ensure_ascii=False)
                        with open(train_path, "a", encoding="utf-8") as f:
                            f.write(json_str + "\n")
                        
                        # 카운터 업데이트 (thread-safe)
                        with domain_processed_lock:
                            shared_counters["domain_counts"][domain]["train"] += 1
                            shared_counters["total_processed"] += 1
                            domain_processed_dict[domain] = domain_processed_dict.get(domain, 0) + 1
                        
                        result["train_count"] += 1
                        train_processed += 1
                        
                    except (TypeError, ValueError) as json_e:
                        error_msg = f"❌ [{dataset_name}] JSON 직렬화 실패: {json_e}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg) from json_e
                
                except Exception as sample_e:
                    logger.error(f"❌ [{dataset_name}] Train 샘플 {idx} 저장 실패: {sample_e}")
                    # 저장 실패 시에도 전체 프로세스 중단
                    raise RuntimeError(f"Train 샘플 저장 실패") from sample_e
            
            logger.debug(f"   ✅ [{dataset_name}] Config {config} Train split 저장 완료: {train_processed}개")
            
            del train_dataset
            gc.collect()
            
        except Exception as e:
            error_msg = f"❌ [{dataset_name}] Config {config} Train split {train_split} 로드 실패: {e}"
            logger.error(error_msg)
            traceback.print_exc()
            raise RuntimeError(error_msg) from e
        
        # Test split 처리 (있는 경우)
        if test_split:
            try:
                logger.debug(f"   🔍 [{dataset_name}] Config {config} Test split {test_split} 로딩...")
                
                # Test Load Args 구성
                # Test Load Args 구성
                test_load_kwargs = {
                    "path": dataset_name,
                    "split": test_split,
                    "streaming": use_streaming,
                    "trust_remote_code": True,
                }
                if config != "default":
                    test_load_kwargs["name"] = config
                
                # [Dynamic Load] Test Parquet 직접 로딩 (동적 매핑)
                if any(broken in dataset_name for broken in broken_metadata_datasets):
                    logger.warning(f"   🛡️ [{dataset_name}] (Test) 메타데이터 오류 회피 -> Parquet 동적 로딩 시작")
                    try:
                        builder = load_dataset_builder(dataset_name, name=config if config != "default" else None, trust_remote_code=True)
                        
                        # 1. Split 매핑
                        available_splits = list(builder.info.splits.keys()) if builder.info.splits else []
                        target_split = test_split
                        
                        if test_split not in available_splits:
                            # 'test'나 'val'이 포함된 Split 검색
                            candidates = [s for s in available_splits if "test" in s or "val" in s]
                            if candidates:
                                target_split = candidates[0]
                                logger.warning(f"   ⚠️ (Test) Split '{test_split}' 부재 -> '{target_split}' 자동 매핑")
                            else:
                                logger.error(f"   ❌ (Test) Split 매핑 실패. 요청: {test_split}, 가용: {available_splits}")
                                raise ValueError(f"Split '{test_split}'을 찾을 수 없습니다.")

                        # 2. 파일 매핑
                        data_files = builder.config.data_files
                        files = None
                        if isinstance(data_files, dict):
                             files = data_files.get(target_split)
                             if not files:
                                 # 키 이름 유연 검색
                                 for k in data_files.keys():
                                     if target_split in k or k in target_split:
                                         files = data_files.get(k)
                                         break
                        else:
                             files = data_files

                        if not files:
                            raise ValueError(f"Files not found for split {target_split}")

                        test_dataset = load_dataset(
                            "parquet", 
                            data_files={target_split: files}, 
                            split=target_split, 
                            streaming=use_streaming
                        )
                        logger.debug(f"   ✅ (Test) Parquet 동적 로딩 성공: Split '{target_split}'")
                    except Exception as pq_e:
                        logger.error(f"   ❌ Parquet 직접 로딩 실패 (Test): {pq_e}")
                        raise pq_e # Fallback 금지
                else:
                    test_dataset = load_dataset(**test_load_kwargs)
                
                test_samples_per_config = int(samples_per_config * test_size)
                
                # 프로세서 호출 (전체 데이터셋 처리)
                log_detail = "rstar" in dataset_name.lower()
                test_converted_results = processor(test_dataset, dataset_name, test_samples_per_config, log_detail=log_detail)
                
                logger.debug(f"   ✅ [{dataset_name}] Config {config} Test 프로세서 완료: {len(test_converted_results)}개 샘플 변환됨")
                
                # 변환된 결과를 파일에 저장
                test_processed = 0
                for idx, converted in enumerate(test_converted_results):
                    try:
                        # messages 텍스트 문자열 보장
                        if "messages" in converted:
                            converted["messages"] = ensure_messages_text_strings(converted["messages"])
                        
                        # 이미지 처리
                        image_paths = []
                        if "images" in converted and converted["images"]:
                            for img_obj in converted["images"]:
                                if isinstance(img_obj, Image.Image):
                                    try:
                                        with image_counter_lock:
                                            current_counter = shared_counters["image_counter"]
                                            img_path = os.path.join(images_dir, f"{current_counter}.png")
                                            img_obj.save(img_path, "PNG")
                                            image_paths.append(img_path)
                                            shared_counters["image_counter"] += 1
                                    except Exception as img_e:
                                        error_msg = f"❌ [{dataset_name}] 이미지 저장 실패: {img_e}"
                                        logger.error(error_msg)
                                        raise RuntimeError(error_msg) from img_e
                        
                        # 최종 데이터 구성
                        converted["images"] = image_paths
                        converted["domain"] = domain
                        
                        # VLM 형식으로 변환 (모든 데이터를 VLM 형식으로 통일)
                        converted = ensure_vlm_format(converted)
                        
                        # JSON 직렬화 가능한 형태로 정리
                        converted = sanitize_sample_for_json(converted)
                        
                        # 파일 쓰기 (도메인별 파일에 append)
                        try:
                            json_str = json.dumps(converted, ensure_ascii=False)
                            with open(test_path, "a", encoding="utf-8") as f:
                                f.write(json_str + "\n")
                            
                            # 카운터 업데이트 (thread-safe)
                            with domain_processed_lock:
                                shared_counters["domain_counts"][domain]["test"] += 1
                                shared_counters["total_processed"] += 1
                            
                            result["test_count"] += 1
                            test_processed += 1
                            
                        except (TypeError, ValueError) as json_e:
                            error_msg = f"❌ [{dataset_name}] JSON 직렬화 실패: {json_e}"
                            logger.error(error_msg)
                            raise RuntimeError(error_msg) from json_e
                    
                    except Exception as sample_e:
                        logger.error(f"❌ [{dataset_name}] Test 샘플 {idx} 저장 실패: {sample_e}")
                        raise RuntimeError(f"Test 샘플 저장 실패") from sample_e
                
                logger.debug(f"   ✅ [{dataset_name}] Config {config} Test split 저장 완료: {test_processed}개")
                
                del test_dataset
                gc.collect()
                
            except Exception as e:
                error_msg = f"❌ [{dataset_name}] Config {config} Test split {test_split} 처리 실패: {e}"
                logger.error(error_msg)
                traceback.print_exc()
                raise RuntimeError(error_msg) from e
    
    except Exception as e:
        error_msg = f"❌ [{dataset_name}] Config {config} 처리 실패: {e}"
        logger.error(error_msg)
        traceback.print_exc()
        raise RuntimeError(error_msg) from e
    
    # 최소한 하나의 샘플은 처리되어야 함
    # (프로세서에서 변환 실패 시 이미 RuntimeError를 발생시키므로, 여기는 최종 체크만)
    if result["train_count"] == 0 and result["test_count"] == 0:
        error_msg = f"❌ [{dataset_name}] Config {config}: 처리된 샘플이 없습니다."
        logger.error(error_msg)
        logger.error(f"   📋 확인 사항:")
        logger.error(f"      - Config: {config}")
        logger.error(f"      - Train split: {train_split}")
        logger.error(f"      - Test split: {test_split}")
        logger.error(f"      - Streaming: {use_streaming}")
        logger.error(f"      - Samples per config: {samples_per_config}")
        
        raise RuntimeError(error_msg)
    
    return result

def _process_domain_datasets(
    domain: str,
    dataset_names: List[str],
    temp_dir: str,
    image_counter_lock: threading.Lock,
    shared_counters: Dict[str, Any],
    images_dir: str,
    max_samples_per_domain: int,
    test_size: float,
    use_streaming: bool,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    단일 도메인의 데이터셋들을 처리하는 함수 (병렬 처리용)
    각 데이터셋, config, split을 순차 처리 (도메인별 병렬은 상위 레벨에서)
    """
    # 도메인별 임시 파일 생성
    domain_train_path = os.path.join(temp_dir, f"{domain}_train.jsonl")
    domain_test_path = os.path.join(temp_dir, f"{domain}_test.jsonl")
    
    # 도메인별 처리 통계
    domain_stats = {
        "total_processed": 0,  # train_count + test_count 합계
    }
    
    # ScienceQA 미러 중복 방지 플래그
    scienceqa_taken = False
    
    # 도메인별 처리량 추적 (thread-safe)
    domain_processed_lock = threading.Lock()
    domain_processed_dict = {domain: 0}
    
    # 모든 데이터셋/config/split 작업 수집
    tasks = []
    
    for dataset_name in dataset_names:
        try:
            logger.debug(f"   📋 {domain} 도메인 - 데이터셋: {dataset_name}")
            
            # 데이터셋 존재 확인
            if not dataset_exists(dataset_name):
                error_msg = f"❌ [{domain}] 데이터셋이 존재하지 않거나 접근할 수 없습니다: {dataset_name}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 데이터셋의 config 목록 가져오기
            try:
                available_configs = get_dataset_config_names(dataset_name)
                if not available_configs:
                    available_configs = ["default"]
            except Exception as e:
                error_msg = f"❌ [{domain}] 데이터셋 {dataset_name}의 Config 목록 가져오기 실패: {e}"
                logger.error(error_msg)
                traceback.print_exc()
                raise RuntimeError(error_msg) from e
            
            # LLaVA-OneVision-Data는 onevision 서브셋만 사용
            if "llava-onevision" in dataset_name.lower() or "llava-onevision-data" in dataset_name.lower():
                filtered = [c for c in available_configs if "onevision" in str(c).lower()]
                if filtered:
                    available_configs = filtered
                else:
                    available_configs = available_configs[:5]
            
            # rStar-Coder에 한해서만 RL 관련 config 제외, SFT용만 사용
            if "rstar-coder" in dataset_name.lower() or "rstar_coder" in dataset_name.lower():
                rl_keywords = ["rl", "reinforcement", "synthetic_rl", "dpo", "ppo", "reward"]
                test_keywords = ["test_case", "test"]  # 테스트 케이스 생성용은 제외
                original_count = len(available_configs)
                
                # SFT 학습용 config만 사용 (test_case 제외)
                sft_configs = [
                    c for c in available_configs 
                    if str(c).lower() in ["seed_sft", "synthetic_sft"] 
                    and not any(test_kw in str(c).lower() for test_kw in test_keywords)
                ]
                if sft_configs:
                    available_configs = sft_configs
                else:
                    # RL 및 test 관련 config 제외
                    filtered_configs = [
                        c for c in available_configs 
                        if not any(keyword in str(c).lower() for keyword in rl_keywords)
                        and not any(test_kw in str(c).lower() for test_kw in test_keywords)
                    ]
                    if filtered_configs:
                        available_configs = filtered_configs
                    else:
                        # 필터링 후 남은 config가 없으면 오류
                        error_msg = f"❌ [{domain}] 데이터셋 {dataset_name}: 모든 config가 RL 또는 test 관련입니다. SFT 학습용 config가 없습니다."
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
            
            # ScienceQA 미러 중복 방지
            if domain == "science" and ("scienceqa" in dataset_name.lower()):
                if scienceqa_taken:
                    continue
                scienceqa_taken = True
            
            # Config별로 샘플 수 계산
            samples_per_config = max(1, max_samples_per_domain // max(len(available_configs), 1))
            
            # 각 config에 대해 작업 생성
            for config in available_configs:
                try:
                    # 사용 가능한 split 확인
                    try:
                        if config == "default":
                            available_splits = get_dataset_split_names(dataset_name)
                        else:
                            available_splits = get_dataset_split_names(dataset_name, config_name=config)
                        
                        # Train split 선택: train_sft > train
                        train_split = None
                        if "train_sft" in available_splits:
                            train_split = "train_sft"
                        elif "train" in available_splits:
                            train_split = "train"
                        else:
                            error_msg = f"❌ [{domain}] 데이터셋 {dataset_name} Config {config}에 train 또는 train_sft split이 없습니다."
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                        
                        # Test split 선택: test_sft > test
                        test_split = None
                        if "test_sft" in available_splits:
                            test_split = "test_sft"
                        elif "test" in available_splits:
                            test_split = "test"
                    except Exception as e:
                        error_msg = f"❌ [{domain}] 데이터셋 {dataset_name} Config {config}의 Split 목록 가져오기 실패: {e}"
                        logger.error(error_msg)
                        traceback.print_exc()
                        raise RuntimeError(error_msg) from e
                    
                    # 작업 추가
                    tasks.append({
                        "domain": domain,
                        "dataset_name": dataset_name,
                        "config": config,
                        "train_split": train_split,
                        "test_split": test_split,
                        "samples_per_config": samples_per_config
                    })
                
                except Exception as e:
                    error_msg = f"❌ [{domain}] 데이터셋 {dataset_name} Config {config} 준비 실패: {e}"
                    logger.error(error_msg)
                    traceback.print_exc()
                    raise RuntimeError(error_msg) from e
        
        except Exception as e:
            error_msg = f"❌ [{domain}] 데이터셋 {dataset_name} 준비 실패: {e}"
            logger.error(error_msg)
            traceback.print_exc()
            raise RuntimeError(error_msg) from e
    
    # 모든 작업을 순차로 실행 (도메인별 병렬 처리는 상위 레벨에서 수행)
    if not tasks:
        error_msg = f"❌ [{domain}] 도메인에 처리할 작업이 없습니다."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.debug(f"   📋 {domain} 도메인: {len(tasks)}개 작업을 순차 처리합니다.")
    
    # 도메인 내부는 순차 처리 (도메인별 병렬은 상위 레벨에서 처리)
    for task in tasks:
        try:
            result = _process_dataset_config_split(
                domain=task["domain"],
                dataset_name=task["dataset_name"],
                config=task["config"],
                train_split=task["train_split"],
                test_split=task["test_split"],
                train_path=domain_train_path,
                test_path=domain_test_path,
                image_counter_lock=image_counter_lock,
                shared_counters=shared_counters,
                images_dir=images_dir,
                samples_per_config=task["samples_per_config"],
                test_size=test_size,
                use_streaming=use_streaming,
                domain_processed_lock=domain_processed_lock,
                domain_processed_dict=domain_processed_dict,
                max_samples_per_domain=max_samples_per_domain
            )
            # train_count + test_count 합계
            total_count = result.get("train_count", 0) + result.get("test_count", 0)
            domain_stats["total_processed"] += total_count
        except Exception as e:
            error_msg = f"❌ [{domain}] 데이터셋 {task['dataset_name']} Config {task['config']} 처리 실패: {e}"
            logger.error(error_msg)
            logger.error("🛑 오류 발생으로 인해 모든 작업을 취소하고 프로세스를 중단합니다.")
            traceback.print_exc()
            raise RuntimeError(error_msg) from e
    
    # 도메인별 처리 통계 로깅 및 검증
    if domain_stats["total_processed"] == 0:
        error_msg = f"❌ [{domain}] 도메인: 처리된 샘플이 없습니다."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.debug(f"   📊 {domain} 도메인 처리 통계: 총 {domain_stats['total_processed']}개 샘플 처리 완료")
    
    return {
        "domain": domain,
        "domain_stats": domain_stats,
        "domain_processed": domain_processed_dict.get(domain, 0),
        "train_path": domain_train_path,
        "test_path": domain_test_path
    }

def get_multi_domain_sft_dataset(
    domain_configs: Optional[Dict[str, List[str]]] = None,
    tokenizer=None,
    max_length: int = 2048,
    max_samples_per_domain: int = 200,
    test_size: float = 0.1,
    use_streaming: bool = True,
    chunk_size: int = 1000,
    max_workers: int = 4,
    use_cache: bool = True,
    allow_text_only: bool = False
):
    """
    멀티 도메인 SFT 데이터셋을 로드합니다.
    
    Args:
        domain_configs: 도메인별 데이터셋 설정 딕셔너리
            예: {"math": ["dataset1", "dataset2"], "science": ["dataset3"]}
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
        max_samples_per_domain: 도메인당 최대 샘플 수
        test_size: 테스트 세트 비율
        use_streaming: 스트리밍 모드 사용 여부
        chunk_size: 청크 크기
        max_workers: 병렬 처리 워커 수
        use_cache: 캐시 사용 여부 (기본값: True)
    
    Returns:
        DatasetDict with train/test splits, 각 샘플에 'domain' 필드 포함
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    if domain_configs is None:
        domain_configs = DOMAIN_DATASETS
    
    # 캐시 키 생성
    cache_key = _generate_cache_key(domain_configs, max_samples_per_domain, test_size, use_streaming, max_workers)
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
            logger.debug(f"💾 캐시된 데이터셋 발견: {cache_key}")
            logger.debug(f"   - 캐시 디렉토리: {cache_dir}")
            
            # 메타데이터 확인
            if os.path.exists(cache_meta_path):
                try:
                    with open(cache_meta_path, "r", encoding="utf-8") as f:
                        cache_meta = json.load(f)
                        logger.debug(f"   - 캐시 생성 시간: {cache_meta.get('created_at', 'N/A')}")
                        logger.debug(f"   - Train 샘플 수: {cache_meta.get('train_count', 'N/A')}")
                        logger.debug(f"   - Test 샘플 수: {cache_meta.get('test_count', 'N/A')}")
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
                
                logger.debug("🧠 캐시된 JSONL 파일로부터 데이터셋 로딩 (Memory Mapping 활성화)...")
                
                # Using load_dataset("json") with explicit schema for memory-mapped loading
                from datasets import load_dataset as hf_load_dataset
                dataset_dict = hf_load_dataset("json", data_files=data_files, features=SFT_JSON_FEATURES)
                
                # CRITICAL RAM FIX: Slice dataset IMMEDIATELY after loading if it exceeds requested size
                # This prevents holding 3M+ Python objects in RAM during the .map() phase
                total_max_samples = max_samples_per_domain * len(domain_configs)
                for split in dataset_dict:
                    current_size = len(dataset_dict[split])
                    if current_size > total_max_samples * 2: # Keep some headroom for filtering
                        logger.debug(f"   ✂️  Slicing {split} dataset from {current_size} to {total_max_samples * 2} to save RAM")
                        dataset_dict[split] = dataset_dict[split].select(range(total_max_samples * 2))
                
                logger.debug(f"   ✅ 데이터셋 로드 완료 (Memory Mapped): {dataset_dict}")
                
                # 이미지 경로 처리 및 캐스팅 (Memory Efficient)
                logger.debug("🖼️ 이미지 경로 처리 및 DatasetImage 캐스팅 (num_proc 활용)...")
                
                # Setup features for casting
                for split in dataset_dict:
                    current_features = dataset_dict[split].features
                    new_features = current_features.copy()
                    new_features['images'] = Sequence(DatasetImage(decode=True))
                    
                    # Use map with num_proc for faster execution
                    dataset_dict[split] = dataset_dict[split].map(
                        _preprocess_images_for_mapping,
                        fn_kwargs={"cache_images_dir": cache_images_dir},
                        batched=False,
                        num_proc=min(max_workers, 8),
                        features=new_features,
                        desc=f"Processing {split} images"
                    )
                
                logger.debug("✅ 캐시된 데이터셋 로드 완료")
                return dataset_dict
                
            except Exception as e:
                logger.warning(f"   ⚠️ 캐시 로드 실패, 재처리합니다: {e}")
                traceback.print_exc()
                # 캐시 로드 실패 시 기존 로직으로 진행
    
    # 캐시가 없거나 사용하지 않는 경우 기존 처리 로직
    logger.debug(f"📦 멀티 도메인 데이터셋 로딩 시작 (캐시 없음)")
    logger.debug(f"   - 도메인 수: {len(domain_configs)}개")
    logger.debug(f"   - 도메인당 최대 샘플: {max_samples_per_domain}개")
    logger.debug(f"   - 총 최대 샘플: {max_samples_per_domain * len(domain_configs)}개")
    logger.debug(f"   - streaming: {use_streaming}")
    logger.debug(f"   - 병렬 처리: {max_workers}개 워커")
    
    log_memory_usage("멀티 도메인 데이터셋 로딩 시작")
    
    # 캐시 디렉토리 사용 (기존 temp_dir 대신)
    os.makedirs(cache_dir, exist_ok=True)
    images_dir = cache_images_dir
    os.makedirs(images_dir, exist_ok=True)

    try:
        train_jsonl_path = cache_train_path
        test_jsonl_path = cache_test_path
        
        # 이미지 카운터 lock (병렬 처리 시 thread-safe)
        image_counter_lock = threading.Lock()
        
        # 공유 카운터 (thread-safe)
        shared_counters = {
            "total_processed": 0,
            "image_counter": 0,
            "domain_counts": defaultdict(lambda: {"train": 0, "test": 0})
        }

        # 각 도메인별로 처리 (병렬화)
        domain_pbar = tqdm(domain_configs.items(), desc="도메인 처리", unit="domain")
        
        # 도메인별 병렬 처리
        executor = ThreadPoolExecutor(max_workers=max_workers)
        future_to_domain = {}
        domain_file_paths = {}  # 도메인별 파일 경로 저장
        
        try:
            for domain, dataset_names in domain_configs.items():
                if not dataset_names:
                    error_msg = f"❌ {domain} 도메인에 데이터셋이 지정되지 않았습니다."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # 각 도메인을 병렬로 처리
                future = executor.submit(
                    _process_domain_datasets,
                    domain=domain,
                    dataset_names=dataset_names,
                    temp_dir=cache_dir,
                    image_counter_lock=image_counter_lock,
                    shared_counters=shared_counters,
                    images_dir=images_dir,
                    max_samples_per_domain=max_samples_per_domain,
                    test_size=test_size,
                    use_streaming=use_streaming,
                    max_workers=max_workers
                )
                future_to_domain[future] = domain
            
            # 완료된 작업 처리
            for future in as_completed(future_to_domain):
                domain = future_to_domain[future]
                try:
                    result = future.result()
                    
                    if result is None:
                        logger.warning(f"⏩ {domain} 도메인 처리 결과가 없습니다 (Skipped).")
                        domain_pbar.update(1)
                        continue
                        
                    domain_file_paths[domain] = {
                        "train": result["train_path"],
                        "test": result["test_path"]
                    }
                    domain_pbar.update(1)
                    domain_pbar.set_description(f"도메인: {domain} 완료")
                except Exception as e:
                    error_msg = f"❌ {domain} 도메인 처리 실패: {e}"
                    logger.error(error_msg)
                    logger.error("🛑 오류 발생으로 인해 모든 작업을 취소하고 프로세스를 중단합니다.")
                    traceback.print_exc()
                    domain_pbar.update(1)
                    
                    # 모든 미완료 작업 취소
                    for f in future_to_domain:
                        if not f.done():
                            f.cancel()
                    
                    os._exit(1)
        finally:
            # 정상 완료 시에만 정상 종료
            try:
                executor.shutdown(wait=True)
            except:
                pass  # 이미 종료된 경우 무시
        
        # 최종 카운터 업데이트
        domain_counts = shared_counters["domain_counts"]
        total_processed = shared_counters["total_processed"]
        image_counter = shared_counters["image_counter"]
        
        # 병렬 처리 완료
        domain_pbar.close()
        
        # 도메인별 파일을 최종 파일로 합치기
        logger.debug("🔄 도메인별 파일을 최종 파일로 합치는 중...")
        with open(train_jsonl_path, "w", encoding="utf-8") as train_f, \
             open(test_jsonl_path, "w", encoding="utf-8") as test_f:
            
            for domain in domain_configs.keys():
                if domain not in domain_file_paths:
                    continue
                
                # Train 파일 합치기
                domain_train_path = domain_file_paths[domain]["train"]
                if os.path.exists(domain_train_path):
                    with open(domain_train_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                train_f.write(line)
                
                # Test 파일 합치기
                domain_test_path = domain_file_paths[domain]["test"]
                if os.path.exists(domain_test_path):
                    with open(domain_test_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                test_f.write(line)

        # 도메인별 통계 출력 및 검증
        if not domain_counts:
            error_msg = "❌ 처리된 도메인이 없습니다."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        total_train_samples = sum(c["train"] for c in domain_counts.values())
        total_test_samples = sum(c["test"] for c in domain_counts.values())
        
        if total_train_samples == 0 and total_test_samples == 0:
            error_msg = "❌ 처리된 샘플이 없습니다."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.debug("📊 도메인별 샘플 통계 (균등화 전):")
        for domain, counts in domain_counts.items():
            logger.debug(f"   - {domain}: Train {counts['train']}개, Test {counts['test']}개")
        
        # 도메인별 샘플 수 균등화
        # 각 도메인에서 동일한 수의 샘플을 사용하도록 조정
        balanced_train_count = 0
        balanced_test_count = 0
        
        if domain_counts:
            min_train = min([c["train"] for c in domain_counts.values()] + [max_samples_per_domain])
            min_test = min([c["test"] for c in domain_counts.values()] + [int(max_samples_per_domain * test_size)])
            
            logger.debug(f"⚖️ 도메인별 샘플 수 균등화:")
            logger.debug(f"   - 최소 Train 샘플 수: {min_train}개")
            logger.debug(f"   - 최소 Test 샘플 수: {min_test}개")
            
            # JSONL 파일을 다시 읽어서 균등화
            if min_train > 0 or min_test > 0:
                logger.debug("🔄 샘플 수 균등화를 위해 JSONL 파일 재처리...")
                
                # 임시 파일로 재작성
                balanced_train_path = os.path.join(cache_dir, "train_balanced.jsonl")
                balanced_test_path = os.path.join(cache_dir, "test_balanced.jsonl")
            
                domain_train_samples = defaultdict(list)
                domain_test_samples = defaultdict(list)
                
                # JSON 파싱 오류 추적 (리스트 사용 - mutable)
                json_parse_errors = {"train": [0], "test": [0]}
                
                # 기존 JSONL 파일 읽기 (robust한 JSON 파싱)
                def safe_json_loads(line, line_num=None, error_counter=None):
                    """안전한 JSON 파싱 - 오류 발생 시 None 반환"""
                    try:
                        # 줄바꿈 제거 및 공백 정리
                        line = line.strip()
                        if not line:
                            return None
                        
                        # JSON 파싱 시도
                        try:
                            sample = json.loads(line)
                        except json.JSONDecodeError as e:
                            # 멀티라인 JSON 시도 (라인 끝에 불완전한 JSON이 있을 수 있음)
                            # 마지막 불완전한 객체를 제거하고 재시도
                            if e.pos is not None and e.pos < len(line):
                                # 불완전한 부분을 제거하고 재시도
                                truncated_line = line[:e.pos].rstrip()
                                # 마지막 불완전한 객체 제거
                                if truncated_line:
                                    # 마지막 불완전한 객체의 시작 부분 찾기
                                    last_brace = truncated_line.rfind('}')
                                    last_bracket = truncated_line.rfind(']')
                                    last_pos = max(last_brace, last_bracket)
                                    if last_pos > 0:
                                        truncated_line = truncated_line[:last_pos + 1]
                                        try:
                                            sample = json.loads(truncated_line)
                                        except:
                                            raise e
                                    else:
                                        raise e
                                else:
                                    raise e
                            else:
                                raise e
                        
                        # 샘플이 dict가 아니면 None 반환
                        if not isinstance(sample, dict):
                            return None
                        
                        # VLM 형식으로 변환
                        sample = ensure_vlm_format(sample)
                        
                        return sample
                    except json.JSONDecodeError as e:
                        if error_counter is not None:
                            error_counter[0] += 1
                        if line_num is not None and error_counter is not None and error_counter[0] <= 10:
                            # 처음 10개 오류만 상세 로그 출력
                            logger.warning(f"⚠️ JSON 파싱 오류 (라인 {line_num}): {e}")
                            logger.warning(f"   문제가 있는 라인 (처음 200자): {line[:200]}")
                        return None
                    except Exception as e:
                        if error_counter is not None:
                            error_counter[0] += 1
                        if line_num is not None and error_counter is not None and error_counter[0] <= 10:
                            # 처음 10개 오류만 상세 로그 출력
                            logger.warning(f"⚠️ 예상치 못한 오류 (라인 {line_num}): {e}")
                        return None
                
                # Train 파일 읽기
                train_line_num = 0
                with open(train_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        train_line_num += 1
                        sample = safe_json_loads(line, train_line_num, error_counter=json_parse_errors["train"])
                        if sample is not None:
                            # Filter out text-only samples if allow_text_only=False
                            if not allow_text_only:
                                images = sample.get("images", [])
                                if not images or len(images) == 0:
                                    continue  # Skip text-only samples

                            domain = sample.get("domain", "unknown")
                            domain_train_samples[domain].append(sample)
                
                # Test 파일 읽기
                test_line_num = 0
                with open(test_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        test_line_num += 1
                        sample = safe_json_loads(line, test_line_num, error_counter=json_parse_errors["test"])
                        if sample is not None:
                            # Filter out text-only samples if allow_text_only=False
                            if not allow_text_only:
                                images = sample.get("images", [])
                                if not images or len(images) == 0:
                                    continue  # Skip text-only samples

                            domain = sample.get("domain", "unknown")
                            domain_test_samples[domain].append(sample)
                
                # JSON 파싱 오류 로그 출력
                train_parse_errors = json_parse_errors["train"][0]
                test_parse_errors = json_parse_errors["test"][0]
                if train_parse_errors > 0 or test_parse_errors > 0:
                    logger.warning(f"⚠️ JSON 파싱 오류 발생: Train {train_parse_errors}개, Test {test_parse_errors}개 (건너뜀)")
                
                # 각 도메인별로 최소 샘플 수만큼만 사용
                balanced_domain_counts = defaultdict(lambda: {"train": 0, "test": 0})
                
                with open(balanced_train_path, "w", encoding="utf-8") as train_f, \
                     open(balanced_test_path, "w", encoding="utf-8") as test_f:
                    
                    for domain in domain_configs.keys():
                        # Train 샘플 균등화
                        train_samples = domain_train_samples[domain]
                        if len(train_samples) > min_train:
                            random.shuffle(train_samples)
                            train_samples = train_samples[:min_train]
                        
                        for sample in train_samples:
                            try:
                                # VLM 형식으로 변환
                                sample = ensure_vlm_format(sample)
                                # JSON 직렬화 가능한 형태로 정리
                                sample = sanitize_sample_for_json(sample)
                                # ensure_ascii=False로 특수 문자 제대로 처리
                                train_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                                balanced_domain_counts[domain]["train"] += 1
                                balanced_train_count += 1
                            except Exception as e:
                                logger.warning(f"⚠️ Train 샘플 저장 실패 (도메인: {domain}): {e}")
                                continue
                        
                        # Test 샘플 균등화
                        test_samples = domain_test_samples[domain]
                        if len(test_samples) > min_test:
                            random.shuffle(test_samples)
                            test_samples = test_samples[:min_test]
                        
                        for sample in test_samples:
                            try:
                                # VLM 형식으로 변환
                                sample = ensure_vlm_format(sample)
                                # JSON 직렬화 가능한 형태로 정리
                                sample = sanitize_sample_for_json(sample)
                                # ensure_ascii=False로 특수 문자 제대로 처리
                                test_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                                balanced_domain_counts[domain]["test"] += 1
                                balanced_test_count += 1
                            except Exception as e:
                                logger.warning(f"⚠️ Test 샘플 저장 실패 (도메인: {domain}): {e}")
                                continue
                
                # 균등화된 파일 검증 및 정제
                logger.info("🔍 균등화된 파일 검증 중...")
                
                # Train 파일 검증
                if os.path.exists(balanced_train_path):
                    valid_train_lines = []
                    with open(balanced_train_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                sample = json.loads(line)
                                sample = ensure_vlm_format(sample)
                                sample = sanitize_sample_for_json(sample)
                                if isinstance(sample, dict) and "messages" in sample:
                                    valid_train_lines.append(json.dumps(sample, ensure_ascii=False))
                            except:
                                continue
                    
                    if valid_train_lines:
                        with open(balanced_train_path, "w", encoding="utf-8") as f:
                            for line in valid_train_lines:
                                f.write(line + "\n")
                        balanced_train_count = len(valid_train_lines)
                    else:
                        logger.warning("⚠️ 균등화된 Train 파일에 유효한 샘플이 없습니다.")
                        balanced_train_count = 0
                
                # Test 파일 검증
                if os.path.exists(balanced_test_path):
                    valid_test_lines = []
                    with open(balanced_test_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                sample = json.loads(line)
                                sample = ensure_vlm_format(sample)
                                sample = sanitize_sample_for_json(sample)
                                if isinstance(sample, dict) and "messages" in sample:
                                    valid_test_lines.append(json.dumps(sample, ensure_ascii=False))
                            except:
                                continue
                    
                    if valid_test_lines:
                        with open(balanced_test_path, "w", encoding="utf-8") as f:
                            for line in valid_test_lines:
                                f.write(line + "\n")
                        balanced_test_count = len(valid_test_lines)
                    else:
                        logger.warning("⚠️ 균등화된 Test 파일에 유효한 샘플이 없습니다.")
                        balanced_test_count = 0
                
                # 균등화된 파일로 교체
                train_jsonl_path = balanced_train_path
                test_jsonl_path = balanced_test_path
                
                logger.info("📊 도메인별 샘플 통계 (균등화 후):")
                for domain, counts in balanced_domain_counts.items():
                    logger.info(f"   - {domain}: Train {counts['train']}개, Test {counts['test']}개")
                
                logger.info(f"✅ 균등화 완료: Train {balanced_train_count}개, Test {balanced_test_count}개")
                
                # 균등화 후 샘플이 없으면 오류 발생
                if balanced_train_count == 0 and balanced_test_count == 0:
                    error_msg = "❌ 균등화 후 샘플이 없습니다. JSON 파싱 오류나 필터링으로 인해 모든 샘플이 제거되었을 수 있습니다."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                total_train = sum(c["train"] for c in domain_counts.values())
                total_test = sum(c["test"] for c in domain_counts.values())
                balanced_train_count = total_train
                balanced_test_count = total_test
                logger.debug(f"✅ 총 샘플 수집 완료: Train {total_train}개, Test {total_test}개")
        else:
            balanced_train_count = 0
            balanced_test_count = 0
        
        # JSONL 파일로부터 데이터셋 로드
        original_data_files = {}
        data_files = {}
        final_train_count = balanced_train_count
        final_test_count = balanced_test_count
        
        if final_train_count > 0:
            original_data_files["train"] = train_jsonl_path
            data_files["train"] = train_jsonl_path
        if final_test_count > 0:
            original_data_files["test"] = test_jsonl_path
            data_files["test"] = test_jsonl_path

        if not data_files:
            error_msg = "❌ 변환된 훈련 샘플이 없습니다. 데이터셋 형식을 확인하세요."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if final_train_count == 0:
            error_msg = "❌ Train 샘플이 없습니다."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("🧠 JSONL 파일로부터 데이터셋 로딩...")
        logger.info(f"   - Train 파일: {data_files.get('train', 'N/A')}")
        logger.info(f"   - Test 파일: {data_files.get('test', 'N/A')}")
        
        # JSONL 파일 검증 및 정제 (빈 파일, 잘못된 JSON 라인 제거)
        cleaned_data_files = {}
        for split_name, file_path in data_files.items():
            try:
                logger.info(f"   📋 {split_name} 파일 검증 및 정제 중...")
                
                # 파일 존재 및 크기 확인
                if not os.path.exists(file_path):
                    logger.error(f"   ❌ {split_name} 파일이 존재하지 않습니다: {file_path}")
                    continue
                
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    logger.error(f"   ❌ {split_name} 파일이 비어있습니다: {file_path}")
                    continue
                
                # 파일 읽기 및 유효한 JSON 라인만 추출
                valid_lines = []
                total_lines = 0
                invalid_lines = 0
                
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        total_lines += 1
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            # JSON 파싱 시도
                            sample = json.loads(line)
                            
                            # VLM 형식으로 변환
                            sample = ensure_vlm_format(sample)
                            
                            # JSON 직렬화 가능한 형태로 정리
                            sample = sanitize_sample_for_json(sample)
                            
                            # 유효한 샘플인지 확인 (messages 필드 필수)
                            if not isinstance(sample, dict) or "messages" not in sample:
                                invalid_lines += 1
                                if invalid_lines <= 5:
                                    logger.warning(f"   ⚠️ {split_name} 파일 {line_num}번째 줄: messages 필드가 없음")
                                continue
                            
                            # 유효한 JSON 라인으로 저장
                            valid_lines.append(json.dumps(sample, ensure_ascii=False))
                            
                        except json.JSONDecodeError as e:
                            invalid_lines += 1
                            if invalid_lines <= 5:
                                logger.warning(f"   ⚠️ {split_name} 파일 {line_num}번째 줄 JSON 파싱 실패: {e}")
                                logger.warning(f"      줄 내용 (처음 200자): {line[:200]}")
                            continue
                        except Exception as e:
                            invalid_lines += 1
                            if invalid_lines <= 5:
                                logger.warning(f"   ⚠️ {split_name} 파일 {line_num}번째 줄 처리 중 오류: {e}")
                            continue
                
                # 유효한 라인이 없으면 건너뛰기
                if not valid_lines:
                    logger.error(f"   ❌ {split_name} 파일에 유효한 JSON 라인이 없습니다 (총 {total_lines}줄, 유효하지 않은 라인 {invalid_lines}개)")
                    continue
                
                # 정제된 파일로 저장
                cleaned_file_path = file_path + ".cleaned"
                with open(cleaned_file_path, "w", encoding="utf-8") as f:
                    for valid_line in valid_lines:
                        f.write(valid_line + "\n")
                
                logger.info(f"   ✅ {split_name} 파일 검증 완료: 총 {total_lines}줄 중 {len(valid_lines)}개 유효 (크기: {file_size / 1024 / 1024:.2f} MB)")
                if invalid_lines > 0:
                    logger.warning(f"   ⚠️ {invalid_lines}개 유효하지 않은 라인 제거됨")
                
                cleaned_data_files[split_name] = cleaned_file_path
                
            except Exception as e:
                logger.error(f"   ❌ {split_name} 파일 검증 실패: {e}")
                traceback.print_exc()
        
        # 정제된 파일이 없으면 오류
        if not cleaned_data_files:
            error_msg = "❌ 정제된 JSONL 파일이 없습니다. 모든 파일이 비어있거나 유효하지 않습니다."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # 정제된 파일로 교체
        data_files = cleaned_data_files
        
        
        try:
            # Reload from JSONL with memory mapping enabled (streaming=False but using load_dataset("json"))
            # JSONL 형식으로 명시적으로 지정 (lines=True)
            logger.info("📦 데이터셋 로딩 중...")
            dataset_dict = load_dataset("json", data_files=data_files, features=SFT_JSON_FEATURES)
        except Exception as load_e:
            logger.error(f"❌ JSONL 파일 로딩 실패: {load_e}")
            logger.error(f"   - Train 파일: {data_files.get('train', 'N/A')}")
            logger.error(f"   - Test 파일: {data_files.get('test', 'N/A')}")
            
            # 문제가 있는 샘플 찾기 (58번째 줄 주변 포함)
            for split_name, file_path in data_files.items():
                logger.debug(f"   🔍 {split_name} 파일에서 문제 샘플 검색 중...")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            # 58번째 줄 주변과 처음 100줄 확인
                            if line_num > 100 and (line_num < 50 or line_num > 70):
                                continue
                            if line.strip():
                                try:
                                    sample = json.loads(line)
                                    # messages 구조 확인
                                    if "messages" in sample:
                                        for msg_idx, msg in enumerate(sample["messages"]):
                                            if "content" in msg:
                                                content = msg["content"]
                                                # content가 문자열인 경우
                                                if isinstance(content, str):
                                                    logger.error(f"   ❌ {split_name} 파일 {line_num}번째 줄, 메시지 {msg_idx}: content가 문자열임 (리스트여야 함)")
                                                    logger.error(f"      content 타입: {type(content)}, 값: {repr(content)[:200]}")
                                                    logger.error(f"      전체 메시지: {json.dumps(msg, ensure_ascii=False, indent=2)[:500]}")
                                                # content가 리스트인 경우
                                                elif isinstance(content, list):
                                                    for content_idx, content_item in enumerate(content):
                                                        # content_item이 문자열인 경우
                                                        if isinstance(content_item, str):
                                                            logger.error(f"   ❌ {split_name} 파일 {line_num}번째 줄, 메시지 {msg_idx}, content[{content_idx}]: 문자열임 (객체여야 함)")
                                                            logger.error(f"      content_item 타입: {type(content_item)}, 값: {repr(content_item)[:200]}")
                                                        # content_item이 dict인 경우
                                                        elif isinstance(content_item, dict):
                                                            if "text" in content_item:
                                                                text_value = content_item["text"]
                                                                if not isinstance(text_value, str):
                                                                    logger.error(f"   ❌ {split_name} 파일 {line_num}번째 줄, 메시지 {msg_idx}, content[{content_idx}]: text가 문자열이 아님")
                                                                    logger.error(f"      타입: {type(text_value)}, 값: {repr(text_value)[:100]}")
                                                else:
                                                    logger.error(f"   ❌ {split_name} 파일 {line_num}번째 줄, 메시지 {msg_idx}: content가 예상치 못한 타입")
                                                    logger.error(f"      content 타입: {type(content)}, 값: {repr(content)[:200]}")
                                except json.JSONDecodeError as e:
                                    logger.error(f"   ❌ {split_name} 파일 {line_num}번째 줄: JSON 파싱 실패 - {e}")
                                    logger.error(f"      줄 내용: {line[:300]}")
                except Exception as e:
                    logger.error(f"   ❌ {split_name} 파일 검색 중 오류: {e}")
                    traceback.print_exc()
            
            raise
        
        # 정제된 파일 정리 (원본 파일로 교체)
        logger.info("🧹 정제된 파일 정리 중...")
        for split_name, cleaned_file_path in cleaned_data_files.items():
            try:
                if split_name in original_data_files:
                    original_file_path = original_data_files[split_name]
                    if os.path.exists(cleaned_file_path):
                        # 정제된 파일로 원본 파일 교체
                        if os.path.exists(original_file_path):
                            os.remove(original_file_path)
                        shutil.move(cleaned_file_path, original_file_path)
                        logger.debug(f"   ✅ {split_name} 파일 정제 완료: {original_file_path}")
            except Exception as e:
                logger.warning(f"   ⚠️ {split_name} 파일 정리 중 오류: {e}")
        
        logger.info("🖼️ 이미지 경로를 이미지 객체로 캐스팅 (lazy loading)...")
        for split in dataset_dict:
            current_features = dataset_dict[split].features
            new_features = current_features.copy()
            if 'images' in new_features:
                def preprocess_images(example):
                    """이미지 데이터 전처리 - 중첩 리스트 평면화"""
                    if 'images' in example and example['images']:
                        example['images'] = validate_image_data(example['images'])
                    # 이미지가 없으면 빈 리스트로 유지
                    elif 'images' not in example:
                        example['images'] = []
                    return example
                
                dataset_dict[split] = dataset_dict[split].map(preprocess_images)
                # 이미지가 있는 샘플만 Sequence(DatasetImage)로 캐스팅
                # 빈 리스트는 그대로 유지
                if isinstance(new_features['images'], Sequence):
                    new_features['images'] = Sequence(DatasetImage(decode=True))
                    dataset_dict[split] = dataset_dict[split].cast(new_features)

        logger.debug("✅ 멀티 도메인 데이터셋 생성 완료")
        
        # 처리 완료 후 메타데이터 저장
        try:
            cache_meta = {
                "created_at": datetime.now().isoformat(),
                "train_count": balanced_train_count,
                "test_count": balanced_test_count,
                "domain_configs": domain_configs,
                "max_samples_per_domain": max_samples_per_domain,
                "test_size": test_size,
                "use_streaming": use_streaming,
                "max_workers": max_workers,
                "cache_key": cache_key
            }
            with open(cache_meta_path, "w", encoding="utf-8") as f:
                json.dump(cache_meta, f, indent=2, ensure_ascii=False)
            logger.debug(f"💾 데이터셋 캐시 저장 완료: {cache_dir}")
        except Exception as e:
            logger.warning(f"⚠️ 캐시 메타데이터 저장 실패: {e}")
        
        return dataset_dict

    except Exception as e:
        logger.error(f"❌ 멀티 도메인 데이터셋 로딩 실패: {e}")
        traceback.print_exc()
        # 실패 시에도 캐시 디렉토리는 유지 (부분적으로 처리된 데이터가 있을 수 있음)
        # shutil.rmtree(cache_dir, ignore_errors=True)
        raise Exception(f"😢 멀티 도메인 데이터셋 로딩 시도가 실패했습니다.") from e


def create_simple_collate_fn(processor, max_length: int = 2048, allow_text_only: bool = True):
    """
    SFTTrainer용 커스텀 data collator - DeepSpeed ZeRO-3 최적화 버전
    (모든 랭크가 동일한 모달리티 구조를 갖도록 대칭성 유지)
    """
    import re
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
    
    class CustomSFTDataCollator(DataCollatorForVisionLanguageModeling):
        def __init__(self, processor, max_length: int = 2048, allow_text_only: bool = True):
            super().__init__(processor=processor, max_length=max_length)
            self.processor = processor
            self.max_length = max_length
            self.allow_text_only = allow_text_only
            
            # No Siglip - use native processor for Qwen3-VL-MoE compatibility

            # Detect image token for Qwen3-VL/Qwen2-VL
            self.image_token = None
            for attr in ['image_token', 'im_start_token', 'vision_token']:
                if hasattr(self.processor, attr):
                    token = getattr(self.processor, attr)
                    if isinstance(token, str): self.image_token = token; break
            if self.image_token is None: self.image_token = '<image>'
            
            # Dummy image for ZeRO-3 symmetry (Compatible with Qwen3-VL-MoE)
            # Use size divisible by patch_size(16) * spatial_merge_size(2)
            self.dummy_image = Image.new('RGB', (64, 64), (0, 0, 0))

        def _collate_language_modeling(self, examples):
            """
            Modified for Universal Exoskeleton:
            1. Enforce specific token count (196) to match Siglip vision tower features.
            2. Bypass Qwen dynamic token calculation by passing images=None to processor.
            3. Manually inject Siglip-processed pixel_values and image_grid_thw.
            """
            messages = [example["messages"] for example in examples]
            
            # 1. Collect Images & Texts
            images = []
            has_real_images = []
            
            for example in examples:
                img = example.get("images", None)
                if img is not None and (isinstance(img, list) and len(img) > 0):
                    extracted_img = img[0] if isinstance(img, list) else img
                    images.append(extracted_img)
                    has_real_images.append(True)
                else:
                    images.append(self.dummy_image)
                    has_real_images.append(False)

            tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
            
            texts = []
            for i, m in enumerate(messages):
                # Check if message has image placeholder
                has_image_in_msg = False
                for msg in m:
                    if isinstance(msg.get('content'), list):
                        for item in msg['content']:
                            if isinstance(item, dict) and item.get('type') == 'image':
                                has_image_in_msg = True
                
                # If no image in message but we have image, inject image placeholder
                if not has_image_in_msg:
                    if m and m[0]['role'] == 'user':
                        if isinstance(m[0]['content'], str):
                            m[0]['content'] = [{"type": "image"}, {"type": "text", "text": m[0]['content']}]
                        elif isinstance(m[0]['content'], list):
                            m[0]['content'].insert(0, {"type": "image"})

                # CRITICAL: Use PROCESSOR.apply_chat_template, not tokenizer!
                # Only the processor knows how to generate image tokens for Qwen3-VL-MoE
                text = self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                texts.append(text)

            # 2. Process with NATIVE processor (text + images together)
            # This ensures image_grid_thw is computed correctly for Qwen3-VL-MoE
            try:
                output = self.processor(
                    text=texts,
                    images=images,  # Let native processor handle images
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
            except Exception as e:
                print(f"⚠️ Native image processing failed: {e}. Falling back to text-only.")
                output = self.processor(
                    text=texts,
                    images=None,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )

            # 5. Handle Labels (Masking)
            input_ids = output["input_ids"]
            if "labels" not in output or output["labels"] is None:
                output["labels"] = input_ids.clone()
            
            labels = output["labels"]
            
            # Mask special vision tokens
            # 151652: <|vision_start|>, 151653: <|vision_end|>, 151655: <|image_pad|>
            special_ids = [151652, 151653, 151655]
            for sid in special_ids:
                labels[labels == sid] = -100
            
            # Mask padding
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100
                
            output["labels"] = labels
            
            return output

        def __call__(self, features):
             # Just pass through to our custom collator logic
             return self.torch_call(examples=features)
    
    return CustomSFTDataCollator(processor, max_length=max_length, allow_text_only=allow_text_only)


# 도메인별 데이터셋 빌더 함수들
def math_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True, max_workers: int = 4):
    """수학 도메인 데이터셋"""
    log_memory_usage("수학 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"math": DOMAIN_DATASETS["math"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming,
        max_workers=max_workers
    )
    log_memory_usage("수학 도메인 데이터셋 완료")
    return dataset

def science_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True, max_workers: int = 4):
    """과학 도메인 데이터셋"""
    log_memory_usage("과학 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"science": DOMAIN_DATASETS["science"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming,
        max_workers=max_workers
    )
    log_memory_usage("과학 도메인 데이터셋 완료")
    return dataset

def code_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True, max_workers: int = 4):
    """코드 도메인 데이터셋"""
    log_memory_usage("코드 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"code": DOMAIN_DATASETS["code"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming,
        max_workers=max_workers
    )
    log_memory_usage("코드 도메인 데이터셋 완료")
    return dataset

def puzzle_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True, max_workers: int = 4):
    """퍼즐 도메인 데이터셋"""
    log_memory_usage("퍼즐 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"puzzle": DOMAIN_DATASETS["puzzle"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming,
        max_workers=max_workers
    )
    log_memory_usage("퍼즐 도메인 데이터셋 완료")
    return dataset

def vision_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True, max_workers: int = 4):
    """비전 도메인 데이터셋"""
    log_memory_usage("비전 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"vision": DOMAIN_DATASETS["vision"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming,
        max_workers=max_workers
    )
    log_memory_usage("비전 도메인 데이터셋 완료")
    return dataset

def ocr_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True, max_workers: int = 4):
    """OCR 도메인 데이터셋"""
    log_memory_usage("OCR 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"ocr": DOMAIN_DATASETS["ocr"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming,
        max_workers=max_workers
    )
    log_memory_usage("OCR 도메인 데이터셋 완료")
    return dataset

def all_domains_dataset(tokenizer, max_samples_per_domain: int = 200, use_streaming: bool = True, max_workers: int = 4):
    """모든 도메인 통합 데이터셋"""
    log_memory_usage("전체 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs=DOMAIN_DATASETS,
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples_per_domain,
        use_streaming=use_streaming,
        max_workers=max_workers
    )
    log_memory_usage("전체 도메인 데이터셋 완료")
    return dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    logger.debug("🚀 멀티 도메인 데이터셋 테스트 시작")
    log_memory_usage("프로그램 시작")
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_memory_usage("토크나이저 로드 후")
    
    # 전체 도메인 데이터셋 테스트
    try:
        logger.debug("📦 전체 도메인 데이터셋 테스트")
        dataset = all_domains_dataset(tokenizer, max_samples_per_domain=50, use_streaming=True)
        log_memory_usage("전체 도메인 데이터셋 생성 후")
        
        logger.debug(f"데이터셋 생성 완료: {dataset}")
        
        # 도메인별 샘플 확인
        if 'train' in dataset:
            train_domains = {}
            for i in range(min(100, len(dataset['train']))):
                sample = dataset['train'][i]
                domain = sample.get('domain', 'unknown')
                train_domains[domain] = train_domains.get(domain, 0) + 1
            
            logger.debug(f"Train 세트 도메인 분포: {train_domains}")
        
    except Exception as e:
        logger.error(f"전체 도메인 데이터셋 테스트 실패: {e}")
        traceback.print_exc()
    
    log_memory_usage("테스트 완료")
    logger.debug("✅ 멀티 도메인 데이터셋 테스트 완료")

