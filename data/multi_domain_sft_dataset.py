import logging
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names, concatenate_datasets
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
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset, Image as DatasetImage, Sequence, Features
from collections import defaultdict

# simple_sft_dataset의 유틸리티 함수들 import
from data.simple_sft_dataset import (
    validate_image_data,
    validate_messages,
    safe_flatten_images,
    get_memory_usage,
    log_memory_usage
)

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

def convert_sample_to_messages(sample: Dict[str, Any], dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    샘플을 messages 형식으로 변환 (멀티 도메인 데이터셋 지원 확장)
    """
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
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": sample["instruction"]}]},
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
    
    # OCR-VQA 계열 (일반 VQA 스키마 재사용)
    if "ocr-vqa" in dataset_name.lower() or "ocrvqa" in dataset_name.lower():
        question = sample.get("question", "")
        answers = sample.get("answers", [])
        answer = ""
        if isinstance(answers, list) and answers:
            if isinstance(answers[0], dict):
                answer = answers[0].get("answer", "")
            else:
                answer = str(answers[0])
        else:
            answer = sample.get("answer", "")
        if not question:
            return None
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
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        if not instruction or not output:
            return None
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": output}]}
        ]
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
            return {"messages": messages, "images": []}
    
    # base_convert 결과에 이미지가 없으면 빈 리스트로 설정
    if result and "images" in result:
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
        "armanc/ScienceQA"
    ],
    "code": [
        "theblackcat102/evol-codealpaca-v1", # Evol-CodeAlpaca: 코드 instruction (학습용)
        "microsoft/rStar-Coder",  # rStar-Coder: 코드 데이터셋
    ],
    "puzzle": [
        "openbmb/UltraInteract_sft",  # UltraInteract_sft: 논리 추론 instruction 데이터셋 (학습용)
        "HuggingFaceH4/ultrafeedback_binarized",  # UltraFeedback: 존재하지 않음, 대체 필요
    ],
    "vision": [
        "lmms-lab/LLaVA-OneVision-Data",  # LLaVA-OneVision-Data: 다양한 비전 태스크 (멀티모달)
        # "textvqa",  # TextVQA: 존재하지 않음, 대체 필요
    ],
    "ocr": [
        "howard-hou/OCR-VQA",  # OCR-VQA: OCR 질의응답 데이터셋
        "allenai/olmOCR-mix-1025",  # olmOCR-mix: PDF OCR 데이터셋
    ],
    "chat": [
        "HuggingFaceTB/smoltalk",  # SmolTalk: 일반 채팅 (멀티모달 가능)
        "Open-Orca/OpenOrca",  # OpenOrca: 일반 대화 (텍스트 전용)
    ]
}

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

def get_multi_domain_sft_dataset(
    domain_configs: Optional[Dict[str, List[str]]] = None,
    tokenizer=None,
    max_length: int = 2048,
    max_samples_per_domain: int = 200,
    test_size: float = 0.1,
    use_streaming: bool = True,
    chunk_size: int = 1000
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
    
    Returns:
        DatasetDict with train/test splits, 각 샘플에 'domain' 필드 포함
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    if domain_configs is None:
        domain_configs = DOMAIN_DATASETS
    
    logger.info(f"📦 멀티 도메인 데이터셋 로딩 시작")
    logger.info(f"   - 도메인 수: {len(domain_configs)}개")
    logger.info(f"   - 도메인당 최대 샘플: {max_samples_per_domain}개")
    logger.info(f"   - 총 최대 샘플: {max_samples_per_domain * len(domain_configs)}개")
    logger.info(f"   - streaming: {use_streaming}")
    
    log_memory_usage("멀티 도메인 데이터셋 로딩 시작")
    
    base_temp_dir = "/mls/conan/tmp"
    os.makedirs(base_temp_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=base_temp_dir)
    logger.info(f"📂 임시 디렉토리 생성: {temp_dir}")
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    try:
        # 도메인별 샘플 카운터
        domain_counts = defaultdict(lambda: {"train": 0, "test": 0})
        total_processed = 0
        image_counter = 0
        
        train_jsonl_path = os.path.join(temp_dir, "train.jsonl")
        test_jsonl_path = os.path.join(temp_dir, "test.jsonl")

        with open(train_jsonl_path, "w", encoding="utf-8") as train_f, \
             open(test_jsonl_path, "w", encoding="utf-8") as test_f:
            
            # 각 도메인별로 처리
            domain_pbar = tqdm(domain_configs.items(), desc="도메인 처리", unit="domain")
            
            for domain, dataset_names in domain_pbar:
                domain_pbar.set_description(f"도메인: {domain}")
                
                # 빈 데이터셋 리스트인 경우 건너뛰기
                if not dataset_names:
                    logger.warning(f"   ⚠️ {domain} 도메인에 데이터셋이 지정되지 않았습니다. 건너뜁니다.")
                    continue
                
                domain_processed = 0
                
                # ScienceQA 미러 중복 방지 플래그
                scienceqa_taken = False
                
                for dataset_name in dataset_names:
                    if domain_processed >= max_samples_per_domain:
                        break
                    
                    try:
                        logger.info(f"   📋 {domain} 도메인 - 데이터셋: {dataset_name}")
                        
                        # 데이터셋 존재 확인
                        if not dataset_exists(dataset_name):
                            continue
                        
                        # 데이터셋의 config 목록 가져오기 (모든 서브셋 확인)
                        try:
                            available_configs = get_dataset_config_names(dataset_name)
                            if not available_configs:
                                logger.warning(f"   ⚠️ {dataset_name}에 config가 없습니다. 기본 split 사용")
                                available_configs = ["default"]
                            else:
                                logger.info(f"   📋 {dataset_name} - 사용 가능한 모든 Config/Subset ({len(available_configs)}개):")
                                # 모든 config를 출력 (제한 없이)
                                # for idx, c in enumerate(available_configs, 1):
                                #     logger.info(f"      {idx}. {c}")
                                logger.info(f"   ✅ 총 {len(available_configs)}개 서브셋 확인 완료")
                        except Exception as e:
                            logger.warning(f"   ⚠️ Config 목록 가져오기 실패: {e}, 기본 split 사용")
                            available_configs = ["default"]
                        
                        # LLaVA-OneVision-Data는 onevision 서브셋만 사용 (없으면 사용 가능한 config 사용)
                        if "llava-onevision" in dataset_name.lower() or "llava-onevision-data" in dataset_name.lower():
                            filtered = [c for c in available_configs if "onevision" in str(c).lower()]
                            if filtered:
                                available_configs = filtered
                            else:
                                # onevision이 없으면 처음 몇 개 config만 사용 (너무 많으면 제한)
                                logger.info(f"   ℹ️ 'onevision' config가 없습니다. 사용 가능한 config 중 일부를 사용합니다.")
                                available_configs = available_configs[:5]  # 처음 5개만 사용
                        
                        # ScienceQA 미러가 다수인 경우 한쪽만 사용
                        if domain == "science" and ("scienceqa" in dataset_name.lower()):
                            if scienceqa_taken:
                                logger.info(f"   🔁 ScienceQA 미러 중복 방지로 건너뜀: {dataset_name}")
                                continue
                            scienceqa_taken = True
                        
                        # Config별로 샘플 수 계산
                        samples_per_config = max(1, max_samples_per_domain // max(len(available_configs), 1))
                        
                        config_pbar = tqdm(available_configs, desc=f"  {domain} config", unit="config", leave=False)
                        
                        for config in config_pbar:
                            if domain_processed >= max_samples_per_domain:
                                break
                            
                            try:
                                # Config 이름으로 도메인 재확인
                                inferred_domain = get_domain_from_config(config, dataset_name)
                                if inferred_domain and inferred_domain != domain:
                                    logger.debug(f"   🔄 Config {config}의 도메인이 {inferred_domain}으로 추론됨 (요청: {domain})")
                                    # 추론된 도메인이 다르면 건너뛰기 (선택적)
                                    # continue
                                
                                config_pbar.set_description(f"  {domain} config: {config[:30]}...")
                                
                                # 사용 가능한 split 확인
                                try:
                                    if config == "default":
                                        available_splits = get_dataset_split_names(dataset_name)
                                    else:
                                        available_splits = get_dataset_split_names(dataset_name, config_name=config)
                                    
                                    logger.info(f"   📋 Config {config} - 사용 가능한 split: {available_splits}")
                                    
                                    # Train split 선택: train_sft > train
                                    train_split = None
                                    if "train_sft" in available_splits:
                                        train_split = "train_sft"
                                        logger.info(f"   ✅ Train split 선택: train_sft")
                                    elif "train" in available_splits:
                                        train_split = "train"
                                        logger.info(f"   ✅ Train split 선택: train")
                                    else:
                                        logger.warning(f"   ⚠️ Config {config}에 train 또는 train_sft split이 없습니다. 건너뜁니다.")
                                        continue
                                    
                                    # Test split 선택: test_sft > test (없어도 계속 진행)
                                    test_split = None
                                    if "test_sft" in available_splits:
                                        test_split = "test_sft"
                                        logger.info(f"   ✅ Test split 선택: test_sft")
                                    elif "test" in available_splits:
                                        test_split = "test"
                                        logger.info(f"   ✅ Test split 선택: test")
                                    else:
                                        logger.info(f"   ℹ️ Config {config}에 test 또는 test_sft split이 없습니다. train만 사용합니다.")
                                    
                                except Exception as e:
                                    logger.warning(f"   ⚠️ Split 목록 가져오기 실패: {e}, 기본 train 사용")
                                    train_split = "train"
                                    test_split = None
                                
                                # Train split 처리
                                try:
                                    if config == "default":
                                        train_dataset = load_dataset(
                                            path=dataset_name,
                                            split=train_split,
                                            streaming=use_streaming
                                        )
                                    else:
                                        train_dataset = load_dataset(
                                            path=dataset_name,
                                            name=config,
                                            split=train_split,
                                            streaming=use_streaming
                                        )
                                    
                                    train_samples_per_config = samples_per_config
                                    if test_split:
                                        # test split이 있으면 train 샘플 수를 조정
                                        train_samples_per_config = int(samples_per_config * (1 - test_size))
                                    
                                    sample_pbar = tqdm(
                                        total=min(train_samples_per_config, max_samples_per_domain - domain_processed),
                                        desc=f"    Train 샘플 처리",
                                        unit="sample",
                                        leave=False
                                    )
                                    
                                    train_processed = 0
                                    for sample in train_dataset:
                                        if domain_processed >= max_samples_per_domain or train_processed >= train_samples_per_config:
                                            break
                                        
                                        # 샘플 변환
                                        converted = convert_sample_to_messages(sample, dataset_name)
                                        if not converted:
                                            continue

                                        # 이미지 처리
                                        image_paths = []
                                        if "images" in converted and converted["images"]:
                                            flattened_images = validate_image_data(converted["images"])
                                            
                                            if flattened_images:
                                                valid_sample = True
                                                
                                                for img_obj in flattened_images:
                                                    if isinstance(img_obj, Image.Image):
                                                        try:
                                                            img_path = os.path.join(images_dir, f"{image_counter}.png")
                                                            img_obj.save(img_path, "PNG")
                                                            image_paths.append(img_path)
                                                            image_counter += 1
                                                        except Exception as img_e:
                                                            logger.warning(f"⚠️ 이미지 저장 실패: {img_e}")
                                                            valid_sample = False
                                                            break
                                                    elif img_obj is not None:
                                                        logger.warning(f"⚠️ 지원되지 않는 이미지 타입: {type(img_obj)}")
                                                        valid_sample = False
                                                        break
                                                
                                                if not valid_sample:
                                                    continue
                                        
                                        converted["images"] = image_paths
                                        converted["domain"] = domain
                                        
                                        train_f.write(json.dumps(converted) + "\n")
                                        domain_counts[domain]["train"] += 1
                                        domain_processed += 1
                                        total_processed += 1
                                        train_processed += 1
                                        
                                        sample_pbar.update(1)
                                        memory_gb = get_memory_usage()
                                        sample_pbar.set_postfix({
                                            "도메인": f"{domain_processed}/{max_samples_per_domain}",
                                            "총 처리": f"{total_processed}",
                                            "메모리": f"{memory_gb:.1f}GB"
                                        })
                                    
                                    sample_pbar.close()
                                    del train_dataset
                                    gc.collect()
                                    
                                except Exception as e:
                                    logger.warning(f"   ⚠️ Train split {train_split} 로드 실패: {e}")
                                    continue
                                
                                # Test split 처리 (있는 경우)
                                if test_split:
                                    try:
                                        if config == "default":
                                            test_dataset = load_dataset(
                                                path=dataset_name,
                                                split=test_split,
                                                streaming=use_streaming
                                            )
                                        else:
                                            test_dataset = load_dataset(
                                                path=dataset_name,
                                                name=config,
                                                split=test_split,
                                                streaming=use_streaming
                                            )
                                        
                                        test_samples_per_config = int(samples_per_config * test_size)
                                        
                                        sample_pbar = tqdm(
                                            total=min(test_samples_per_config, int(max_samples_per_domain * test_size)),
                                            desc=f"    Test 샘플 처리",
                                            unit="sample",
                                            leave=False
                                        )
                                        
                                        test_processed = 0
                                        for sample in test_dataset:
                                            if test_processed >= test_samples_per_config:
                                                break
                                            
                                            # 샘플 변환
                                            converted = convert_sample_to_messages(sample, dataset_name)
                                            if not converted:
                                                continue

                                            # 이미지 처리
                                            image_paths = []
                                            if "images" in converted and converted["images"]:
                                                flattened_images = validate_image_data(converted["images"])
                                                
                                                if flattened_images:
                                                    valid_sample = True
                                                    
                                                    for img_obj in flattened_images:
                                                        if isinstance(img_obj, Image.Image):
                                                            try:
                                                                img_path = os.path.join(images_dir, f"{image_counter}.png")
                                                                img_obj.save(img_path, "PNG")
                                                                image_paths.append(img_path)
                                                                image_counter += 1
                                                            except Exception as img_e:
                                                                logger.warning(f"⚠️ 이미지 저장 실패: {img_e}")
                                                                valid_sample = False
                                                                break
                                                        elif img_obj is not None:
                                                            logger.warning(f"⚠️ 지원되지 않는 이미지 타입: {type(img_obj)}")
                                                            valid_sample = False
                                                            break
                                                    
                                                    if not valid_sample:
                                                        continue
                                            
                                            converted["images"] = image_paths
                                            converted["domain"] = domain
                                            
                                            test_f.write(json.dumps(converted) + "\n")
                                            domain_counts[domain]["test"] += 1
                                            total_processed += 1
                                            test_processed += 1
                                            
                                            sample_pbar.update(1)
                                            memory_gb = get_memory_usage()
                                            sample_pbar.set_postfix({
                                                "총 처리": f"{total_processed}",
                                                "메모리": f"{memory_gb:.1f}GB"
                                            })
                                        
                                        sample_pbar.close()
                                        del test_dataset
                                        gc.collect()
                                        
                                    except Exception as e:
                                        logger.warning(f"   ⚠️ Test split {test_split} 로드 실패: {e}")
                                        # Test split 실패해도 계속 진행
                                
                            except Exception as e:
                                logger.warning(f"   ⚠️ Config {config} 처리 실패: {e}")
                                continue
                        
                        config_pbar.close()
                        
                    except Exception as e:
                        logger.warning(f"   ⚠️ 데이터셋 {dataset_name} 처리 실패: {e}")
                        continue
                
                logger.info(f"   ✅ {domain} 도메인 완료: Train {domain_counts[domain]['train']}개, Test {domain_counts[domain]['test']}개")
            
            domain_pbar.close()

        # 도메인별 통계 출력
        logger.info("📊 도메인별 샘플 통계 (균등화 전):")
        for domain, counts in domain_counts.items():
            logger.info(f"   - {domain}: Train {counts['train']}개, Test {counts['test']}개")
        
        # 도메인별 샘플 수 균등화
        # 각 도메인에서 동일한 수의 샘플을 사용하도록 조정
        balanced_train_count = 0
        balanced_test_count = 0
        
        if domain_counts:
            min_train = min([c["train"] for c in domain_counts.values()] + [max_samples_per_domain])
            min_test = min([c["test"] for c in domain_counts.values()] + [int(max_samples_per_domain * test_size)])
            
            logger.info(f"⚖️ 도메인별 샘플 수 균등화:")
            logger.info(f"   - 최소 Train 샘플 수: {min_train}개")
            logger.info(f"   - 최소 Test 샘플 수: {min_test}개")
            
            # JSONL 파일을 다시 읽어서 균등화
            if min_train > 0 or min_test > 0:
                logger.info("🔄 샘플 수 균등화를 위해 JSONL 파일 재처리...")
                
                # 임시 파일로 재작성
                balanced_train_path = os.path.join(temp_dir, "train_balanced.jsonl")
                balanced_test_path = os.path.join(temp_dir, "test_balanced.jsonl")
            
                domain_train_samples = defaultdict(list)
                domain_test_samples = defaultdict(list)
                
                # 기존 JSONL 파일 읽기
                with open(train_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            domain = sample.get("domain", "unknown")
                            domain_train_samples[domain].append(sample)
                
                with open(test_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            domain = sample.get("domain", "unknown")
                            domain_test_samples[domain].append(sample)
                
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
                            train_f.write(json.dumps(sample) + "\n")
                            balanced_domain_counts[domain]["train"] += 1
                            balanced_train_count += 1
                        
                        # Test 샘플 균등화
                        test_samples = domain_test_samples[domain]
                        if len(test_samples) > min_test:
                            random.shuffle(test_samples)
                            test_samples = test_samples[:min_test]
                        
                        for sample in test_samples:
                            test_f.write(json.dumps(sample) + "\n")
                            balanced_domain_counts[domain]["test"] += 1
                            balanced_test_count += 1
                
                # 균등화된 파일로 교체
                train_jsonl_path = balanced_train_path
                test_jsonl_path = balanced_test_path
                
                logger.info("📊 도메인별 샘플 통계 (균등화 후):")
                for domain, counts in balanced_domain_counts.items():
                    logger.info(f"   - {domain}: Train {counts['train']}개, Test {counts['test']}개")
                
                logger.info(f"✅ 균등화 완료: Train {balanced_train_count}개, Test {balanced_test_count}개")
            else:
                total_train = sum(c["train"] for c in domain_counts.values())
                total_test = sum(c["test"] for c in domain_counts.values())
                balanced_train_count = total_train
                balanced_test_count = total_test
                logger.info(f"✅ 총 샘플 수집 완료: Train {total_train}개, Test {total_test}개")
        else:
            balanced_train_count = 0
            balanced_test_count = 0
        
        # JSONL 파일로부터 데이터셋 로드
        data_files = {}
        final_train_count = balanced_train_count
        final_test_count = balanced_test_count
        
        if final_train_count > 0:
            data_files["train"] = train_jsonl_path
        if final_test_count > 0:
            data_files["test"] = test_jsonl_path

        if not data_files:
            raise ValueError("변환된 훈련 샘플이 없습니다. 데이터셋 형식을 확인하세요.")
        
        logger.info("🧠 JSONL 파일로부터 데이터셋 로딩...")
        dataset_dict = load_dataset("json", data_files=data_files)
        
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

        logger.info("✅ 멀티 도메인 데이터셋 생성 완료")
        
        return dataset_dict

    except Exception as e:
        logger.error(f"❌ 멀티 도메인 데이터셋 로딩 실패: {e}")
        traceback.print_exc()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise Exception(f"😢 멀티 도메인 데이터셋 로딩 시도가 실패했습니다.") from e


def create_simple_collate_fn(processor, max_length: int = 2048):
    """SFTTrainer용 커스텀 data collator - 멀티 도메인 지원"""
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
    
    class CustomSFTDataCollator(DataCollatorForVisionLanguageModeling):
        def __init__(self, processor, max_length: int = 2048):
            super().__init__(processor=processor, max_length=max_length)
            self.processor = processor
            self.max_length = max_length
            
        def __call__(self, features):
            assert features is not None, "features is None"

            for i, feature in enumerate(features):
                if "messages" in feature:
                    feature["messages"] = validate_messages(feature["messages"])
                if 'images' not in feature or not feature['images']:
                    raise ValueError(f"샘플 {i}에 이미지가 없습니다! 모든 샘플은 이미지를 포함해야 합니다.")
                
                feature['images'] = validate_image_data(feature['images'])
                if not feature['images']:
                    raise ValueError(f"샘플 {i}의 이미지가 유효하지 않습니다!")
            
            try:
                return self.torch_call(examples=features)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"⚠️ Processor 처리 중 오류: {e}")
                raise
    
    return CustomSFTDataCollator(processor, max_length)


# 도메인별 데이터셋 빌더 함수들
def math_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True):
    """수학 도메인 데이터셋"""
    log_memory_usage("수학 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"math": DOMAIN_DATASETS["math"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming
    )
    log_memory_usage("수학 도메인 데이터셋 완료")
    return dataset

def science_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True):
    """과학 도메인 데이터셋"""
    log_memory_usage("과학 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"science": DOMAIN_DATASETS["science"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming
    )
    log_memory_usage("과학 도메인 데이터셋 완료")
    return dataset

def code_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True):
    """코드 도메인 데이터셋"""
    log_memory_usage("코드 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"code": DOMAIN_DATASETS["code"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming
    )
    log_memory_usage("코드 도메인 데이터셋 완료")
    return dataset

def puzzle_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True):
    """퍼즐 도메인 데이터셋"""
    log_memory_usage("퍼즐 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"puzzle": DOMAIN_DATASETS["puzzle"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming
    )
    log_memory_usage("퍼즐 도메인 데이터셋 완료")
    return dataset

def vision_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True):
    """비전 도메인 데이터셋"""
    log_memory_usage("비전 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"vision": DOMAIN_DATASETS["vision"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming
    )
    log_memory_usage("비전 도메인 데이터셋 완료")
    return dataset

def ocr_domain_dataset(tokenizer, max_samples: int = 200, use_streaming: bool = True):
    """OCR 도메인 데이터셋"""
    log_memory_usage("OCR 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs={"ocr": DOMAIN_DATASETS["ocr"]},
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples,
        use_streaming=use_streaming
    )
    log_memory_usage("OCR 도메인 데이터셋 완료")
    return dataset

def all_domains_dataset(tokenizer, max_samples_per_domain: int = 200, use_streaming: bool = True):
    """모든 도메인 통합 데이터셋"""
    log_memory_usage("전체 도메인 데이터셋 시작")
    dataset = get_multi_domain_sft_dataset(
        domain_configs=DOMAIN_DATASETS,
        tokenizer=tokenizer,
        max_samples_per_domain=max_samples_per_domain,
        use_streaming=use_streaming
    )
    log_memory_usage("전체 도메인 데이터셋 완료")
    return dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    logger.info("🚀 멀티 도메인 데이터셋 테스트 시작")
    log_memory_usage("프로그램 시작")
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    log_memory_usage("토크나이저 로드 후")
    
    # 전체 도메인 데이터셋 테스트
    try:
        logger.info("📦 전체 도메인 데이터셋 테스트")
        dataset = all_domains_dataset(tokenizer, max_samples_per_domain=50, use_streaming=True)
        log_memory_usage("전체 도메인 데이터셋 생성 후")
        
        logger.info(f"데이터셋 생성 완료: {dataset}")
        
        # 도메인별 샘플 확인
        if 'train' in dataset:
            train_domains = {}
            for i in range(min(100, len(dataset['train']))):
                sample = dataset['train'][i]
                domain = sample.get('domain', 'unknown')
                train_domains[domain] = train_domains.get(domain, 0) + 1
            
            logger.info(f"Train 세트 도메인 분포: {train_domains}")
        
    except Exception as e:
        logger.error(f"전체 도메인 데이터셋 테스트 실패: {e}")
        traceback.print_exc()
    
    log_memory_usage("테스트 완료")
    logger.info("✅ 멀티 도메인 데이터셋 테스트 완료")

