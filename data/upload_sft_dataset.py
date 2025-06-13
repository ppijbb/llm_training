from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, Sequence, Image as ImageFeature
import json
from typing import List, Dict, Any
from tqdm import tqdm
import os
import requests
from PIL import Image
from io import BytesIO

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
    ("Lin-Chen/ShareGPT4V", "ShareGPT4V")
]

def construct_image_url(image_path, dataset_name):
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

def load_image_from_url_or_path(image_source, dataset_name=None):
    """
    URL이나 경로에서 실제 이미지를 로드합니다.
    """
    try:
        # 이미 PIL Image 객체인 경우
        if hasattr(image_source, 'size') and hasattr(image_source, 'convert'):
            return image_source.convert('RGB')
        
        # 문자열인 경우 (URL 또는 파일명)
        if isinstance(image_source, str):
            # HTTP/HTTPS URL인 경우
            if image_source.startswith('http://') or image_source.startswith('https://'):
                response = requests.get(image_source, timeout=15)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                return image.convert('RGB')
            
            # 로컬 파일 경로인 경우
            elif os.path.exists(image_source):
                image = Image.open(image_source)
                return image.convert('RGB')
            
            # 파일명만 있는 경우 - URL 구성 시도
            else:
                if dataset_name:
                    constructed_url = construct_image_url(image_source, dataset_name)
                    if constructed_url:
                        print(f"🔗 URL 구성 시도: {constructed_url}")
                        try:
                            response = requests.get(constructed_url, timeout=15)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content))
                            print(f"✅ 구성된 URL에서 이미지 로드 성공: {image.size}")
                            return image.convert('RGB')
                        except:
                            print(f"⚠️ 구성된 URL에서 로드 실패: {constructed_url}")
                
                print(f"⚠️ 이미지를 로드할 수 없음: {image_source}")
                return None
        
        # bytes 데이터인 경우
        elif isinstance(image_source, bytes):
            image = Image.open(BytesIO(image_source))
            return image.convert('RGB')
            
        else:
            return None
        
    except Exception as e:
        print(f"⚠️ 이미지 로드 실패: {e}")
        return None

def convert_to_target_format(sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    각 데이터셋의 샘플을 목표 형식으로 변환합니다.
    텍스트 전용 데이터셋과 멀티모달 데이터셋을 모두 처리합니다.
    목표 형식:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "질문", "index": null},
                    {"type": "image", "text": null, "index": 0}  # 멀티모달인 경우만
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": "답변", "index": null}
                ]
            }
        ],
        "images": [actual_image_object],  # 멀티모달인 경우만
        "source_dataset": "dataset_name",
        "original_data": {...}
    }
    """
    
    result = {
        "messages": [],
        "images": [],
        "source_dataset": dataset_name,
        "original_data": sample.copy()
    }
    
    try:
        # 텍스트 전용 데이터셋들 처리
        if dataset_name == "HuggingFaceTB/smoltalk":
            if "messages" in sample and isinstance(sample["messages"], list):
                for msg in sample["messages"]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        result["messages"].append({
                            "role": msg["role"],
                            "content": [{"type": "text", "text": msg["content"], "index": None}]
                        })
        
        elif dataset_name == "R0k1e/UltraLink":
            if "data" in sample and isinstance(sample["data"], list) and len(sample["data"]) >= 2:
                data = sample["data"]
                for i in range(0, len(data), 2):
                    if i + 1 < len(data):
                        result["messages"].extend([
                            {"role": "user", "content": [{"type": "text", "text": data[i], "index": None}]},
                            {"role": "assistant", "content": [{"type": "text", "text": data[i + 1], "index": None}]}
                        ])
        
        elif dataset_name == "PrincetonPLI/Instruct-SkillMix-SDD":
            if "instruction" in sample and "output" in sample:
                user_content = sample["instruction"]
                if "input" in sample and sample["input"].strip():
                    user_content += f"\n\nInput: {sample['input']}"
                
                result["messages"] = [
                    {"role": "user", "content": [{"type": "text", "text": user_content, "index": None}]},
                    {"role": "assistant", "content": [{"type": "text", "text": sample["output"], "index": None}]}
                ]
        
        elif dataset_name == "allenai/WildChat-1M":
            if "conversation" in sample and isinstance(sample["conversation"], list):
                for conv in sample["conversation"]:
                    if isinstance(conv, dict) and "role" in conv and "content" in conv:
                        result["messages"].append({
                            "role": conv["role"],
                            "content": [{"type": "text", "text": conv["content"], "index": None}]
                        })
        
        elif dataset_name == "nvidia/OpenCodeInstruct":
            if "input" in sample and "output" in sample:
                result["messages"] = [
                    {"role": "user", "content": [{"type": "text", "text": sample["input"], "index": None}]},
                    {"role": "assistant", "content": [{"type": "text", "text": sample["output"], "index": None}]}
                ]
        
        elif dataset_name == "microsoft/orca-agentinstruct-1M-v1":
            if "messages" in sample and isinstance(sample["messages"], list):
                for msg in sample["messages"]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        result["messages"].append({
                            "role": msg["role"],
                            "content": [{"type": "text", "text": msg["content"], "index": None}]
                        })
        
        elif "Nemotron" in dataset_name:
            if "conversations" in sample and isinstance(sample["conversations"], list):
                for conv in sample["conversations"]:
                    if isinstance(conv, dict) and "from" in conv and "value" in conv:
                        role = "user" if conv["from"] in ["human", "user"] else "assistant"
                        result["messages"].append({
                            "role": role,
                            "content": [{"type": "text", "text": conv["value"], "index": None}]
                        })
        
        elif dataset_name == "open-r1/Mixture-of-Thoughts":
            if "messages" in sample and isinstance(sample["messages"], list):
                for msg in sample["messages"]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        result["messages"].append({
                            "role": msg["role"],
                            "content": [{"type": "text", "text": msg["content"], "index": None}]
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
                print(f"🖼️ 이미지 로드 성공: {getattr(image_obj, 'size', 'unknown size')}")
            
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
                    text_content = conv.get("value", "")
                    
                    if text_content:
                        # <image> 태그 제거 (이미지는 별도 처리)
                        text_content = text_content.replace("<image>", "").strip()
                        
                        if text_content:  # 빈 문자열이 아닌 경우만
                            content_list.append({
                                "type": "text",
                                "text": text_content,
                                "index": None
                            })
                    
                    # 첫 번째 user 메시지에 이미지 추가
                    if role == "user" and i == 0 and result["images"]:
                        content_list.append({
                            "type": "image", 
                            "text": None,
                            "index": 0
                        })
                    
                    if content_list:  # content가 있는 경우만 추가
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
                print(f"🖼️ 이미지 로드 성공: {getattr(image_obj, 'size', 'unknown size')}")
            
            # caption을 대화 형식으로 변환
            caption = sample.get("caption", "").strip()
            if not caption:
                caption = sample.get("cogvlm_caption", "").strip()
            
            if caption:
                # 첫 번째 user 메시지에 이미지 포함
                user_content = [{"type": "text", "text": "Describe this image.", "index": None}]
                if result["images"]:
                    user_content.append({"type": "image", "text": None, "index": 0})
                
                result["messages"] = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": caption, "index": None}]}
                ]
        
        # 빈 messages인 경우 None 반환
        if not result["messages"]:
            return None
            
        return result
        
    except Exception as e:
        print(f"Error converting sample from {dataset_name}: {str(e)}")
        return None

def process_dataset(dataset_name: str, config_name: str = None, max_samples: int = None):
    """데이터셋을 처리하여 목표 형식으로 변환합니다."""
    try:
        print(f"\n🔄 처리 중: {dataset_name}")
        if config_name:
            print(f"   Config: {config_name}")
        
        # 특정 데이터셋들의 split 설정
        if dataset_name == "microsoft/orca-agentinstruct-1M-v1":
            split = "creative_content"  # 첫 번째 사용 가능한 split 사용
        elif dataset_name == "MaziyarPanahi/Llama-Nemotron-Post-Training-Dataset-v1-ShareGPT":
            split = "chat"  # chat split 사용
        elif dataset_name == "nvidia/Llama-Nemotron-Post-Training-Dataset":
            split = "chat"  # chat split 사용
        else:
            split = "train"
        
        # 데이터셋 로드 - 스트리밍 모드로 메모리 효율성 확보
        try:
            if config_name:
                full_dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
            else:
                full_dataset = load_dataset(dataset_name, split=split, streaming=True)
        except Exception as e:
            print(f"❌ 데이터셋 로드 실패 ({split} split): {e}")
            # train split으로 재시도
            if split != "train":
                try:
                    print(f"🔄 train split으로 재시도...")
                    if config_name:
                        full_dataset = load_dataset(dataset_name, config_name, split="train", streaming=True)
                    else:
                        full_dataset = load_dataset(dataset_name, split="train", streaming=True)
                except Exception as e2:
                    print(f"❌ train split으로도 실패: {e2}")
                    return
            else:
                return
        
        print(f"📊 스트리밍 모드로 데이터셋 로드 완료")

        processed_samples = []
        success_count = 0
        total_count = 0
        
        # 스트리밍 데이터 처리
        for sample in full_dataset:
            if max_samples and total_count >= max_samples:
                break
            
            total_count += 1
            
            # 변환 시도
            converted = convert_to_target_format(sample, dataset_name)
            if converted:
                processed_samples.append(converted)
                success_count += 1
                
                # 처음 몇 개 샘플에서 이미지 확인 (멀티모달 데이터셋의 경우)
                if success_count <= 3 and converted["images"]:
                    print(f"✅ {dataset_name}: {len(converted['images'])}개 이미지 포함")
            
            # 메모리 관리를 위한 배치 처리
            if len(processed_samples) >= 1000:  # 1000개씩 yield
                yield processed_samples
                processed_samples = []
                print(f"📊 {dataset_name}: {success_count}/{total_count} 샘플 처리 완료 (배치 yield)")
        
        # 남은 샘플들 처리
        if processed_samples:
            yield processed_samples
            
        print(f"✅ {dataset_name}: {success_count}/{total_count} 샘플 변환 완료")

    except Exception as e:
        print(f"❌ {dataset_name} 처리 중 오류: {str(e)}")

def merge_and_create_dataset(output_name: str = "unified-multimodal-sft", max_samples_per_dataset: int = None):
    """모든 멀티모달 데이터셋을 병합하고 목표 형식으로 생성합니다."""
    print("🚀 멀티모달 데이터셋 병합 시작...")
    
    all_samples = []
    
    for dataset_name, config_name in dataset_configs:
        try:
            for batch in process_dataset(dataset_name, config_name, max_samples_per_dataset):
                all_samples.extend(batch)
                print(f"📊 현재까지 수집된 샘플 수: {len(all_samples)}")
        except Exception as e:
            print(f"❌ {dataset_name} 처리 실패: {str(e)}")
            continue
    
    if not all_samples:
        print("❌ 변환된 샘플이 없습니다.")
        return None
    
    print(f"\n🎯 총 {len(all_samples)}개 샘플 변환 완료")
    
    # 데이터 검증
    print("\n=== 데이터 품질 검사 ===")
    valid_samples = 0
    invalid_samples = 0
    multimodal_samples = 0
    text_only_samples = 0

    for i, sample in enumerate(all_samples):
        messages = sample.get("messages", [])
        images = sample.get("images", [])

        # 메시지 형식 검증
        is_valid = True
        has_multimodal = False

        if not isinstance(messages, list) or len(messages) == 0:
            is_valid = False
        else:
            for msg in messages:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    is_valid = False
                    break
                if msg["role"] not in ["user", "assistant", "system"]:
                    is_valid = False
                    break

                # content 형식 검증
                content = msg.get("content", [])
                if not isinstance(content, list):
                    is_valid = False
                    break

                # content 내용 검증
                for content_item in content:
                    if not isinstance(content_item, dict) or "type" not in content_item:
                        is_valid = False
                        break

                    if content_item["type"] == "image":
                        has_multimodal = True
                    elif content_item["type"] == "text":
                        if "text" not in content_item:
                            is_valid = False
                            break
        
        # 이미지 배열이 있고 실제 이미지가 있는 경우도 멀티모달로 처리
        if images and len(images) > 0:
            has_multimodal = True

        if is_valid:
            valid_samples += 1
            if has_multimodal:
                multimodal_samples += 1
            else:
                text_only_samples += 1
        else:
            invalid_samples += 1

    print(f"✅ 유효한 샘플: {valid_samples}개")
    print(f"❌ 무효한 샘플: {invalid_samples}개")
    print(f"🖼️ 멀티모달 샘플: {multimodal_samples}개")
    print(f"📝 텍스트 전용 샘플: {text_only_samples}개")
    
    # HuggingFace Dataset feature 구조 정의
    # 텍스트 전용과 멀티모달을 모두 지원하기 위해 유연하게 구성
    print("📦 Dataset 객체 생성 중...")
    
    # Features를 명시적으로 정의하지 않고 자동 추론되도록 함
    # 이렇게 하면 텍스트 전용/멀티모달 데이터를 모두 처리 가능
    unified_dataset = Dataset.from_list(all_samples)
    
    # 로컬 저장
    print("💾 로컬 저장 중...")
    unified_dataset.save_to_disk(f"./{output_name}")
    
    # 허깅페이스 업로드 시도
    try:
        print("🚀 허깅페이스 업로드 시도...")
        unified_dataset.push_to_hub(output_name, private=False)
        print(f"✅ 성공적으로 {output_name}으로 업로드!")
    except Exception as e:
        print(f"⚠️ 업로드 실패: {str(e)}")
        print("💾 로컬 저장은 완료되었습니다.")
    
    return unified_dataset

def inspect_dataset(dataset_path: str = "./unified-multimodal-sft"):
    """생성된 데이터셋 검사"""
    try:
        print(f"🔍 데이터셋 검사: {dataset_path}")
        
        dataset = Dataset.load_from_disk(dataset_path)
        print(f"📊 총 샘플 수: {len(dataset)}")
        
        # 구조 검사
        sample_with_image = None
        sample_without_image = None
        
        for sample in dataset:
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
            
            # 첫 번째 메시지 구조 확인
            if sample_with_image['messages']:
                first_msg = sample_with_image['messages'][0]
                print(f"   첫 번째 메시지 role: {first_msg.get('role')}")
                print(f"   첫 번째 메시지 content 수: {len(first_msg.get('content', []))}")
                
                for j, content in enumerate(first_msg.get('content', [])[:3]):
                    content_type = content.get('type', 'unknown')
                    if content_type == 'text':
                        text_preview = content.get('text', '')[:50] + "..." if len(content.get('text', '')) > 50 else content.get('text', '')
                        print(f"     Content {j+1}: {content_type} - '{text_preview}'")
                    else:
                        print(f"     Content {j+1}: {content_type} - index: {content.get('index')}")
        
        # 통계
        image_count = sum(1 for sample in dataset if sample.get("images"))
        print(f"\n📈 이미지 포함 샘플: {image_count}/{len(dataset)} ({image_count/len(dataset)*100:.1f}%)")
        
        # 원본 데이터셋별 통계
        source_stats = {}
        for sample in dataset:
            source = sample.get("source_dataset", "unknown")
            source_stats[source] = source_stats.get(source, 0) + 1
        
        print(f"\n📊 원본 데이터셋별 분포:")
        for source, count in sorted(source_stats.items()):
            print(f"   {source}: {count}개 ({count/len(dataset)*100:.1f}%)")
        
        # 원본 데이터 보존 확인
        original_data_count = sum(1 for sample in dataset if sample.get("original_data"))
        print(f"\n💾 원본 데이터 보존: {original_data_count}/{len(dataset)} ({original_data_count/len(dataset)*100:.1f}%)")
        
        # 원본 데이터 예시 (첫 번째 샘플)
        if dataset[0].get("original_data"):
            print(f"\n🔍 원본 데이터 예시 (첫 번째 샘플):")
            original = dataset[0]["original_data"]
            print(f"   원본 데이터 키: {list(original.keys())}")
            for key, value in list(original.items())[:3]:  # 처음 3개 키만 표시
                if isinstance(value, str) and len(value) > 50:
                    print(f"   {key}: {value[:50]}...")
                else:
                    print(f"   {key}: {value}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 검사 중 오류: {str(e)}")
        return None

def main():
    """메인 함수"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "merge":
            if len(sys.argv) < 3:
                print("❌ 리포지토리 이름이 필요합니다!")
                print("사용법: python upload_sft_dataset.py merge <repository_name> [max_samples_per_dataset]")
                return
            
            repository_name = sys.argv[2]
            max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None
            
            print(f"🎯 타겟 리포지토리: {repository_name}")
            dataset = merge_and_create_dataset(output_name=repository_name, max_samples_per_dataset=max_samples)
            if dataset:
                print("🎉 병합 완료!")
                
        elif sys.argv[1] == "inspect":
            dataset_path = sys.argv[2] if len(sys.argv) > 2 else "./unified-multimodal-sft"
            inspect_dataset(dataset_path)
            
    else:
        print("사용법:")
        print("  python upload_sft_dataset.py merge <repository_name> [max_samples_per_dataset]")
        print("  python upload_sft_dataset.py inspect [dataset_path]")
        print("")
        print("📝 텍스트 + 멀티모달 통합 데이터셋 처리")
        print("포함된 데이터셋:")
        for dataset_name, config_name in dataset_configs:
            if config_name:
                print(f"  - {dataset_name} ({config_name})")
            else:
                print(f"  - {dataset_name}")
        print("")
        print("예시:")
        print("  python upload_sft_dataset.py merge my-unified-dataset 1000")
        print("  python upload_sft_dataset.py merge my-unified-dataset")  # 전체 데이터
        print("  python upload_sft_dataset.py inspect ./my-unified-dataset")

if __name__ == "__main__":
    main()