from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, Sequence, Image as ImageFeature
import json
from typing import List, Dict, Any
from tqdm.auto import tqdm
import os
import requests
from PIL import Image
from io import BytesIO
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()  # 진행 표시줄 비활성화

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
                        try:
                            response = requests.get(constructed_url, timeout=15)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content))
                            return image.convert('RGB')
                        except:
                            pass  # 조용히 실패 처리
                
                return None
        
        # bytes 데이터인 경우
        elif isinstance(image_source, bytes):
            image = Image.open(BytesIO(image_source))
            return image.convert('RGB')
            
        else:
            return None
        
    except Exception as e:
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
        # 특정 데이터셋들의 split 설정
        if dataset_name == "microsoft/orca-agentinstruct-1M-v1":
            split = "creative_content"
        elif dataset_name == "MaziyarPanahi/Llama-Nemotron-Post-Training-Dataset-v1-ShareGPT":
            split = "chat"
        elif dataset_name == "nvidia/Llama-Nemotron-Post-Training-Dataset":
            split = "chat"
        else:
            split = "train"
        
        # 데이터셋 로드
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

        processed_samples = []
        success_count = 0
        total_count = 0
        
        # 진행 상황 표시를 위한 tqdm 설정
        desc = f"{dataset_name.split('/')[-1]}"
        if config_name:
            desc += f"({config_name})"
        
        progress_bar = tqdm(desc=desc, unit="samples")
        
        # 스트리밍 데이터 처리
        for sample in full_dataset:
            if max_samples and total_count >= max_samples:
                break
            
            total_count += 1
            progress_bar.update(1)
            
            # 변환 시도
            converted = convert_to_target_format(sample, dataset_name)
            if converted:
                processed_samples.append(converted)
                success_count += 1
                
                # 이미지 로드 성공 시 진행바에 표시
                if converted["images"] and success_count <= 3:
                    progress_bar.set_postfix({"images": f"{len(converted['images'])}개"})
            
            # 메모리 관리를 위한 배치 처리
            if len(processed_samples) >= 1000:
                yield processed_samples
                processed_samples = []
                progress_bar.set_postfix({"processed": f"{success_count}/{total_count}"})
        
        progress_bar.close()
        
        # 남은 샘플들 처리
        if processed_samples:
            yield processed_samples
            
        tqdm.write(f"✅ {dataset_name}: {success_count}/{total_count} 샘플 변환 완료")

    except Exception as e:
        print(f"❌ {dataset_name} 처리 중 오류: {str(e)}")

def merge_and_create_dataset(output_name: str = "unified-multimodal-sft", max_samples_per_dataset: int = None):
    """모든 멀티모달 데이터셋을 병합하고 목표 형식으로 생성합니다."""
    print("🚀 멀티모달 데이터셋 병합 시작...")
    
    all_samples = []
    dataset_progress = tqdm(dataset_configs, desc="데이터셋 처리", unit="dataset")

    for dataset_name, config_name in dataset_progress:
        dataset_progress.set_description(f"처리중: {dataset_name.split('/')[-1]}")
        try:
            for batch in process_dataset(dataset_name, config_name, max_samples_per_dataset):
                all_samples.extend(batch)
                dataset_progress.set_postfix({"총 샘플": len(all_samples)})
        except Exception as e:
            print(f"❌ {dataset_name} 처리 실패: {str(e)}")
            continue

    dataset_progress.close()

    if not all_samples:
        print("❌ 변환된 샘플이 없습니다.")
        return None
    
    tqdm.write(f"\n🎯 총 {len(all_samples)}개 샘플 변환 완료")
    
    # 데이터 검증 (샘플링해서 빠르게)
    sample_size = min(1000, len(all_samples))
    valid_samples = 0
    image_samples = 0
    
    validation_progress = tqdm(range(sample_size), desc="데이터 검증", leave=False)
    for i in validation_progress:
        sample = all_samples[i]
        if "messages" in sample and "images" in sample:
            valid_samples += 1
            if sample["images"]:
                image_samples += 1
    
    tqdm.write(f"📋 샘플 검증 ({sample_size}개): {valid_samples}/{sample_size} 유효, {image_samples}/{sample_size} 이미지 포함")
    
    # Dataset 생성
    tqdm.write("📦 Dataset 객체 생성 중...")
    unified_dataset = Dataset.from_list(all_samples)

    # 로컬 저장
    tqdm.write("💾 로컬 저장 중...")
    unified_dataset.save_to_disk(f"./{output_name}")
    
    # 허깅페이스 업로드 시도
    try:
        tqdm.write("🚀 허깅페이스 업로드 시도...")
        
        # 업로드 전 데이터셋 정보 확인
        tqdm.write(f"   - 총 샘플 수: {len(unified_dataset):,}")
        tqdm.write(f"   - 컬럼: {list(unified_dataset.column_names)}")
        
        # push_to_hub 호출 - 더 나은 파라미터와 함께
        unified_dataset.push_to_hub(
            output_name, 
            private=False,
            max_shard_size="1GB",  # 샤드 크기 제한
            commit_message=f"Upload unified SFT dataset with {len(unified_dataset):,} samples"
        )
        
        tqdm.write(f"✅ 성공적으로 {output_name}으로 업로드!")
        tqdm.write(f"🔗 https://huggingface.co/datasets/{output_name}")
        
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
    