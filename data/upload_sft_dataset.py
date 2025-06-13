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
    목표 형식:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "질문", "index": null},
                    {"type": "image", "text": null, "index": 0}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": "답변", "index": null}
                ]
            }
        ],
        "images": [actual_image_object]
    }
    """
    
    result = {
        "messages": [],
        "images": []
    }
    
    # 이미지 추출 및 로드
    image_obj = None
    if "image" in sample and sample["image"] is not None:
        # dataset_name을 전달하여 URL 구성 가능하도록 함
        image_obj = load_image_from_url_or_path(sample["image"], dataset_name)
    elif "images" in sample and sample["images"] is not None:
        if isinstance(sample["images"], list) and len(sample["images"]) > 0:
            image_obj = load_image_from_url_or_path(sample["images"][0], dataset_name)
        else:
            image_obj = load_image_from_url_or_path(sample["images"], dataset_name)
    elif dataset_name == "Salesforce/blip3-kale" and "url" in sample:
        # blip3-kale은 url 필드에 이미지 URL이 있음
        image_obj = load_image_from_url_or_path(sample["url"], dataset_name)
    
    if image_obj is not None:
        result["images"].append(image_obj)
        print(f"🖼️ 이미지 로드 성공: {getattr(image_obj, 'size', 'unknown size')}")
    
    # 데이터셋별 conversations 처리
    conversations = None
    
    if dataset_name == "Lin-Chen/ShareGPT4V":
        conversations = sample.get("conversations", [])
    elif dataset_name == "liuhaotian/LLaVA-Instruct-150K":  
        conversations = sample.get("conversations", [])
    elif dataset_name == "Salesforce/blip3-kale":
        # blip3-kale은 caption 필드를 사용
        caption = sample.get("caption", "").strip()
        if caption:
            # caption을 assistant 응답으로 처리
            conversations = [
                {"from": "human", "value": "Describe this image."},
                {"from": "gpt", "value": caption}
            ]
        else:
            # cogvlm_caption 시도
            cogvlm_caption = sample.get("cogvlm_caption", "").strip()
            if cogvlm_caption:
                conversations = [
                    {"from": "human", "value": "Describe this image."},
                    {"from": "gpt", "value": cogvlm_caption}
                ]
    
    if not conversations:
        return None  # 변환할 수 없는 샘플
    
    # conversations를 messages로 변환
    for i, conv in enumerate(conversations):
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
    
    return result if result["messages"] else None

def process_dataset(dataset_name: str, config_name: str = None, max_samples: int = None):
    """데이터셋을 처리하여 목표 형식으로 변환합니다."""
    try:
        print(f"\n🔄 처리 중: {dataset_name}")
        if config_name:
            print(f"   Config: {config_name}")
        
        # 데이터셋 로드 - 작은 배치로 처리하여 메모리 효율성 확보
        try:
            if config_name:
                full_dataset = load_dataset(dataset_name, config_name, split="train")
            else:
                full_dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            print(f"❌ 데이터셋 로드 실패: {e}")
            return
        
        # 처리할 샘플 수 결정
        total_samples = len(full_dataset)
        samples_to_process = min(max_samples or total_samples, total_samples)
        
        print(f"📊 총 {total_samples}개 샘플 중 {samples_to_process}개 처리 예정")
        
        processed_samples = []
        success_count = 0
        
        # 작은 배치로 나누어 처리 (메모리 관리)
        batch_size = 100
        for start_idx in range(0, samples_to_process, batch_size):
            end_idx = min(start_idx + batch_size, samples_to_process)
            batch = full_dataset.select(range(start_idx, end_idx))
            
            print(f"🔄 배치 처리 중: {start_idx+1}-{end_idx}/{samples_to_process}")
            
            for sample in tqdm(batch, desc=f"Processing batch"):
                # 변환 시도
                converted = convert_to_target_format(sample, dataset_name)
                if converted:
                    processed_samples.append(converted)
                    success_count += 1
                    
                    # 처음 몇 개 샘플에서 이미지 확인
                    if success_count <= 3 and converted["images"]:
                        print(f"✅ {dataset_name}: {len(converted['images'])}개 이미지 포함")
                
                # 메모리 관리를 위한 중간 yield
                if len(processed_samples) >= 200:
                    yield processed_samples
                    processed_samples = []
            
            # 배치 처리 후 메모리 정리
            del batch
        
        if processed_samples:
            yield processed_samples
            
        print(f"✅ {dataset_name}: {success_count}/{count} 샘플 변환 완료")
        
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
    valid_samples = 0
    image_samples = 0
    
    for sample in all_samples[:100]:  # 처음 100개만 검증
        if "messages" in sample and "images" in sample:
            valid_samples += 1
            if sample["images"]:
                image_samples += 1
    
    print(f"📋 샘플 검증 (처음 100개): {valid_samples}/100 유효, {image_samples}/100 이미지 포함")
    
    # Dataset 생성 - 이미지 feature를 명시적으로 지정
    print("📦 Dataset 객체 생성 중...")
    
    # HuggingFace Image feature 구조 정의
    features = Features({
        "messages": Sequence({
            "role": Value("string"),
            "content": Sequence({
                "type": Value("string"),
                "text": Value("string"),
                "index": Value("int64")
            })
        }),
        "images": Sequence(ImageFeature())  # 이미지 feature 명시
    })
    
    unified_dataset = Dataset.from_list(all_samples, features=features)
    
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
        print("예시:")
        print("  python upload_sft_dataset.py merge my-multimodal-dataset 1000")
        print("  python upload_sft_dataset.py merge my-multimodal-dataset")  # 전체 데이터
        print("  python upload_sft_dataset.py inspect ./my-multimodal-dataset")

if __name__ == "__main__":
    main()