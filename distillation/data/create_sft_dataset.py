import os
import pandas as pd
from datasets import load_dataset # Hugging Face datasets 라이브러리
from huggingface_hub import HfApi, HfFolder, create_repo

# --- Hugging Face Hub 설정 ---
HF_REPO_ID = "YOUR_USERNAME/YOUR_DATASET_REPO_NAME" # TODO: 실제 값으로 변경! 예: "johndoe/my-sft-dataset"
HF_DATASET_FILENAME = "merged_sft_dataset" # Hub에 업로드될 파일명
# HF_TOKEN = "YOUR_HF_WRITE_TOKEN" # 스크립트 실행 환경에 따라 직접 토큰을 설정하거나, login()을 통해 인증

# --- 데이터셋 저장 경로 (로컬 임시 저장용) ---
TEMP_DATA_DIR = "./temp_data/" # Hugging Face Hub 업로드 전 임시 저장
MERGED_DATA_FILE_TEMP = os.path.join(TEMP_DATA_DIR, "merged_sft_dataset_temp.parquet")

# Push the model to Hugging Face Hub
repo_name = HF_DATASET_FILENAME
api = HfApi()
token = HfFolder.get_token()
# --- 데이터셋별 다운로드 및 전처리 함수 (예시) ---

def download_and_process_coco(limit=None):
    """COCO 2017 캡션 데이터를 다운로드하고 전처리하는 예시 함수"""
    print("Processing COCO dataset...")
    try:
        coco_train = load_dataset("HuggingFaceM4/COCO", "2017_captions", split="train", trust_remote_code=True)
        coco_val = load_dataset("HuggingFaceM4/COCO", "2017_captions", split="validation", trust_remote_code=True)

        processed_data = []

        def extract_coco_info(dataset_split, split_name):
            count = 0
            for item in dataset_split:
                if limit and count >= limit:
                    break
                for caption in item.get('captions', [{'raw': ''}]):
                    processed_data.append({
                        "source": "coco",
                        "image_id": str(item.get('image_id', '')),
                        "image_path": item.get('filepath', ''),
                        "text_input": None,
                        "text_output": caption.get('raw', ''),
                        "language": "en",
                        "modality": "image_text"
                    })
                count += 1
            print(f"Processed {count} items from COCO {split_name}")

        extract_coco_info(coco_train, "train")
        extract_coco_info(coco_val, "validation")

        return pd.DataFrame(processed_data)

    except Exception as e:
        print(f"Error processing COCO dataset: {e}")
        return pd.DataFrame()

def download_and_process_laion2b_en(sample_n=1000, shuffle=True):
    """LAION-2B-en 데이터셋의 일부를 샘플링하여 가져오는 예시 함수 (Hugging Face datasets 활용)"""
    print("Processing LAION-2B-en dataset...")
    try:
        dataset = load_dataset("laion/laion2B-en", split="train", streaming=True)
        
        processed_data = []
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        count = 0
        for item in dataset:
            if count >= sample_n:
                break
            processed_data.append({
                "source": "laion2b_en",
                "image_id": str(item.get('SAMPLE_ID', '')),
                "image_path": item.get('URL', ''),
                "text_input": None,
                "text_output": item.get('TEXT', ''),
                "language": "en",
                "modality": "image_text"
            })
            count += 1
        print(f"Processed {count} items from LAION-2B-en")
        return pd.DataFrame(processed_data)
    except Exception as e:
        print(f"Error processing LAION-2B-en: {e}")
        return pd.DataFrame()

def download_and_process_oscar_multi(languages=['en', 'ko', 'ja'], sample_n_per_lang=1000, shuffle=True):
    """OSCAR 다국어 데이터셋에서 지정된 언어의 텍스트를 샘플링하는 예시 함수"""
    print(f"Processing OSCAR dataset for languages: {languages}...")
    all_lang_data = []
    try:
        for lang in languages:
            print(f"Processing OSCAR for language: {lang}")
            try:
                dataset = load_dataset("oscar-corpus/OSCAR-2301",
                                       language=lang,
                                       split="train",
                                       streaming=True,
                                       trust_remote_code=True)
            except Exception:
                 dataset = load_dataset(f"oscar", f"unshuffled_deduplicated_{lang}", split="train", streaming=True, trust_remote_code=True)

            processed_data = []
            if shuffle:
                dataset = dataset.shuffle(buffer_size=10000)

            count = 0
            for item in dataset:
                if count >= sample_n_per_lang:
                    break
                text = item.get('text', '')
                if text and len(text.strip()) > 50:
                    processed_data.append({
                        "source": f"oscar_{lang}",
                        "image_id": None,
                        "image_path": None,
                        "text_input": text,
                        "text_output": None,
                        "language": lang,
                        "modality": "text_only"
                    })
                    count += 1
            all_lang_data.extend(processed_data)
            print(f"Processed {count} items from OSCAR {lang}")
        return pd.DataFrame(all_lang_data)
    except Exception as e:
        print(f"Error processing OSCAR dataset: {e}")
        return pd.DataFrame()

def download_and_process_alpaca(limit=None):
    """Stanford Alpaca 데이터셋을 다운로드하고 전처리하는 예시 함수"""
    print("Processing Alpaca dataset...")
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        processed_data = []
        count = 0
        for item in dataset:
            if limit and count >= limit:
                break
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output_text = item.get('output', '')

            full_input = instruction
            if input_text:
                full_input += "\n" + input_text

            processed_data.append({
                "source": "alpaca",
                "image_id": None,
                "image_path": None,
                "text_input": full_input,
                "text_output": output_text,
                "language": "en",
                "modality": "text_only"
            })
            count += 1
        print(f"Processed {count} items from Alpaca")
        return pd.DataFrame(processed_data)
    except Exception as e:
        print(f"Error processing Alpaca dataset: {e}")
        return pd.DataFrame()

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    # DATA_DIR이 현재 폴더를 의미하므로 별도 생성 로직 불필요
    # if not os.path.exists(DATA_DIR):
    #     os.makedirs(DATA_DIR)
    if not os.path.exists(TEMP_DATA_DIR):
        os.makedirs(TEMP_DATA_DIR)

    all_dataframes = []

    coco_df = download_and_process_coco(limit=1000)
    if not coco_df.empty:
        all_dataframes.append(coco_df)

    laion_df = download_and_process_laion2b_en(sample_n=1000)
    if not laion_df.empty:
        all_dataframes.append(laion_df)

    oscar_df = download_and_process_oscar_multi(languages=['en', 'ko', 'ja'], sample_n_per_lang=1000)
    if not oscar_df.empty:
        all_dataframes.append(oscar_df)

    alpaca_df = download_and_process_alpaca(limit=1000)
    if not alpaca_df.empty:
        all_dataframes.append(alpaca_df)

    if all_dataframes:
        print("\nMerging all datasets...")
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Total processed items: {len(merged_df)}")
        print("Sample of merged data:")
        print(merged_df.sample(min(5, len(merged_df))))

        # --- 임시 파일로 저장 ---
        try:
            merged_df.to_parquet(MERGED_DATA_FILE_TEMP, index=False)
            print(f"Merged dataset temporarily saved to: {MERGED_DATA_FILE_TEMP}")

            # --- Hugging Face Hub에 업로드 ---
            print(f"\nUploading to Hugging Face Hub repository: {HF_REPO_ID}...")
            
            # HfApi 인스턴스 생성 (필요시 token 전달)
            # token = HfFolder.get_token() # 로컬에 저장된 토큰 가져오기
            # if token is None and HF_TOKEN:
            #     token = HF_TOKEN
            # api = HfApi(token=token)
            api = HfApi() # huggingface-cli login으로 인증된 경우 토큰 불필요

            # 리포지토리 생성 (이미 존재하면 무시)
            try:
                create_repo(HF_REPO_ID, repo_type="dataset", exist_ok=True)
                print(f"Repository {HF_REPO_ID} ensured.")
            except Exception as e_repo:
                print(f"Could not create or access repository {HF_REPO_ID}: {e_repo}")
                print("Please ensure the repository exists or you have rights to create it.")
                exit()

            # 파일 업로드
            api.upload_file(
                path_or_fileobj=MERGED_DATA_FILE_TEMP,
                path_in_repo=HF_DATASET_FILENAME,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"Upload merged SFT dataset: {HF_DATASET_FILENAME}",
            )
            print(f"Successfully uploaded {HF_DATASET_FILENAME} to {HF_REPO_ID}")

            # 임시 파일 삭제 (선택적)
            # os.remove(MERGED_DATA_FILE_TEMP)
            # print(f"Temporary file {MERGED_DATA_FILE_TEMP} removed.")

        except Exception as e:
            print(f"An error occurred during saving or uploading: {e}")
    else:
        print("No dataframes to merge. Please check data processing steps.")

    print("\nScript finished.") 