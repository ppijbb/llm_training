"""
TRL 표준 데이터 로더 for GRPO training
"""

import logging
from typing import Dict, Any, List, Optional
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoProcessor
import json

logger = logging.getLogger(__name__)

class GRPODataLoader:
    """TRL 표준 데이터 로더 for GRPO training"""

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit",
        max_length: int = 2048
    ):
        self.model_name = model_name
        self.max_length = max_length

        # Load tokenizer only (TRL handles the rest)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"✅ TRL DataLoader initialized with model: {model_name}")
    
    def load_dataset(
        self, 
        dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
        split: str = "train_prefs",
        max_samples: Optional[int] = None,
        streaming: bool = False
    ) -> DatasetDict:
        """
        Load dataset from HuggingFace Hub
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            split: Dataset split to load
            max_samples: Maximum number of samples to load
            streaming: Whether to use streaming mode
        """
        logger.info(f"📦 Loading dataset: {dataset_name}")
        
        try:
            if streaming:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
                if max_samples:
                    dataset = dataset.take(max_samples)
                return dataset
            else:
                dataset = load_dataset(dataset_name, split=split)
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                return dataset
                
        except Exception as e:
            logger.error(f"❌ Failed to load dataset {dataset_name}: {e}")
            raise
    
    def load_custom_dataset(self, data_path: str) -> DatasetDict:
        """
        Load custom dataset from local files
        
        Args:
            data_path: Path to the dataset file (JSON, JSONL, CSV, etc.)
        """
        logger.info(f"📁 Loading custom dataset from: {data_path}")
        
        try:
            if data_path.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=data_path)
            elif data_path.endswith('.json'):
                dataset = load_dataset('json', data_files=data_path)
            elif data_path.endswith('.csv'):
                dataset = load_dataset('csv', data_files=data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
                
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to load custom dataset: {e}")
            raise
    
    def prepare_grpo_data(self, dataset: Dataset) -> Dataset:
        """
        TRL 표준 데이터 형식으로 변환

        TRL GRPO는 다음 형식의 데이터를 기대합니다:
        - prompt/chosen/rejected 필드
        또는
        - messages 필드 (대화 형식)
        """
        logger.info("🔄 Converting to TRL standard format")

        def convert_to_trl_format(example):
            """Convert to TRL standard format"""
            # 이미 TRL 형식이면 그대로 반환
            if "prompt" in example and "chosen" in example and "rejected" in example:
                return example

            # UltraFeedback 형식 변환
            if "chosen" in example and "rejected" in example:
                chosen = example["chosen"]
                rejected = example["rejected"]

                if isinstance(chosen, list) and isinstance(rejected, list):
                    # chosen과 rejected가 리스트인 경우 (메시지 형식)
                    chosen_text = chosen[-1]["content"] if chosen else ""
                    rejected_text = rejected[-1]["content"] if rejected else ""
                    prompt = chosen[0]["content"] if chosen else ""

                    return {
                        "prompt": prompt,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }

            # 기본적으로 원본 반환 (TRL이 처리)
            return example

        # 데이터 변환
        processed_dataset = dataset.map(
            convert_to_trl_format,
            desc="Converting to TRL format"
        )

        logger.info(f"✅ Converted {len(processed_dataset)} samples to TRL format")
        return processed_dataset
    
    def get_sample_data(self, dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized") -> Dict[str, Any]:
        """
        샘플 데이터를 가져와서 TRL 형식 확인

        Args:
            dataset_name: 확인할 데이터셋 이름
        """
        logger.info(f"🔍 Getting sample data from {dataset_name}")

        try:
            # 작은 샘플 로드
            dataset = self.load_dataset(dataset_name, max_samples=5)

            # 첫 번째 샘플 반환
            if len(dataset) > 0:
                sample = dict(dataset[0])
                logger.info("✅ Sample data retrieved successfully")
                logger.info(f"📋 Sample keys: {list(sample.keys())}")
                return sample
            else:
                logger.warning("⚠️ No samples found in dataset")
                return {}

        except Exception as e:
            logger.error(f"❌ Failed to get sample data: {e}")
            return {}


def create_grpo_dataloader(
    model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit",
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    max_samples: int = 1000,
    max_length: int = 2048
) -> tuple[GRPODataLoader, Dataset]:
    """
    TRL 표준 데이터 로더 생성 및 데이터셋 로드

    Args:
        model_name: 모델 이름
        dataset_name: 데이터셋 이름
        max_samples: 최대 샘플 수
        max_length: 최대 시퀀스 길이

    Returns:
        (data_loader, dataset) 튜플
    """
    # 데이터 로더 생성
    data_loader = GRPODataLoader(
        model_name=model_name,
        max_length=max_length
    )

    # 데이터셋 로드 및 TRL 형식으로 변환
    dataset = data_loader.load_dataset(dataset_name, max_samples=max_samples)
    processed_dataset = data_loader.prepare_grpo_data(dataset)

    return data_loader, processed_dataset


