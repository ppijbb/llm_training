"""
Data loader for GRPO training with HuggingFace datasets
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoProcessor
import torch
from torch.utils.data import DataLoader
import json
import os

logger = logging.getLogger(__name__)

class GRPODataLoader:
    """Data loader for GRPO training with HuggingFace datasets"""
    
    def __init__(
        self,
        model_name: str = "unsloth/llama-3.1-8b-bnb-4bit",
        max_length: int = 2048,
        batch_size: int = 4,
        num_workers: int = 4,
        use_processor: bool = False
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_processor = use_processor
        
        # Load tokenizer or processor
        if use_processor:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = None
            
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"✅ DataLoader initialized with model: {model_name}")
    
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
    
    def prepare_grpo_data(self, dataset: Dataset, dataset_type: str = "ultrafeedback") -> Dataset:
        """
        Prepare dataset for GRPO training
        
        Args:
            dataset: Input dataset
            dataset_type: Type of dataset (ultrafeedback, custom, etc.)
        """
        logger.info(f"🔄 Preparing GRPO data for {dataset_type} dataset")
        
        def process_ultrafeedback(example):
            """Process UltraFeedback dataset format"""
            chosen = example.get("chosen", [])
            rejected = example.get("rejected", [])
            
            if not chosen or not rejected:
                return None
                
            # Extract messages from chosen and rejected responses
            chosen_messages = chosen[-1]["content"] if chosen else ""
            rejected_messages = rejected[-1]["content"] if rejected else ""
            
            return {
                "prompt": chosen[0]["content"] if chosen else "",
                "chosen": chosen_messages,
                "rejected": rejected_messages,
                "messages": [
                    {"role": "user", "content": chosen[0]["content"] if chosen else ""},
                    {"role": "assistant", "content": chosen_messages}
                ]
            }
        
        def process_custom(example):
            """Process custom dataset format"""
            # Assume custom format has prompt, chosen, rejected fields
            return {
                "prompt": example.get("prompt", ""),
                "chosen": example.get("chosen", ""),
                "rejected": example.get("rejected", ""),
                "messages": [
                    {"role": "user", "content": example.get("prompt", "")},
                    {"role": "assistant", "content": example.get("chosen", "")}
                ]
            }
        
        # Choose processing function based on dataset type
        if dataset_type == "ultrafeedback":
            process_func = process_ultrafeedback
        else:
            process_func = process_custom
        
        # Process dataset
        processed_dataset = dataset.map(
            process_func,
            remove_columns=dataset.column_names,
            desc="Processing GRPO data"
        )
        
        # Filter out None values
        processed_dataset = processed_dataset.filter(lambda x: x is not None)
        
        logger.info(f"✅ Processed {len(processed_dataset)} samples for GRPO training")
        return processed_dataset
    
    def create_dataloader(
        self, 
        dataset: Dataset, 
        shuffle: bool = True,
        collate_fn: Optional[callable] = None
    ) -> DataLoader:
        """
        Create PyTorch DataLoader
        
        Args:
            dataset: Processed dataset
            shuffle: Whether to shuffle data
            collate_fn: Custom collate function
        """
        if collate_fn is None:
            collate_fn = self._default_collate_fn
            
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        logger.info(f"✅ DataLoader created with batch_size={self.batch_size}")
        return dataloader
    
    def _default_collate_fn(self, batch):
        """Default collate function for GRPO training"""
        # This is a placeholder - actual implementation depends on the model
        return batch
    
    def get_sample_data(self, dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized") -> Dict[str, Any]:
        """
        Get sample data for testing
        
        Args:
            dataset_name: Dataset name to sample from
        """
        logger.info(f"🔍 Getting sample data from {dataset_name}")
        
        try:
            # Load small sample
            dataset = self.load_dataset(dataset_name, max_samples=10)
            
            # Process for GRPO
            processed = self.prepare_grpo_data(dataset, "ultrafeedback")
            
            # Return first sample
            if len(processed) > 0:
                sample = processed[0]
                logger.info("✅ Sample data retrieved successfully")
                return sample
            else:
                logger.warning("⚠️ No samples found in dataset")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to get sample data: {e}")
            return {}


def create_grpo_dataloader(
    model_name: str = "unsloth/llama-3.1-8b-bnb-4bit",
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    max_samples: int = 1000,
    batch_size: int = 4,
    max_length: int = 2048,
    use_processor: bool = False
) -> tuple[GRPODataLoader, Dataset]:
    """
    Create GRPO data loader and load dataset
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        max_samples: Maximum number of samples
        batch_size: Batch size for training
        max_length: Maximum sequence length
        use_processor: Whether to use processor instead of tokenizer
    
    Returns:
        Tuple of (data_loader, dataset)
    """
    # Create data loader
    data_loader = GRPODataLoader(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        use_processor=use_processor
    )
    
    # Load dataset
    dataset = data_loader.load_dataset(dataset_name, max_samples=max_samples)
    
    # Prepare for GRPO
    processed_dataset = data_loader.prepare_grpo_data(dataset, "ultrafeedback")
    
    return data_loader, processed_dataset


