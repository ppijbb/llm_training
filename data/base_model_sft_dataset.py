import logging
from datasets import load_dataset
from transformers import AutoProcessor
import torch
import re
import json
from PIL import Image
import io
from typing import List, Dict, Any
import base64
from datasets import DatasetDict

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def process_vision_info(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    """Placeholder for extracting image information from messages.
    This function needs to be implemented based on the actual structure of image data.
    For now, it returns an empty list.
    """
    images = []
    for message in messages:
        if 'image' in message and message['image'] is not None:
            try:
                # Assuming message['image'] is base64 encoded string or bytes
                # This part needs actual implementation based on data format
                image_bytes = base64.b64decode(message['image']) if isinstance(message['image'], str) else message['image']
                images.append(Image.open(io.BytesIO(image_bytes)))
            except Exception as e:
                logger.warning(f"Could not load image from message: {e}")
    return images

def create_multimodal_collate_fn(processor):
    """
    Create a collate function for multimodal data that handles both images and text.
    Based on TRL documentation for VLM SFT training.
    """
    def collate_fn(examples):
        # Extract messages from examples
        if "messages" in examples[0]:
            messages_list = [example["messages"] for example in examples]
            
            # Apply chat template to get text
            texts = []
            images_list = []
            
            for messages in messages_list:
                # Apply chat template
                text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                ).strip()
                texts.append(text)
                
                # Extract images from messages
                images = process_vision_info(messages)
                images_list.append(images)
            
            # Process texts and images together
            batch = processor(
                text=texts, 
                images=images_list, 
                return_tensors="pt", 
                padding=True
            )
            
            # The labels are the input_ids, and we mask the padding tokens in the loss computation
            labels = batch["input_ids"].clone()
            
            # Get special token IDs for masking
            pad_token_id = processor.tokenizer.pad_token_id
            
            # Mask padding tokens
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            
            # Mask image tokens if they exist
            try:
                # Try to get image token ID
                if hasattr(processor.tokenizer, 'special_tokens_map') and 'boi_token' in processor.tokenizer.special_tokens_map:
                    image_token_id = processor.tokenizer.convert_tokens_to_ids(
                        processor.tokenizer.special_tokens_map["boi_token"]
                    )
                    labels[labels == image_token_id] = -100
                
                # Mask image soft tokens (gemma specific)
                labels[labels == 262144] = -100
                
            except Exception as e:
                logger.warning(f"Warning: Could not mask image tokens: {e}")
            
            batch["labels"] = labels
            return batch
        elif "input_ids" in examples[0]:
            # input_ids와 attention_mask 추출
            input_ids = [torch.tensor(example["input_ids"]) for example in examples if "input_ids" in example]
            attention_mask = [torch.tensor(example["attention_mask"]) for example in examples if "attention_mask" in example]
            
            if not input_ids:
                return None
            
            # 패딩 처리
            from torch.nn.utils.rnn import pad_sequence
            
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
            
            # labels는 input_ids와 동일 (causal LM)
            labels = input_ids.clone()
            
            # 패딩 토큰은 loss 계산에서 제외
            pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
            labels[labels == pad_token_id] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            raise ValueError("Unknown dataset format")
    
    return collate_fn

def get_dataset(
    dataset_name: str = "Gunulhona/open_m_3",
    tokenizer=None,
    max_length: int = 131072,
    test_size: float = 0.1,
    text_only: bool = False,
    streaming: bool = False
) -> DatasetDict:
    """
    Load and process the open_m_3 dataset for SFT training.
    
    Args:
        dataset_name: Dataset name on Hugging Face Hub
        tokenizer: Tokenizer to use for processing
        max_length: Maximum sequence length
        test_size: Fraction of data to use for testing
        text_only: If True, only process text content (ignore images)
        streaming: If True, use streaming mode for large datasets
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")
    
    logger.info(f"📦 로딩 중: {dataset_name}, streaming: {streaming}")

    # Load dataset
    loaded_dataset = load_dataset(dataset_name, streaming=streaming)

    processed_dataset = {}

    if streaming:
        # For streaming datasets, we directly process the available splits.
        # loaded_dataset is an IterableDatasetDict or IterableDataset.
        if isinstance(loaded_dataset, dict): # It's an IterableDatasetDict
            splits_to_process = loaded_dataset
        else: # It's a single IterableDataset (e.g., if split='train' was used in load_dataset outside this function)
            splits_to_process = {"train": loaded_dataset} # Treat it as a 'train' split
        
        logger.info(f"📊 스트리밍 데이터셋 처리: {list(splits_to_process.keys())}")
        for split_name, split_data in splits_to_process.items():
            try:
                # Streaming map does not support num_proc directly like non-streaming
                processed_split = split_data.map(
                    processing,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "max_length": max_length,
                        "text_only": text_only
                    },
                    remove_columns=split_data.column_names if hasattr(split_data, 'column_names') else None
                )
                processed_split["conversations"] = processed_split["messages"].to_list()
                processed_dataset[split_name] = processed_split
                logger.info(f"   ✅ {split_name} 스트리밍 처리 완료")
            except Exception as e:
                logger.error(f"❌ {split_name} 스트리밍 처리 실패: {e}")
                raise # Re-raise for streaming as fallback to single-threaded map might not be meaningful

    else: # Not streaming
        # For non-streaming datasets, we expect a DatasetDict and perform train_test_split.
        if "train" not in loaded_dataset:
            raise ValueError("Non-streaming dataset must contain a 'train' split to perform train_test_split.")
        
        splits_for_training = loaded_dataset["train"].train_test_split(test_size=test_size)
        
        # Add other splits (validation, test etc.) if they exist in the original dataset
        for key, value in loaded_dataset.items():
            if key != "train":
                splits_for_training[key] = value

        logger.info(f"📊 비스트리밍 데이터셋 분할: 훈련/테스트 및 기타")
        for split_name, split_data in splits_for_training.items():
            try:
                processed_split = split_data.map(
                    processing,
                    batched=False,
                    num_proc=8, # Use num_proc for non-streaming
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "max_length": max_length,
                        "text_only": text_only
                    },
                    remove_columns=split_data.column_names
                )
                processed_dataset[split_name] = processed_split
                logger.info(f"   ✅ {split_name} 처리 완료")
            except Exception as e:
                logger.error(f"❌ {split_name} 처리 실패: {e}")
                logger.info("   🔄 싱글 스레드 처리로 폴백...")
                processed_split = split_data.map(
                    processing,
                    batched=False,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "max_length": max_length,
                        "text_only": text_only
                    },
                )
                processed_dataset[split_name] = processed_split
                logger.info(f"   ✅ {split_name} 싱글 스레드 처리 완료")

    if not processed_dataset:
        raise RuntimeError("데이터셋 처리 실패: 어떤 스플릿도 처리되지 않았습니다.")

    return DatasetDict(processed_dataset)

def process_content(contents_list):
    processed_contents = []
    for index, content in enumerate(contents_list["type"]):
        key = "image" if content == "image" else "text"
        if key == "image" and contents_list['text'][index] is None:
            continue
        processed_contents.append({"type": key, key: contents_list[key][index]})
    return processed_contents

def processing(
    example, # Changed from 'examples' to 'example' to reflect single row
    tokenizer,
    max_length: int = 2048,
    text_only: bool = False
):
    """
    Applies the chat template directly to each conversation in the 'messages' column.
    The apply_chat_template function handles both templating and tokenization.
    """
    # A single example is passed, extract its messages
    conversation = example["messages"]
    images = example["images"]
    # A conversation should be a list of message dictionaries.
    # Skip if not a list, or if malformed (e.g., None).
    if not isinstance(conversation, list):
        # This part of the logic seems to try and fix malformed 'conversation'
        # if it's a dict with 'role' and 'content' keys that are lists.
        # This might be specific to the dataset format.
        if isinstance(conversation, dict) and "role" in conversation and "content" in conversation:
            try:
                conversation = [{
                    "role": r, 
                    "content": process_content(c)
                    } 
                    for r, c in zip(conversation["role"], conversation["content"])]
            except Exception as e:
                logger.warning(f"Malformed conversation dict during conversion: , Error: {e}")
                return None # Return None if conversion fails
        else:
            logger.warning(f"Skipping example due to malformed messages (not a list or expected dict format): ")
            return None # Skip this example
    
    # If the conversion from malformed dict to list of dicts failed or was not applicable,
    # and it's still not a list, return None.
    if not isinstance(conversation, list):
        return None

    try:
        # apply_chat_template with tokenize=True returns token IDs directly
        token_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,  # Crucial for SFT
            max_length=max_length,
            truncation=True,
            return_dict=True, 
            return_tensors="pt"
        )
        assert token_ids is not None, f"Token IDs is None for conversation: {conversation}"
        
        # Return the tokenized result directly (not a list of results for single example)
        # The map function expects a dict from the processing function if not batched.
        # So return the result in the format { "input_ids": [...], "attention_mask": [...] }
        result = {}
        for key, value in token_ids.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze().tolist()
            else:
                result[key] = value
        
        return result

    except Exception as e:
        # If a conversation is malformed and causes an error during templating,
        # skip it and continue with the rest of the batch.
        logger.warning(f"Skipping example due to error during tokenization: {e}")
        logger.warning(f"Conversation that caused error: {conversation}")
        return None # Return None to indicate this example should be skipped


if __name__ == "__main__":
    tokenizer = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    with open("/home/conan_jung/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
        chat_template = f.read()
    tokenizer.chat_template = chat_template
    dataset = get_dataset(tokenizer=tokenizer)
