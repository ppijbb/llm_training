from datasets import load_dataset
from transformers import AutoProcessor
import torch
import re
import json
from PIL import Image
import io

def create_multimodal_collate_fn(processor):
    """
    Create a collate function for multimodal data that handles both images and text.
    Based on TRL documentation for VLM SFT training.
    """
    def collate_fn(examples):
        # Extract messages from examples
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
            print(f"Warning: Could not mask image tokens: {e}")
        
        batch["labels"] = labels
        return batch
    
    return collate_fn

def get_dataset(
    dataset_name: str = "Gunulhona/open_m_3",
    tokenizer=None,
    max_length: int = 16384,
    test_size: float = 0.1,
    text_only: bool = False,
    streaming: bool = False
):
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
    
    # Load dataset
    dataset = load_dataset(dataset_name, streaming=streaming)

    # Split into train/test first if not streaming
    if not streaming:
        dataset = dataset["train"].train_test_split(test_size=test_size)
    
    # Process each split separately
    processed_dataset = {}
    for split_name, split_data in dataset.items():
        try:
            processed_split = split_data.map(
                processing, 
                batched=False, 
                num_proc=1,
                fn_kwargs={
                    "tokenizer": tokenizer, 
                    "max_length": max_length,
                    "text_only": text_only
                },
                remove_columns=split_data.column_names
            )
            processed_dataset[split_name] = processed_split
        except Exception as e:
            print(f"Error during {split_name} processing: {e}")
            print("Falling back to single-threaded processing...")
            processed_split = split_data.map(
                processing, 
                batched=False,
                fn_kwargs={
                    "tokenizer": tokenizer, 
                    "max_length": max_length,
                    "text_only": text_only
                },
                # remove_columns=split_data.column_names
            )
            processed_dataset[split_name] = processed_split
    
    # Convert back to DatasetDict
    from datasets import DatasetDict
    return DatasetDict(processed_dataset)

def processing(
    examples,
    tokenizer,
    max_length: int = 2048,
    text_only: bool = False  # This argument is kept for compatibility but is not used
):
    """
    Applies the chat template directly to each conversation in the 'messages' column.
    The apply_chat_template function handles both templating and tokenization.
    """
    input_ids_list = []
    
    for index, conversation in enumerate(examples["messages"]):
        # A conversation should be a list of message dictionaries.
        # Skip any entries that are not lists (e.g., None or other malformed data).
        if not isinstance(conversation, list):
            continue
        
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
            input_ids_list.append(token_ids)
        except Exception as e:
            # If a conversation is malformed and causes an error during templating,
            # skip it and continue with the rest of the batch.
            print(f"Skipping conversation due to error: [{index}] {e}")
            print(f"Conversation: {conversation}")
            continue
    
    # Return the tokenized results directly
    return [{k:v for k,v in data.items()} for data in input_ids_list]


if __name__ == "__main__":
    tokenizer = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    with open("/home/conan_jung/workspace/llm_training/sft/config/chat_template.txt", "r") as f:
        chat_template = f.read()
    tokenizer.chat_template = chat_template
    dataset = get_dataset(tokenizer=tokenizer)
    print(dataset)
