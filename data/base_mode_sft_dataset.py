from datasets import load_dataset
from transformers import AutoProcessor
import torch
import re
import json

def get_dataset(
    dataset_name: str = "Gunulhona/open_m_3",
    tokenizer=None,
    max_length: int = 2048,
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
                truncation=True
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
    return {
        "input_ids": input_ids_list,
        "labels": input_ids_list.copy()  # For language model fine-tuning, labels are the same as input_ids
    }

if __name__ == "__main__":  
    dataset = get_dataset(tokenizer=AutoProcessor.from_pretrained("google/gemma-3-4b-it"))
    print(dataset)