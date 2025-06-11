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
    
    # Process the dataset
    try:
        processed_dataset = dataset.map(
            processing, 
            batched=True, 
            num_proc=8 if not streaming else None,
            fn_kwargs={
                "tokenizer": tokenizer, 
                "max_length": max_length,
                "text_only": text_only
            },
            remove_columns=dataset["train"].column_names if not streaming else None
        )
    except Exception as e:
        print(f"Error during dataset processing: {e}")
        print("Falling back to single-threaded processing...")
        processed_dataset = dataset.map(
            processing, 
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer, 
                "max_length": max_length,
                "text_only": text_only
            },
            remove_columns=dataset["train"].column_names if not streaming else None
        )
    
    # Split into train/test if not streaming
    if not streaming:
        processed_dataset = processed_dataset.train_test_split(test_size=test_size)
    
    return processed_dataset

def processing(
    examples,
    tokenizer,
    max_length: int = 2048,
    text_only: bool = False  # This argument is kept for compatibility but is not used
):
    """
    Applies the chat template directly to each conversation in the 'messages' column.

    This simplified function assumes the tokenizer's `apply_chat_template` can
    natively handle the dataset's message structure, including the nested list in
    the 'content' field. It iterates through each conversation, applies the
    template, and tokenizes the result. Malformed conversations that cause
    an error during templating are skipped.
    """
    prompts = []
    for conversation in examples["messages"]:
        # A conversation should be a list of message dictionaries.
        # Skip any entries that are not lists (e.g., None or other malformed data).
        if not isinstance(conversation, list):
            continue
        
        try:
            # Directly apply the chat template to the conversation.
            # The tokenizer is expected to correctly process the format:
            # [{'role': 'user', 'content': [{'type': 'text', 'text': '...'}]}]
            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False  # Crucial for SFT
            )
            prompts.append(prompt)
        except Exception:
            # If a conversation is malformed and causes an error during templating,
            # skip it and continue with the rest of the batch.
            print(f"Skipping conversation due to error: {e}")
            print(f"Conversation: {conversation}")
            continue
    print(prompts)
    # Tokenize the entire batch of valid, formatted prompt strings.
    tokenized_outputs = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,  # The DataCollator will handle padding.
    )

    # For language model fine-tuning, the `labels` are the `input_ids`.
    # The trainer will handle shifting them for next-token prediction.
    tokenized_outputs["labels"] = tokenized_outputs["input_ids"][:]

    return tokenized_outputs

if __name__ == "__main__":  
    dataset = get_dataset(tokenizer=AutoProcessor.from_pretrained("google/gemma-3-4b-it"))
    print(dataset)