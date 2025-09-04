import os
from datasets import load_dataset
from transformers import AutoTokenizer
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

os.environ["HF_DATASETS_CACHE"] = "/mnt/disks/local-ssd/datasets_cache"

def process_content(contents_list):
    processed_contents = []
    for index, content in enumerate(contents_list["type"]):
        key = "image" if content == "image" else "text"
        if key == "image" and contents_list['text'][index] is None:
            continue
        processed_contents.append({"type": key, key: contents_list[key][index]})
    return processed_contents

total_tokens = 0

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
            conversation = [{
                "role": r, 
                "content": process_content(c)
                } 
                for r, c in zip(conversation["role"], conversation["content"])]
            result = tokenizer.apply_chat_template(conversation, tokenize=True)

            tokens_by_seq = len(result)
            assert bool(result), "Tokenization failed"
            global total_tokens
            total_tokens += tokens_by_seq
        else:
            return None # Skip this example
    
    # If the conversion from malformed dict to list of dicts failed or was not applicable,
    # and it's still not a list, return None.
    if not isinstance(conversation, list):
        return None

    # Return a dictionary to update the dataset instead of just the conversation list
    return {
        "messages": conversation
    }

tokenizer = AutoTokenizer.from_pretrained("Gunulhona/Gemma-3-4B")
data = load_dataset("Gunulhona/open_m_3")
data['train'] = data['train'].map(processing, batched=False, fn_kwargs={"tokenizer": tokenizer},)
print("Processed total tokens: ", total_tokens)
test = data['train'].select(range(3))
print(test['messages'])
data.push_to_hub("Gunulhona/open_m_3")
