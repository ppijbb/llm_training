# orjson is a faster JSON library. Install with: pip install orjson
try:
    import orjson as json
except ImportError:
    print("orjson not found, falling back to the standard json library. For faster parsing, run 'pip install orjson'")
    import json

from datasets import Dataset, Features, Value, Sequence
from tqdm.auto import tqdm

def generate_cleaned_records(file_path: str):
    """
    Reads a JSONL file line-by-line, cleans the data, and yields records.
    This generator approach is highly memory-efficient.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # tqdm will show processing speed (it/s) without a total count,
        # which avoids reading the file twice.
        for line in tqdm(f, desc="Streaming and cleaning records"):
            try:
                record = json.loads(line)
                
                # Clean the 'messages' field in-place for efficiency
                if 'messages' in record and isinstance(record['messages'], list):
                    for message in record['messages']:
                        if 'content' in message and isinstance(message['content'], list):
                            for content_item in message['content']:
                                # Fix 1: Ensure 'index' is always an integer (None -> -1)
                                if content_item.get('index') is None:
                                    content_item['index'] = -1
                                
                                # Fix 2: Ensure 'text' is always a string (None -> "")
                                if content_item.get('text') is None:
                                    content_item['text'] = ""

                yield record

            except (json.JSONDecodeError, TypeError):
                print(f"Skipping malformed line: {line.strip()}")

# Define the final, clean schema for the dataset
features = Features({
    'messages': Sequence(
        Features({
            'role': Value('string'),
            'content': Sequence(
                Features({
                    'type': Value('string'),
                    'text': Value('string'),
                    'index': Value('int64')
                })
            )
        })
    ),
    'images': Sequence(Value('string')),
    'source_dataset': Value('string'),
    'original_data': Value('string')
})

# Path to your data file
jsonl_path = "/mnt/disks/local-ssd/open_m_3_staging/data.jsonl"

# Create the dataset from the generator.
# This is highly memory-efficient as it processes the file as a stream
# and doesn't load the entire dataset into RAM.
print("Creating dataset from generator...")
dataset = Dataset.from_generator(
    generate_cleaned_records,
    features=features,
    gen_kwargs={"file_path": jsonl_path}
)

print("\nDataset loaded successfully using a memory-efficient generator!")
print(dataset)

# dataset_dict = DatasetDict.load_from_disk("/mnt/disks/local-ssd/open_m_3_staging")
# load_dataset("/mnt/disks/local-ssd/open_m_3_staging")
