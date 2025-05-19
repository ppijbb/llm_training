import torch
from torch import nn
from torch.optim import Adam

from datasets import Dataset
from trl import GKDConfig, GKDTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from .data_module import DistillationDataModule


NUM_DUMMY_SAMPLES = 100

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# The model to optimise
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# The teacher model to calculate the KL divergence against
teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

train_dataset = Dataset.from_dict(
    {
        "messages": [
            [
                {"role": "user", "content": "Hi, how are you?"},
                {"role": "assistant", "content": "I'm great thanks"},
            ]
        ]
        * NUM_DUMMY_SAMPLES
    }
)
eval_dataset = Dataset.from_dict(
    {
        "messages": [
            [
                {"role": "user", "content": "What colour is the sky?"},
                {"role": "assistant", "content": "The sky is blue"},
            ]
        ]
        * NUM_DUMMY_SAMPLES
    }
)

training_args = GKDConfig(output_dir="gkd-model", per_device_train_batch_size=1)
trainer = GKDTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
