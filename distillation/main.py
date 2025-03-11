from modules.trainer import train_distillation
from model.g2moe_model import G2MoEForCausalLM
from model.g2moe_config import G2MoEConfig
from accelerate import PartialState
from transformers import AutoTokenizer
import torch
import os

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} M"
    else:
        return str(number)
    
if __name__ == "__main__":
    train_distillation()