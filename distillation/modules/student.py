import torch

import torch.nn as nn
import torch.nn.functional as F

class StudentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def distillation_loss(self, student_output, teacher_output, temperature=2.0):
        student_log_probs = F.log_softmax(student_output / temperature, dim=1)
        teacher_probs = F.softmax(teacher_output / temperature, dim=1)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        return loss
    
from model.g2moe_model import G2MoEForCausalLM
from model.g2moe_config import G2MoEConfig

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} M"
    else:
        return str(number)
    

test_model = G2MoEForCausalLM(
    config=G2MoEConfig(
        **{
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 6400,
            "max_position_embeddings": 131072,
            "model_type": "g2moe",
            "num_attention_heads": 32,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "num_local_experts": 16,
            "original_max_position_embeddings": 4096,
            # "input_jitter_noise": 0.01, # No such argument
        }))
format_parameters(test_model.num_parameters())
