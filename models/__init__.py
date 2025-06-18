"""
G3MoE Models module

This module contains the G3MoE (Generative 3rd MoE) models implementation.
"""

import os
import torch

# Disable torch.compile for G3MoE models to avoid data-dependent branching issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True

from .moe_config import *
from .moe_model import *
from .g3moe_config import G3MoEConfig, G3MoETextConfig
from .g3moe_model import (
    G3MoEPreTrainedModel,
    G3MoETextModel,
    G3MoEForCausalLM,
    G3MoEForConditionalGeneration,
    G3MoEModel,
)
from .g2moe_config import *
from .g2moe_model import *

__all__ = [
    "G3MoEConfig",
    "G3MoETextConfig", 
    "G3MoEPreTrainedModel",
    "G3MoETextModel",
    "G3MoEForCausalLM",
    "G3MoEForConditionalGeneration",
    "G3MoEModel",
]