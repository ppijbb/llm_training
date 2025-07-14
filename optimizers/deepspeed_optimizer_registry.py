#!/usr/bin/env python3
"""
DeepSpeed Custom Optimizer Registry
"""

import os
import sys
from typing import Dict, Any, Type
import torch.optim as optim

# Add custom optimizers to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_optimizers import LionOptimizer, AdaFactorOptimizer, SophiaOptimizer, MuonOptimizer


def register_custom_optimizers():
    """
    Register custom optimizers with DeepSpeed
    This function should be called before DeepSpeed initialization
    """
    try:
        import deepspeed
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        
        # Register custom optimizers with DeepSpeed
        if hasattr(deepspeed, 'ops') and hasattr(deepspeed.ops, 'adam'):
            # Add custom optimizers to DeepSpeed's optimizer registry
            deepspeed.ops.adam.LionOptimizer = LionOptimizer
            deepspeed.ops.adam.AdaFactorOptimizer = AdaFactorOptimizer
            deepspeed.ops.adam.SophiaOptimizer = SophiaOptimizer
            deepspeed.ops.adam.MuonOptimizer = MuonOptimizer
            
            print("✅ Custom optimizers registered with DeepSpeed")
        else:
            print("⚠️ DeepSpeed ops not found, custom optimizers may not work")
            
    except ImportError:
        print("⚠️ DeepSpeed not found, custom optimizers will not be registered")


def get_optimizer_class(optimizer_name: str):
    """
    Get optimizer class by name
    
    Args:
        optimizer_name: Name of the optimizer
        
    Returns:
        Optimizer class
    """
    optimizer_map = {
        'adamw': optim.AdamW,
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'lion': LionOptimizer,
        'adafactor': AdaFactorOptimizer,
        'sophia': SophiaOptimizer,
        'muon': MuonOptimizer,
    }
    
    return optimizer_map.get(optimizer_name.lower())


def create_optimizer_from_config(optimizer_config: Dict[str, Any], model_params):
    """
    Create optimizer from DeepSpeed config
    
    Args:
        optimizer_config: Optimizer configuration from DeepSpeed config
        model_params: Model parameters
        
    Returns:
        Optimizer instance
    """
    optimizer_type = optimizer_config.get('type', 'AdamW')
    optimizer_params = optimizer_config.get('params', {})
    
    # Get optimizer class
    optimizer_class = get_optimizer_class(optimizer_type)
    
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Create optimizer instance
    optimizer = optimizer_class(model_params, **optimizer_params)
    
    return optimizer


# Register optimizers when module is imported
register_custom_optimizers() 