#!/bin/bash

# G3MoE SFT Training Script with DeepSpeed Support
# Usage: ./run_g3moe_deepspeed.sh [config_file] [num_gpus]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    G3MoE SFT DeepSpeed Training       ${NC}"
echo -e "${GREEN}========================================${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default config file and GPU count
CONFIG_FILE="${1:-$SCRIPT_DIR/config/g3moe_deepspeed_config.json}"
NUM_GPUS="${2:-1}"

echo -e "${YELLOW}Project Root:${NC} $PROJECT_ROOT"
echo -e "${YELLOW}Config File:${NC} $CONFIG_FILE"
echo -e "${YELLOW}Number of GPUs:${NC} $NUM_GPUS"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found at $CONFIG_FILE${NC}"
    echo "Available configs:"
    find "$SCRIPT_DIR/config" -name "*.json" -type f 2>/dev/null | sort || echo "  No config files found"
    exit 1
fi

# Check if DeepSpeed config is specified and exists
DEEPSPEED_CONFIG=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(config['model_config'].get('deepspeed_config', ''))" 2>/dev/null || echo "")
if [ -n "$DEEPSPEED_CONFIG" ] && [ "$DEEPSPEED_CONFIG" != "null" ] && [ "$DEEPSPEED_CONFIG" != "None" ]; then
    if [ ! -f "$DEEPSPEED_CONFIG" ]; then
        echo -e "${RED}Error: DeepSpeed config not found at $DEEPSPEED_CONFIG${NC}"
        exit 1
    fi
    echo -e "${BLUE}DeepSpeed Config:${NC} $DEEPSPEED_CONFIG"
fi

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS-1)))}
export TOKENIZERS_PARALLELISM=false

# For 120K context length, increase memory limits
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export CUDA_LAUNCH_BLOCKING=0

# Create output directory
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training_config']['output_dir'])")
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Output Directory:${NC} $OUTPUT_DIR"
echo -e "${YELLOW}CUDA Devices:${NC} $CUDA_VISIBLE_DEVICES"

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
for package in torch transformers trl peft datasets wandb deepspeed; do
    if ! python3 -c "import $package" 2>/dev/null; then
        echo -e "${RED}Error: $package is not installed${NC}"
        echo "Please install required packages:"
        echo "pip install torch transformers trl peft datasets wandb deepspeed"
        exit 1
    fi
done
echo -e "${GREEN}All dependencies found!${NC}"

# Check GPU availability and memory
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo -e "${GREEN}GPU detected: $GPU_COUNT device(s) available${NC}"
    
    # Check GPU memory for each device
    python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    memory_gb = props.total_memory / (1024**3)
    print(f'GPU {i}: {props.name}, {memory_gb:.1f}GB VRAM')
    if memory_gb < 40:
        print(f'WARNING: GPU {i} has less than 40GB VRAM. 120K context may cause OOM.')
"
else
    echo -e "${RED}Error: No GPU detected. DeepSpeed requires CUDA.${NC}"
    exit 1
fi

# Check system memory for DeepSpeed CPU offload
TOTAL_RAM_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
echo -e "${YELLOW}System RAM:${NC} ${TOTAL_RAM_GB}GB"
if [ "$TOTAL_RAM_GB" -lt 64 ]; then
    echo -e "${YELLOW}Warning: Less than 64GB RAM. DeepSpeed CPU offload may be slow.${NC}"
fi

# Determine the training command based on number of GPUs
cd "$PROJECT_ROOT"

# To run with 1 GPU, use the following command:
# python3 $SCRIPT_DIR/custom_model_sft.py --config $CONFIG_FILE

# To run with multiple GPUs, use the following command:
# torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_DIR/custom_model_sft.py --config $CONFIG_FILE

if [ "$NUM_GPUS" -eq 1 ]; then
    echo -e "${GREEN}Starting single-GPU DeepSpeed training...${NC}"
    TRAIN_CMD="uv run accelerate launch --num_processes=1 $SCRIPT_DIR/custom_model_sft.py --config $CONFIG_FILE"
else
    echo -e "${GREEN}Starting multi-GPU DeepSpeed training with $NUM_GPUS GPUs...${NC}"
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_DIR/custom_model_sft.py --config $CONFIG_FILE"
fi

echo -e "${BLUE}Command:${NC} $TRAIN_CMD"
echo ""

# Run training with error handling
if eval "$TRAIN_CMD"; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}Model saved to: $OUTPUT_DIR${NC}"
    
    # Show model size info
    echo -e "${BLUE}Model information:${NC}"
    python3 -c "
import os
import torch
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
if (output_dir / 'pytorch_model.bin').exists():
    model_file = output_dir / 'pytorch_model.bin'
    size_mb = model_file.stat().st_size / (1024*1024)
    print(f'Model file size: {size_mb:.1f} MB')
elif (output_dir / 'adapter_model.safetensors').exists():
    adapter_file = output_dir / 'adapter_model.safetensors'
    size_mb = adapter_file.stat().st_size / (1024*1024)
    print(f'LoRA adapter size: {size_mb:.1f} MB')
"
else
    echo -e "${RED}Training failed!${NC}"
    echo -e "${YELLOW}Troubleshooting tips for 120K context:${NC}"
    echo "1. Try reducing batch size: per_device_train_batch_size=1"
    echo "2. Increase gradient accumulation: gradient_accumulation_steps=64"
    echo "3. Use DeepSpeed ZeRO-3 with CPU offload"
    echo "4. Enable gradient checkpointing"
    echo "5. Reduce LoRA rank if using LoRA"
    exit 1
fi

# Memory usage summary
echo -e "${BLUE}Final GPU memory usage:${NC}"
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        cached = torch.cuda.memory_reserved(i) / (1024**3)
        print(f'GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached')
"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}      DeepSpeed Training Complete!     ${NC}"
echo -e "${GREEN}========================================${NC}" 