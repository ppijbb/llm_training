# LLM Training Project

## Description
This project provides a comprehensive framework for training and fine-tuning Large Language Models (LLMs) using various methods including Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), and quantization techniques.

## Project Structure
```
llm_training/
├── data/                    # Training and evaluation datasets
├── gkd/                     # Generalized Knowledge Distillation related code
├── lightning_trainer/       # PyTorch Lightning-based training framework
├── models/                  # Model definitions and configurations
├── quantization/            # Model quantization utilities
├── rlhf/                    # Reinforcement Learning from Human Feedback training
├── sft/                     # Supervised Fine-Tuning with llama_recipes
├── summary_format/          # Summary formatting utilities
├── training_utils/          # Common training utilities
├── main.py                  # Main inference script
├── pyproject.toml           # Poetry configuration
└── requirements.txt         # Python dependencies
```

## Installation

### Prerequisites
- Python 3.12 or higher
- CUDA-compatible GPU (recommended)
- Poetry (for dependency management)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd llm_training
```

2. Install uv:
```bash
pip install uv
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt --no-build-isolation --index-strategy unsafe-best-match
```

- if run on GCP VM, use local ssd

    ```sh
    sudo lsblk -o NAME,SIZE,TYPE,MOUNTPOINT | grep nvme0n1
    sudo mkfs.ext4 -F /dev/nvme0n1
    sudo mkdir -p /mnt/disks/local-ssd
    sudo mount /dev/nvme0n1 /mnt/disks/local-ssd
    sudo chmod a+w /mnt/disks/local-ssd
    UUID=$(sudo blkid -s UUID -o value /dev/nvme0n1)
    echo "UUID=$UUID /mnt/disks/local-ssd ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
    ```

## Usage

### Model Inference
Run the main inference script to test a fine-tuned model:
```bash
python main.py
```

### Supervised Fine-Tuning (SFT)
1. Navigate to the SFT directory:
```bash
cd sft
```

2. Configure your training parameters in the config files and run:
```bash
python llama_finetuning.py
```

### Lightning Trainer
1. Navigate to the lightning trainer directory:
```bash
cd lightning_trainer
```

2. Set up environment variables:
```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY=<your_wandb_api_key>
export HF_SECRET_KEY=<your_huggingface_token>
export HF_DATASETS_CACHE=<your_cache_directory>
```

3. Login to required services:
```bash
huggingface-cli login --token $HF_SECRET_KEY
wandb login --relogin $WANDB_API_KEY
```

4. Start training with tmux (recommended):
```bash
tmux new -s lightning -d
tmux attach -t lightning

python trainer.py fit \
    --trainer.fast_dev_run false \
    --trainer.max_epochs 5 \
    --model.learning_rate 3e-3 \
    --data.train_batch_size 4 \
    --data.eval_batch_size 4
```

### RLHF Training
1. Navigate to the RLHF directory:
```bash
cd rlhf
```

2. Set up environment variables:
```bash
export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False'
export TORCH_USE_CUDA_DSA=1
export WANDB_API_KEY=<your_wandb_api_key>
export HF_SECRET_KEY=<your_huggingface_token>
export HF_DATASETS_CACHE=<your_cache_directory>
```

3. Login to required services:
```bash
huggingface-cli login --token $HF_SECRET_KEY
wandb login --relogin $WANDB_API_KEY
```

4. Start training with accelerate:
```bash
tmux new -s rlhf -d
tmux attach -t rlhf

accelerate launch \
    --config_file "accelerate_config.yaml" \
    train.py
```

## Features
- **Multiple Training Methods**: SFT, RLHF, and Lightning-based training
- **Model Quantization**: Support for efficient model compression
- **Distributed Training**: Multi-GPU support with accelerate
- **Monitoring**: Integration with Weights & Biases for experiment tracking
- **Flexible Configuration**: Easy-to-modify configuration files

## Dependencies
Key dependencies include:
- PyTorch 2.4.0+ (with CUDA support)
- Transformers (latest from GitHub)
- LangChain 0.2.5+
- Lightning AI framework
- Hugging Face ecosystem (datasets, tokenizers, etc.)

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Memory Management & ZeRO Optimization

### ZeRO Memory Optimization (ZeRO-2/3 + ZenFlow)
This project uses DeepSpeed ZeRO with ZenFlow for advanced memory optimization, providing the best of both worlds for large MoE models like SPECTRA.

#### Supported Configurations
- **ZeRO-2 + ZenFlow**: Official combination for maximum performance
- **ZeRO-3 + ZenFlow**: Experimental combination (may not be officially supported)
- **ZeRO-3 Only**: Standard ZeRO-3 partitioning without ZenFlow

#### ZenFlow Features (when enabled)
- **Asynchronous Gradient Updates**: Reduces CPU-GPU stalls during offloading
- **Selective Gradient Updates**: Prioritizes important gradients for GPU updates
- **Communication Overlap**: Overlaps computation with communication
- **Up to 5x speedup** over standard ZeRO-Offload

#### ZenFlow Configuration
```json
"zenflow": {
  "topk_ratio": 0.05,        // Top 5% important gradients updated on GPU
  "select_strategy": "auto",  // Adaptive selection strategy
  "select_interval": "auto",  // Adaptive reselection frequency
  "update_interval": 4,       // Update unimportant gradients every 4 steps
  "overlap_step": true        // Enable computation-communication overlap
}
```

#### ZeRO-3 Features (when stage=3)
- **Parameter Partitioning**: 16-bit model parameters partitioned across processes
- **Gradient Partitioning**: Gradients partitioned for memory efficiency
- **Optimizer State Partitioning**: Optimizer states partitioned to reduce redundancy
- **NVMe Offloading**: Parameters and optimizer states offloaded to NVMe storage

#### Memory Monitoring
- Automatic GPU and system RAM monitoring during training
- Warnings when memory usage exceeds safe thresholds (90% GPU memory, 85% system RAM)
- ZenFlow/ZeRO-3 specific memory usage tracking and optimization suggestions

#### Environment Variables
```bash
# Force disable ZenFlow if RAM OOM occurs
export DISABLE_ZENFLOW=1
```

#### Memory Optimization Tips
1. **Choose the right ZeRO stage**: ZeRO-2 + ZenFlow for best performance, ZeRO-3 for maximum memory efficiency
2. **Monitor memory usage** in training logs for optimization warnings
3. **Reduce batch size** if memory issues persist (`per_device_train_batch_size`)
4. **Enable gradient checkpointing** for additional memory savings (`gradient_checkpointing: true`)
5. **Use NVMe offload** for maximum memory savings (already configured)
6. **Disable ZenFlow** (`DISABLE_ZENFLOW=1`) if experiencing RAM OOM issues

## License
This project is licensed under the MIT License.
