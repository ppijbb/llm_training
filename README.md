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

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:
```bash
poetry install
```

Alternatively, you can use pip with requirements.txt:
```bash
pip install -r requirements.txt
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

## License
This project is licensed under the MIT License.
