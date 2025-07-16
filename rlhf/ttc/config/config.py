import os

class TTCConfig:
    # Model Configuration
    LLM_MODEL_NAME = "meta-llama/Llama-3-8B" # Placeholder, replace with actual model
    REWARD_MODEL_NAME = "openai/gpt2" # Placeholder for a small base model for reward
    QUANTIZATION_CONFIG = None # Example: bitsandbytes.BitsAndBytesConfig(...)

    # Training Configuration for Reward Model
    TRAINING_DATA_PATH = "data/reward_training_data.jsonl"
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 3
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 1

    # Inference Configuration
    MAX_NEW_TOKENS = 256
    BEST_OF_N = 5 # For Best-of-N sampling
    MCTS_ITERATIONS = 100 # For Monte Carlo Tree Search
    TEMPERATURE = 0.7

    # Paths
    REWARD_MODEL_OUTPUT_DIR = "models/reward_model_checkpoints"
    LOG_DIR = "logs"
    DATA_DIR = "data"

    def __init__(self):
        # Create necessary directories if they don't exist
        os.makedirs(self.REWARD_MODEL_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)

# Instantiate the config for easy import
ttc_config = TTCConfig() 