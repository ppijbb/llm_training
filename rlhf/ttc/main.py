from ttc.config.config import ttc_config
from ttc.models.llm_model import LLMModel
from ttc.models.reward_model import RewardModel
from ttc.training.trainer import RewardModelTrainer
from ttc.inference.inference_engine import TTCInferenceEngine
from ttc.utils.data_loader import DataLoader
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run Test-Time Compute (TTC) operations.")
    parser.add_argument("--mode", type=str, choices=["train_reward", "inference"], required=True,
                        help="Mode to run: 'train_reward' to train the reward model, 'inference' to run TTC inference.")
    parser.add_argument("--prompt", type=str, help="Prompt for inference mode.")
    parser.add_argument("--strategy", type=str, default="best_of_n", choices=["best_of_n", "mcts"],
                        help="Inference strategy to use: 'best_of_n' or 'mcts'.")
    args = parser.parse_args()

    # Initialize models
    # Note: For production, consider lazy loading or singleton patterns for models
    print(f"Loading LLM model: {ttc_config.LLM_MODEL_NAME}...")
    llm_model = LLMModel(model_name=ttc_config.LLM_MODEL_NAME, quantization_config=ttc_config.QUANTIZATION_CONFIG)
    print("LLM model loaded.")

    print(f"Loading Reward model: {ttc_config.REWARD_MODEL_NAME}...")
    reward_model = RewardModel(model_name=ttc_config.REWARD_MODEL_NAME)
    print("Reward model loaded.")

    if args.mode == "train_reward":
        print("Starting Reward Model Training...")
        data_loader = DataLoader(ttc_config.TRAINING_DATA_PATH)
        train_dataset = data_loader.load_for_reward_model_training()

        # You might need to preprocess your dataset here for the specific TRL trainer
        # For RewardTrainer, typically format is {'prompt': ..., 'chosen': ..., 'rejected': ...}
        # For SFTTrainer with a score, it's {'text': ..., 'label': ...}
        # For this example, assuming SFTTrainer for a simple text-to-score task:
        # Ensure your dataset has a 'text' column and a 'label' column (the score)
        # If your data is in a different format, you'll need a mapping function.
        # Example of simple mapping if 'text' and 'score' are present:
        # def format_dataset_for_sft(example):
        #     return {"text": example["prompt"] + example["response"], "label": example["score"]}
        # train_dataset = train_dataset.map(format_dataset_for_sft)

        trainer = RewardModelTrainer(reward_model=reward_model, train_dataset=train_dataset)
        trainer.train()
        print("Reward Model Training Completed and Model Saved.")

    elif args.mode == "inference":
        if not args.prompt:
            print("Error: --prompt is required for inference mode.")
            return

        print(f"Running Inference with strategy: {args.strategy}")
        inference_engine = TTCInferenceEngine(llm_model, reward_model)
        try:
            response, score = inference_engine.run_inference(args.prompt, args.strategy)
            print("\n--- Inference Result ---")
            print(f"Prompt: {args.prompt}")
            print(f"Response: {response}")
            print(f"Score: {score:.4f}")
            print("------------------------")
        except ValueError as e:
            print(f"Inference Error: {e}")

if __name__ == "__main__":
    main() 