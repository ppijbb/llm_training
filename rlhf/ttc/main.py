import sys
import os
# Add the current directory to Python path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import ttc_config
from models.llm_model import LLMModel
from models.reward_model import RewardModel
from models.teacher_model import TeacherModel
from training.trainer import RewardModelTrainer
from training.rlt_trainer import RLTTrainer
from inference.inference_engine import TTCInferenceEngine
from inference.rlt_inference_engine import RLTInferenceEngine
from utils.data_loader import DataLoader
from utils.rlt_data_loader import RLTDataLoader
import argparse
from typing import Optional

def main():
    parser = argparse.ArgumentParser(description="Run RLT (Reinforcement Learning Teachers) Test-Time Compute operations.")
    parser.add_argument("--mode", type=str, 
                        choices=["train_reward", "train_rlt", "train_sft", "train_rl", "inference", "rlt_inference"], 
                        required=True,
                        help="Mode to run: 'train_reward' for reward model, 'train_rlt' for full RLT pipeline, 'train_sft' for SFT only, 'train_rl' for RL only, 'inference' for original TTC, 'rlt_inference' for RLT inference.")
    parser.add_argument("--prompt", type=str, help="Prompt/question for inference mode.")
    parser.add_argument("--strategy", type=str, default="adaptive", 
                        choices=["adaptive", "best_of_n", "majority_voting", "fixed_scaling"],
                        help="Inference strategy for RLT inference.")
    parser.add_argument("--scaling_factor", type=float, default=1.0,
                        help="Scaling factor for test-time compute.")
    parser.add_argument("--target_quality", type=float, default=0.8,
                        help="Target quality for adaptive scaling.")
    parser.add_argument("--dataset_path", type=str, help="Path to custom dataset.")
    parser.add_argument("--model_path", type=str, help="Path to pre-trained model.")
    args = parser.parse_args()

    # Initialize models based on mode
    teacher_model: Optional[TeacherModel] = None
    llm_model: Optional[LLMModel] = None
    reward_model: Optional[RewardModel] = None

    if args.mode in ["train_rlt", "train_sft", "train_rl", "rlt_inference"]:
        # RLT modes - use teacher model
        print(f"Loading Teacher model: {ttc_config.TEACHER_MODEL_NAME}...")
        teacher_model = TeacherModel(
            model_name=args.model_path or ttc_config.TEACHER_MODEL_NAME, 
            quantization_config=ttc_config.QUANTIZATION_CONFIG
        )
        print("Teacher model loaded.")
        
        # Load reward model for RL phase
        if args.mode in ["train_rl", "train_rlt"]:
            print(f"Loading Reward model: {ttc_config.REWARD_MODEL_NAME}...")
            reward_model = RewardModel(model_name=ttc_config.REWARD_MODEL_NAME)
            print("Reward model loaded.")
    
    else:
        # Original TTC modes
        print(f"Loading LLM model: {ttc_config.LLM_MODEL_NAME}...")
        llm_model = LLMModel(model_name=ttc_config.LLM_MODEL_NAME, quantization_config=ttc_config.QUANTIZATION_CONFIG)
        print("LLM model loaded.")

        print(f"Loading Reward model: {ttc_config.REWARD_MODEL_NAME}...")
        reward_model = RewardModel(model_name=ttc_config.REWARD_MODEL_NAME)
        print("Reward model loaded.")

    # Training modes
    if args.mode == "train_reward":
        print("Starting Reward Model Training...")
        if reward_model is None:
            print("Error: Reward model is required for reward training")
            return
            
        data_loader = DataLoader(ttc_config.TRAINING_DATA_PATH)
        train_dataset = data_loader.load_for_reward_model_training()
        
        # Ensure train_dataset is a Dataset, not DatasetDict
        if hasattr(train_dataset, 'train'):
            train_dataset = train_dataset['train']
            
        trainer = RewardModelTrainer(reward_model=reward_model, train_dataset=train_dataset)
        trainer.train()
        print("Reward Model Training Completed and Model Saved.")

    elif args.mode in ["train_rlt", "train_sft", "train_rl"]:
        print(f"Starting RLT Training: {args.mode}...")
        
        if teacher_model is None:
            print("Error: Teacher model is required for RLT training")
            return
        
        # Load reasoning dataset
        rlt_data_loader = RLTDataLoader(data_path=args.dataset_path)
        dataset = rlt_data_loader.load_reasoning_dataset()
        
        # Validate dataset format
        if not rlt_data_loader.validate_dataset_format(dataset):
            print("Dataset format validation failed. Exiting.")
            return
        
        # Create train/eval split
        train_dataset, eval_dataset = rlt_data_loader.create_train_eval_split(dataset)
        
        # Initialize RLT trainer
        rlt_trainer = RLTTrainer(teacher_model=teacher_model, reward_model=reward_model)
        
        if args.mode == "train_sft":
            # SFT phase only
            sft_model_path = rlt_trainer.train_sft_phase(train_dataset, eval_dataset)
            print(f"SFT training completed. Model saved to {sft_model_path}")
            
        elif args.mode == "train_rl":
            # RL phase only (requires pre-trained SFT model)
            if not args.model_path:
                print("Error: --model_path is required for RL-only training (should point to SFT model)")
                return
            rl_model_path = rlt_trainer.train_rl_phase(train_dataset, eval_dataset)
            print(f"RL training completed. Model saved to {rl_model_path}")
            
        elif args.mode == "train_rlt":
            # Full RLT pipeline
            final_model_path = rlt_trainer.train_full_pipeline(train_dataset, eval_dataset)
            print(f"Full RLT training completed. Model saved to {final_model_path}")

    # Inference modes
    elif args.mode == "inference":
        if not args.prompt:
            print("Error: --prompt is required for inference mode.")
            return

        if llm_model is None or reward_model is None:
            print("Error: LLM model and reward model are required for inference")
            return

        print(f"Running Original TTC Inference...")
        inference_engine = TTCInferenceEngine(llm_model, reward_model)
        try:
            response, score = inference_engine.run_inference(args.prompt, "best_of_n")
            print("\n--- Original TTC Inference Result ---")
            print(f"Prompt: {args.prompt}")
            print(f"Response: {response}")
            print(f"Score: {score:.4f}")
            print("--------------------------------------")
        except ValueError as e:
            print(f"Inference Error: {e}")

    elif args.mode == "rlt_inference":
        if not args.prompt:
            print("Error: --prompt is required for RLT inference mode.")
            return

        if teacher_model is None:
            print("Error: Teacher model is required for RLT inference")
            return

        print(f"Running RLT Inference with strategy: {args.strategy}")
        rlt_inference_engine = RLTInferenceEngine(teacher_model, reward_model)
        
        try:
            # Prepare inference arguments
            inference_kwargs = {}
            if args.strategy == "adaptive":
                inference_kwargs['target_quality'] = args.target_quality
            elif args.strategy == "fixed_scaling":
                inference_kwargs['scaling_factor'] = args.scaling_factor
            elif args.strategy == "best_of_n":
                inference_kwargs['n'] = ttc_config.BEST_OF_N
                inference_kwargs['scaling_factor'] = args.scaling_factor
            elif args.strategy == "majority_voting":
                inference_kwargs['num_votes'] = 5
                inference_kwargs['scaling_factor'] = args.scaling_factor
            
            result = rlt_inference_engine.run_inference(args.prompt, args.strategy, **inference_kwargs)
            
            print("\n--- RLT Inference Result ---")
            print(f"Question: {result['question']}")
            print(f"Reasoning Trace: {result['reasoning_trace']}")
            print(f"Solution: {result['solution']}")
            print(f"Quality Score: {result['quality_score']:.4f}")
            print(f"Scaling Factor: {result['scaling_factor']}")
            print(f"Tokens Used: {result['tokens_used']}")
            
            if 'adaptive_scaling_info' in result:
                info = result['adaptive_scaling_info']
                print(f"Adaptive Scaling Info:")
                print(f"  Target Quality: {info['target_quality']}")
                print(f"  Scaling Factors Tried: {info['scaling_factors_tried']}")
                print(f"  Quality Achieved: {info['quality_achieved']:.4f}")
                if info.get('target_not_reached'):
                    print("  Target quality not reached with available scaling factors")
            
            if 'majority_voting_info' in result:
                info = result['majority_voting_info']
                print(f"Majority Voting Info:")
                print(f"  Number of Votes: {info['num_votes']}")
                print(f"  Most Common Solution: {info['most_common_solution']}")
            
            print("----------------------------")
            
        except Exception as e:
            print(f"RLT Inference Error: {e}")

if __name__ == "__main__":
    main() 