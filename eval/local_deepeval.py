from typing import List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM


class LocalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str | PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        *args,
        **kwargs
    ):
        super().__init__(model_name=model, *args, **kwargs)
        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(self.model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.device = self.model.device if hasattr(self.model, 'device') else "cuda"
        elif tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = self.model.device if hasattr(self.model, 'device') else "cuda"
        else:
            raise ValueError("Either model or tokenizer must be provided")

    def load_model(self):
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)

    def _call(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt, handling DeepEval's prompt and schema parameters
        """
        # Remove DeepEval-specific parameters that the model doesn't expect
        model_kwargs = kwargs.copy()
        
        
        # Tokenize the prompt
        try:
            # Try to use the tokenizer's apply_chat_template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                # Check if prompt looks like a chat format
                if self.tokenizer.bos_token in prompt or self.tokenizer.eos_token in prompt or 'user' in prompt.lower():
                    # It's already formatted, use as is
                    input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                else:
                    # Format as chat
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=True, 
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )
                    input_ids = formatted_prompt.to(self.device)
            else:
                # Standard tokenization
                input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        except Exception as e:
            print(f"Tokenization error: {e}")
            # Fallback to simple tokenization
            input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        try:
            generated_ids = self.model.generate(
                **input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                **model_kwargs
            )
            
            # Decode the generated text
            if hasattr(input_ids, 'input_ids'):
                input_length = input_ids.input_ids.shape[1]
            else:
                input_length = input_ids['input_ids'].shape[1]
            
            response_ids = generated_ids[0][input_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return f"Error during generation: {str(e)}"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate text for multiple prompts
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt)
                results.append(result)
            except Exception as e:
                print(f"Error generating for prompt: {e}")
                results.append(f"Error: {str(e)}")
        return results

    def get_model_name(self):
        if isinstance(self.model_name, str):
            return self.model_name
        else:
            return getattr(self.model, 'name_or_path', 'unknown_model')


def run_deepeval()->None:
    try:
        from deepeval.benchmarks import HellaSwag
        from deepeval.benchmarks import MMLU
        from deepeval.benchmarks import BigBenchHard
        from deepeval.benchmarks import HumanEval
        from deepeval.benchmarks.tasks import HumanEvalTask
        from deepeval.benchmarks import SQuAD
        from deepeval.benchmarks.tasks import SQuADTask
        from deepeval.benchmarks import GSM8K
        from deepeval.benchmarks import MathQA
        from deepeval.benchmarks.tasks import MathQATask

        benchmarks = [
            MMLU(),
            # HellaSwag(n_shots=3),
            # BigBenchHard(enable_cot=True),
            # HumanEval(
            #     tasks=[HumanEvalTask.HAS_CLOSE_ELEMENTS, HumanEvalTask.SORT_NUMBERS],
            #     n=100
            # ),
            # SQuAD(
            #     tasks=[SQuADTask.PHARMACY, SQuADTask.NORMANS],
            #     n_shots=3
            # ),
            # GSM8K(
            #     n_problems=10,
            #     n_shots=3,
            #     enable_cot=True
            # ),
            #  MathQA(
            #     tasks=[MathQATask.PROBABILITY, MathQATask.GEOMETRY],
            #     n_shots=3
            # )
        ]
            
        model_name = "google/gemma-3-4b-it"
        eval_model = LocalModel(model_name=model_name)

        for benchmark in benchmarks:
            results = benchmark.evaluate(model=eval_model, batch_size=8)
            print(f"Overall Score: {benchmark.__name__} - {results}")

    except Exception as e:
        import traceback
        traceback.print_exc()

    finally:
        print("--- Done ---")
    
    
if __name__ == "__main__":
    run_deepeval()
