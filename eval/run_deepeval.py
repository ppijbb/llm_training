from typing import List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM


class Mistral7B(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str
    ):
        self.model_name = model_name

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)

    def _call(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    # This is optional.
    def batch_generate(self, prompts: List[str]) -> List[str]:
        model = self.load_model()
        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer(prompts, return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)

    def get_model_name(self):
        return self.model_name


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
            
        model_name = "Gunulhona/Gemma-3-4B"
        eval_model = Mistral7B(model_name=model_name)

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
