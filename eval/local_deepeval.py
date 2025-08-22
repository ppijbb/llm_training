import torch
import outlines
from typing import List, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from transformers.processing_utils import ProcessorMixin
from transformers.generation.logits_process import LogitsProcessorList
from deepeval.models.base_model import DeepEvalBaseLLM

# Setting high precision for matmul
torch.set_float32_matmul_precision('high')


class CustomOutlinesTransformers(outlines.models.TransformersMultiModal):
    def _generate_output_seq(self, prompts, inputs, **inference_kwargs):
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        if "token_type_ids" in inference_kwargs:
            del inference_kwargs["token_type_ids"]
        return super()._generate_output_seq(prompts, inputs, **inference_kwargs)


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
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(model).to(self.device)
            config = AutoConfig.from_pretrained(model).to_dict()
            if "text_config" in config or "vision_config" in config or "audio_config" in config:
                self.tokenizer = AutoProcessor.from_pretrained(model)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model)

        elif tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = self.model.device if hasattr(self.model, 'device') else "cuda"
        else:
            raise ValueError("Either model or tokenizer must be provided")
        # outlines.from_transformers(self.model, self.tokenizer)
        self.structured_model = CustomOutlinesTransformers(self.model, self.tokenizer)

    def load_model(self):
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)

    def _call(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, prompt: str, schema: Optional[Any] = None, **kwargs) -> str:
        """
        Generate text from prompt, handling DeepEval's prompt and schema parameters
        """
        # Remove DeepEval-specific parameters that the model doesn't expect
        model_kwargs = kwargs.copy()
        _prompt = model_kwargs.get("prompt", None)
        if schema is None:
            schema = model_kwargs.get("schema", None)
        try:
            response = self.structured_model(
                model_input={"text": prompt}, 
                output_type=schema,
                generation_config=GenerationConfig(
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=1.0
                ))
            return schema.model_validate_json(response)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Generation error: {e}")
            return f"Error during generation: {str(e)}"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str], schemas: Optional[Any] = None, **kwargs) -> List[str]:
        """
        Generate text for multiple prompts
        """
        results = []
        for prompt, schema in zip(prompts, schemas):
            try:
                result = self.generate(prompt, schema, **kwargs)
                results.append([result])
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
        from deepeval.benchmarks.mmlu.task import MMLUTask
        from deepeval.benchmarks import BigBenchHard
        from deepeval.benchmarks import HumanEval
        from deepeval.benchmarks.tasks import HumanEvalTask
        from deepeval.benchmarks import SQuAD
        from deepeval.benchmarks.tasks import SQuADTask
        from deepeval.benchmarks import GSM8K
        from deepeval.benchmarks import MathQA
        from deepeval.benchmarks.tasks import MathQATask

        benchmarks = [
            MMLU(
                tasks=[task for task in MMLUTask],
                n_shots=5
            ),
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
        eval_model = LocalModel(model=model_name)

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
