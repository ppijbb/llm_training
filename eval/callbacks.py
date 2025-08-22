from typing import List, Dict, Optional
import os
import sys
import json
import torch
from tqdm import tqdm
import re
from datasets import load_dataset

from deepeval.metrics import BaseMetric
from deepeval.evaluate.execute import execute_test_cases
from deepeval.dataset import EvaluationDataset
from deepeval.integrations.hugging_face import DeepEvalHuggingFaceCallback
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.integrations.hugging_face.utils import generate_test_cases
from transformers import (
        Trainer,
        TrainingArguments,
        TrainerState,
        TrainerControl,
        AutoTokenizer,
        AutoProcessor,
        AutoConfig,
        GenerationConfig
    )
from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteriaList, MaxLengthCriteria
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.hellaswag.hellaswag import HellaSwag
from deepeval.benchmarks import BigBenchHard
from deepeval.benchmarks import HumanEval
from deepeval.benchmarks.tasks import HumanEvalTask
from deepeval.benchmarks import SQuAD
from deepeval.benchmarks.tasks import SQuADTask
from deepeval.benchmarks import GSM8K
from deepeval.benchmarks import MathQA
from deepeval.benchmarks.tasks import MathQATask
from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks import IFEval
# from deepeval.benchmarks import AGIEval
from deepeval.benchmarks import ARC
from deepeval.benchmarks import HellaSwag
# from deepeval.benchmarks import OpenBookQA
# from deepeval.benchmarks import PIQA
# from deepeval.benchmarks import WinoGrande
from deepeval.benchmarks import BoolQ
# from deepeval.benchmarks import CB
# from deepeval.benchmarks import COPA
# from deepeval.benchmarks import MultiRC
# from deepeval.benchmarks import ReCoRD
# from deepeval.benchmarks import RTE
# from deepeval.benchmarks import WiC
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import ArenaTestCase, LLMTestCase
from deepeval.metrics import ArenaGEval
from deepeval.integrations.hugging_face.rich_manager import RichManager
from trl import SFTTrainer
from transformers.trainer_callback import ProgressCallback
from eval.local_deepeval import LocalModel
# Add parent directory to path for custom model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Distributed rank helper
def _is_main_process() -> bool:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True

# Disable torch.compile to avoid data-dependent branching issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch.compiler.disable()
torch._dynamo.reset()


@torch.inference_mode()
def run_mme_evaluation(model, tokenizer, max_samples_per_task=50):
    """
    MME 벤치마크를 수동으로 실행하여 모델의 비전-언어 능력을 평가합니다.
    """
    print("\n" + "="*50)
    print("MME (Multimodal Model Evaluation) 수동 평가 시작")
    print("="*50)

    try:
        print("MME 데이터셋 로딩 중...")
        mme_dataset = load_dataset("MMMU/MME")
        print("MME 데이터셋 로드 완료.")
    except Exception as e:
        print(f"MME 데이터셋 로딩 실패: {e}")
        return {}

    tasks_to_run = ['color', 'count', 'position', 'posters', 'ocr']
    results = {}
    device = model.device

    for task in tasks_to_run:
        if task not in mme_dataset:
            print(f"경고: MME 데이터셋에 '{task}' 태스크가 없습니다. 건너뜁니다.")
            continue
            
        print(f"\n--- '{task}' 태스크 평가 중 ---")
        task_dataset = mme_dataset[task]
        correct_predictions = 0
        total_samples = 0

        # Limit samples for faster evaluation during training
        samples_to_evaluate = task_dataset[:max_samples_per_task]

        for sample in tqdm(samples_to_evaluate, desc=f"Evaluating {task}"):
            image = sample['image']
            question = sample['question']
            ground_truth = sample['answer'].strip().lower()

            prompt = f"<image>\n{question}\nAnswer with Yes or No."
            
            inputs = tokenizer(text=[prompt], images=[image], return_tensors="pt").to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            response_text = tokenizer.decode(generated_ids[0])
            cleaned_prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            answer_text = response_text.replace(cleaned_prompt, '').strip().lower()

            match = re.search(r'\b(yes|no)\b', answer_text)
            predicted_answer = match.group(0) if match else "n/a"
            
            if predicted_answer == ground_truth:
                correct_predictions += 1
            total_samples += 1
        
        if total_samples > 0:
            accuracy = (correct_predictions / total_samples) * 100
            results[task] = accuracy
            print(f"'{task}' 태스크 정확도: {accuracy:.2f}% ({correct_predictions}/{total_samples})")

    if results:
        overall_accuracy = sum(results.values()) / len(results)
        results['overall'] = overall_accuracy
        print(f"\n  - {'평균':<10}: {overall_accuracy:.2f}%")
    
    return results


def get_model_eval_callback(
    trainer: Trainer,
    evaluation_dataset: Optional[EvaluationDataset] = None,
    metrics: Optional[List[BaseMetric]] = None,
    tokenizer_args: Optional[Dict] = {},
    aggregation_method: str = "avg",
    show_table: bool = False,
    enable_benchmarks: bool = False,
    benchmarks_to_run: List[str] = ['mmlu', 'hellaswag', 'gsm8k', 'truthfulqa', 'arc', 'piqa'],
    benchmark_eval_frequency: int = 1000,  # Changed to 1000 steps default
    eval_mode: str = "step",  # Added eval_mode parameter with step as default
    mme_max_samples: int = 20,  # Limit MME samples for faster evaluation
): 
    if evaluation_dataset is None:
        """ ref
        evaluation_dataset = EvaluationDataset(
            goldens=[
                Golden(input="..."),
                Golden(input="...")
            ]
        )
        """
        benchmark = MMLU()
        evaluation_dataset = EvaluationDataset(
            goldens=benchmark.load_benchmark_dataset(
                task=MMLUTask.HIGH_SCHOOL_EUROPEAN_HISTORY
            )
        )
        # print("Golden dataset: ","\n",  evaluation_dataset.goldens, "\n")
        # evaluation_dataset = EvaluationDataset(
        #     test_cases=[
        #         LLMTestCase(
        #             name="GPT-4",
        #             input="What is the capital of France?",
        #             actual_output="Paris",
        #         ),
        #         LLMTestCase(
        #             name="Claude",
        #             input="What is the capital of France?",
        #             actual_output="Paris is the capital of France.",
        #         )
        #     ]
        # )
        evaluation_dataset.alias = evaluation_dataset._alias
    if metrics is None:
        metrics = [
             ArenaGEval(
                name="Friendly",
                criteria="Choose the winter of the more friendly contestant based on the input and actual output",
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
            )
        ]
    tokenizer_args.update({
        "return_tensors": "pt", 
        "padding": "max_length", 
        "truncation": True, 
        "max_length": 8192
        })
    return ModelEvalCallback(
        trainer=trainer,
        evaluation_dataset=evaluation_dataset,
        metrics=metrics,
        tokenizer_args=tokenizer_args,
        aggregation_method=aggregation_method,
        show_table=show_table,
        enable_benchmarks=enable_benchmarks,
        benchmarks_to_run=benchmarks_to_run,
        benchmark_eval_frequency=benchmark_eval_frequency,
        eval_mode=eval_mode,
        mme_max_samples=mme_max_samples,
    )

class EmptyRichManager(RichManager): 
    def __init__(self, show_table: bool, total_train_epochs: int) -> None:
        """
        Initialize RichManager.

        Args:
            show_table (bool): Flag to show or hide the table.
            total_train_epochs (int): Total number of training epochs.
        """
        pass

    def _initialize_progress_trackers(self) -> None:
        pass

    def change_spinner_text(self,*args, **kwargs) -> None:
        pass

    def stop(self) -> None:
        pass

    def start(self) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def create_column(self, *args, **kwargs):
        return None, None

    def advance_progress(self) -> None:
        pass


class ModelEvalCallback(DeepEvalHuggingFaceCallback):
    '''
    This callback is used to evaluate the model during training.
    It is used to evaluate the model on the evaluation dataset and save the results to a file.
    The results are saved to a file in the format of a pandas dataframe.
    The file is saved to the path specified by the `results_file` argument.
    The file is saved to the path specified by the `results_file` argument.
    '''
    def __init__(
        self, 
        trainer: Trainer,
        evaluation_dataset: Optional[EvaluationDataset] = None,
        metrics: Optional[List[BaseMetric]] = None,
        tokenizer_args: Optional[Dict] = None,
        aggregation_method: str = "avg",
        show_table: bool = False,
        enable_benchmarks: bool = False,
        benchmarks_to_run: List[str] = ['mmlu', 'hellaswag', 'gsm8k', 'truthfulqa', 'arc', 'piqa'],
        benchmark_eval_frequency: int = 1000,  # Changed to step-based (1000 steps)
        eval_mode: str = "step",  # Changed default to step-based
        mme_max_samples: int = 20,
        *args, 
        **kwargs
    )->None:
        super().__init__(
            trainer=trainer, 
            evaluation_dataset=evaluation_dataset, 
            metrics=metrics,
            tokenizer_args=tokenizer_args,
            show_table=show_table)

        self.trainer.add_callback(ProgressCallback)
        self.rich_manager = EmptyRichManager(show_table=show_table, total_train_epochs=trainer.args.num_train_epochs)
        self.eval_model = LocalModel(model=trainer.model, tokenizer=trainer.tokenizer)
        self.enable_benchmarks = enable_benchmarks
        self.benchmarks_to_run = benchmarks_to_run
        self.benchmark_eval_frequency = benchmark_eval_frequency
        self.eval_mode = eval_mode  # "step" or "epoch"
        self.mme_max_samples = mme_max_samples
        self.benchmark_results_history = []
        self.last_eval_step = 0  # Track last evaluation step
        self.is_main_process = _is_main_process()

    
    def _calculate_metric_scores(self) -> Dict[str, List[float]]:
        # return super()._calculate_metric_scores()
        on_log_metrics = IFEval(n_problems=10, verbose_mode=True)
        self.eval_model = LocalModel(model=self.trainer.model, tokenizer=self.trainer.tokenizer)
        try:
            test_results = on_log_metrics.evaluate(self.eval_model)
            scores = {
                "ifeval": [test_results["overall_accuracy"]]
            }
        except Exception as e:
            print(f"Error in _calculate_metric_scores: {e}\nIFEval scores set to 0.0")
            scores = self._aggregate_scores({"ifeval": [0.0]})
        return scores
    def _aggregate_scores(
        self, 
        scores: Dict[str, List[float]]
    ) -> Dict[str, float]:
        return super()._aggregate_scores(scores)
        
    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_epoch_begin(args, state, control, **kwargs)
        
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Step-based benchmark evaluation"""
        # Only run heavy benchmarking on the main process
        if not self.is_main_process:
            return control
        # Only run if step-based evaluation is enabled
        if (self.eval_mode == "step" and 
            self.enable_benchmarks and
            state.global_step > 0 and
            state.global_step - self.last_eval_step >= self.benchmark_eval_frequency):
            
            with torch.inference_mode():
                print(f"\n{'='*60}")
                print(f"Step {state.global_step} Benchmark Evaluation")
                print(f"{'='*60}")
                
                benchmark_results = self._run_benchmark_evaluation()
                
                if benchmark_results:
                    self.benchmark_results_history.append({
                        'step': state.global_step,
                        'epoch': state.epoch,
                        'results': benchmark_results
                    })
                    
                    # Log benchmark results to trainer
                    for metric_name, score in benchmark_results.items():
                        self.trainer.log({
                            f"benchmark/{metric_name}": score,
                            "step": state.global_step,
                            "epoch": state.epoch
                        })
                    
                    print(f"Benchmark results logged for step {state.global_step}")
                
                self.last_eval_step = state.global_step

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Only run epoch-based evaluation if eval_mode is "epoch"
        if not self.is_main_process:
            return control
        if self.eval_mode == "epoch":
            with torch.inference_mode():
                # with torch.device("cuda"):
                #     control.should_log = True
                #     self.rich_manager.change_spinner_text(
                #         self.task_descriptions["generating"]
                #     )
                #     test_cases = generate_test_cases(
                #         self.eval_model,
                #         self.trainer.tokenizer.tokenizer,
                #         self.tokenizer_args,
                #         self.evaluation_dataset,
                #     )
                #     self.evaluation_dataset.test_cases = test_cases

                # Run benchmark evaluation if enabled and it's the right epoch
                if (self.enable_benchmarks and 
                    state.epoch is not None and 
                    (state.epoch) % self.benchmark_eval_frequency == 0):
                    
                    print(f"\n{'='*60}")
                    print(f"Epoch {state.epoch} Benchmark Evaluation")
                    print(f"{'='*60}")
                    
                    benchmark_results = self._run_benchmark_evaluation()
                    
                    if benchmark_results:
                        self.benchmark_results_history.append({
                            'epoch': state.epoch,
                            'step': state.global_step,
                            'results': benchmark_results
                        })
                        
                        # Log benchmark results to trainer
                        for metric_name, score in benchmark_results.items():
                            self.trainer.log({
                                f"benchmark/{metric_name}": score,
                                "epoch": state.epoch,
                                "step": state.global_step
                            })
                        
                        print(f"Benchmark results logged for epoch {state.epoch}")
    
    @torch.no_grad()
    def _run_benchmark_evaluation(self) -> Dict[str, float]:
        """
        Run benchmark evaluation on the current model
        """
        results = {}
        
        try:
            # Get current model and tokenizer from trainer
            model = self.eval_model
            tokenizer = self.trainer.tokenizer
            
            if tokenizer is None:
                print("Warning: No tokenizer available for benchmark evaluation")
                return results
            
            # Run text-based benchmarks using self as the model wrapper
            if 'mmlu' in self.benchmarks_to_run:
                try:
                    print("Running MMLU benchmark...")
                    mmlu_benchmark = MMLU(n_shots=3)
                    mmlu_benchmark.evaluate(model=self.eval_model)
                    results['mmlu'] = mmlu_benchmark.overall_score
                    print(f"MMLU Score: {mmlu_benchmark.overall_score:.4f}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"MMLU evaluation failed: {e}")
                    results['mmlu'] = 0.0
            
            if 'hellaswag' in self.benchmarks_to_run:
                try:
                    print("Running HellaSwag benchmark...")
                    hellaswag_benchmark = HellaSwag(n_shots=3)
                    hellaswag_benchmark.evaluate(model=self.eval_model)
                    results['hellaswag'] = hellaswag_benchmark.overall_score
                    print(f"HellaSwag Score: {hellaswag_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"HellaSwag evaluation failed: {e}")
                    results['hellaswag'] = 0.0
            
            if 'gsm8k' in self.benchmarks_to_run:
                try:
                    print("Running GSM8K benchmark...")
                    gsm8k_benchmark = GSM8K(n_problems=10, n_shots=3, enable_cot=True)
                    gsm8k_benchmark.evaluate(model=self.eval_model)
                    results['gsm8k'] = gsm8k_benchmark.overall_score
                    print(f"GSM8K Score: {gsm8k_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"GSM8K evaluation failed: {e}")
                    results['gsm8k'] = 0.0
            
            if 'truthfulqa' in self.benchmarks_to_run:
                try:
                    print("Running TruthfulQA benchmark...")
                    truthfulqa_benchmark = TruthfulQA()
                    truthfulqa_benchmark.evaluate(model=self.eval_model)
                    results['truthfulqa'] = truthfulqa_benchmark.overall_score
                    print(f"TruthfulQA Score: {truthfulqa_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"TruthfulQA evaluation failed: {e}")
                    results['truthfulqa'] = 0.0
            
            if 'arc' in self.benchmarks_to_run:
                try:
                    print("Running ARC benchmark...")
                    arc_benchmark = ARC(n_shots=3)
                    arc_benchmark.evaluate(model=self.eval_model)
                    results['arc'] = arc_benchmark.overall_score
                    print(f"ARC Score: {arc_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"ARC evaluation failed: {e}")
                    results['arc'] = 0.0
            
            # if 'piqa' in self.benchmarks_to_run:
            #     try:
            #         print("Running PIQA benchmark...")
            #         piqa_benchmark = PIQA(n_shots=3)
            #         piqa_benchmark.evaluate(model=self)
            #         results['piqa'] = piqa_benchmark.overall_score
            #         print(f"PIQA Score: {piqa_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"PIQA evaluation failed: {e}")
            #         results['piqa'] = 0.0
            
            if 'bigbenchhard' in self.benchmarks_to_run:
                try:
                    print("Running BigBenchHard benchmark...")
                    bigbench_benchmark = BigBenchHard(enable_cot=True)
                    bigbench_benchmark.evaluate(model=self.eval_model)
                    results['bigbenchhard'] = bigbench_benchmark.overall_score
                    print(f"BigBenchHard Score: {bigbench_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"BigBenchHard evaluation failed: {e}")
                    results['bigbenchhard'] = 0.0
            
            if 'humaneval' in self.benchmarks_to_run:
                try:
                    print("Running HumanEval benchmark...")
                    humaneval_benchmark = HumanEval(
                        tasks=[HumanEvalTask.HAS_CLOSE_ELEMENTS, HumanEvalTask.SORT_NUMBERS],
                        n=50
                    )
                    humaneval_benchmark.evaluate(model=self.eval_model)
                    results['humaneval'] = humaneval_benchmark.overall_score
                    print(f"HumanEval Score: {humaneval_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"HumanEval evaluation failed: {e}")
                    results['humaneval'] = 0.0
            
            if 'squad' in self.benchmarks_to_run:
                try:
                    print("Running SQuAD benchmark...")
                    squad_benchmark = SQuAD(
                        tasks=[SQuADTask.PHARMACY, SQuADTask.NORMANS],
                        n_shots=3
                    )
                    squad_benchmark.evaluate(model=self.eval_model)
                    results['squad'] = squad_benchmark.overall_score
                    print(f"SQuAD Score: {squad_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"SQuAD evaluation failed: {e}")
                    results['squad'] = 0.0
            
            if 'mathqa' in self.benchmarks_to_run:
                try:
                    print("Running MathQA benchmark...")
                    mathqa_benchmark = MathQA(
                        tasks=[MathQATask.PROBABILITY, MathQATask.GEOMETRY],
                        n_shots=3
                    )
                    mathqa_benchmark.evaluate(model=self.eval_model)
                    results['mathqa'] = mathqa_benchmark.overall_score
                    print(f"MathQA Score: {mathqa_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"MathQA evaluation failed: {e}")
                    results['mathqa'] = 0.0
            
            # if 'agieval' in self.benchmarks_to_run:
            #     try:
            #         print("Running AGIEval benchmark...")
            #         agieval_benchmark = AGIEval(n_shots=3)
            #         agieval_benchmark.evaluate(model=self.eval_model)
            #         results['agieval'] = agieval_benchmark.overall_score
            #         print(f"AGIEval Score: {agieval_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"AGIEval evaluation failed: {e}")
            #         results['agieval'] = 0.0
            
            # if 'openbookqa' in self.benchmarks_to_run:
            #     try:
            #         print("Running OpenBookQA benchmark...")
            #         openbookqa_benchmark = OpenBookQA(n_shots=3)
            #         openbookqa_benchmark.evaluate(model=self.eval_model)
            #         results['openbookqa'] = openbookqa_benchmark.overall_score
            #         print(f"OpenBookQA Score: {openbookqa_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"OpenBookQA evaluation failed: {e}")
            #         results['openbookqa'] = 0.0
            
            # if 'winogrande' in self.benchmarks_to_run:
            #     try:
            #         print("Running WinoGrande benchmark...")
            #         winogrande_benchmark = WinoGrande(n_shots=3)
            #         winogrande_benchmark.evaluate(model=self.eval_model)
            #         results['winogrande'] = winogrande_benchmark.overall_score
            #         print(f"WinoGrande Score: {winogrande_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"WinoGrande evaluation failed: {e}")
            #         results['winogrande'] = 0.0
            
            if 'boolq' in self.benchmarks_to_run:
                try:
                    print("Running BoolQ benchmark...")
                    boolq_benchmark = BoolQ(n_shots=3)
                    boolq_benchmark.evaluate(model=self.eval_model)
                    results['boolq'] = boolq_benchmark.overall_score
                    print(f"BoolQ Score: {boolq_benchmark.overall_score:.4f}")
                except Exception as e:
                    print(f"BoolQ evaluation failed: {e}")
                    results['boolq'] = 0.0
            
            # if 'cb' in self.benchmarks_to_run:
            #     try:
            #         print("Running CB benchmark...")
            #         cb_benchmark = CB(n_shots=3)
            #         cb_benchmark.evaluate(model=self.eval_model)
            #         results['cb'] = cb_benchmark.overall_score
            #         print(f"CB Score: {cb_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"CB evaluation failed: {e}")
            #         results['cb'] = 0.0
            
            # if 'copa' in self.benchmarks_to_run:
            #     try:
            #         print("Running COPA benchmark...")
            #         copa_benchmark = COPA(n_shots=3)
            #         copa_benchmark.evaluate(model=self.eval_model)
            #         results['copa'] = copa_benchmark.overall_score
            #         print(f"COPA Score: {copa_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"COPA evaluation failed: {e}")
            #         results['copa'] = 0.0
            
            # if 'multirc' in self.benchmarks_to_run:
            #     try:
            #         print("Running MultiRC benchmark...")
            #         multirc_benchmark = MultiRC(n_shots=3)
            #         multirc_benchmark.evaluate(model=self.eval_model)
            #         results['multirc'] = multirc_benchmark.overall_score
            #         print(f"MultiRC Score: {multirc_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"MultiRC evaluation failed: {e}")
            #         results['multirc'] = 0.0
            
            # if 'record' in self.benchmarks_to_run:
            #     try:
            #         print("Running ReCoRD benchmark...")
            #         record_benchmark = ReCoRD(n_shots=3)
            #         record_benchmark.evaluate(model=self.eval_model)
            #         results['record'] = record_benchmark.overall_score
            #         print(f"ReCoRD Score: {record_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"ReCoRD evaluation failed: {e}")
            #         results['record'] = 0.0
            
            # if 'rte' in self.benchmarks_to_run:
            #     try:
            #         print("Running RTE benchmark...")
            #         rte_benchmark = RTE(n_shots=3)
            #         rte_benchmark.evaluate(model=self.eval_model)
            #         results['rte'] = rte_benchmark.overall_score
            #         print(f"RTE Score: {rte_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"RTE evaluation failed: {e}")
            #         results['rte'] = 0.0
            
            # if 'wic' in self.benchmarks_to_run:
            #     try:
            #         print("Running WiC benchmark...")
            #         wic_benchmark = WiC(n_shots=3)
            #         wic_benchmark.evaluate(model=self.eval_model)
            #         results['wic'] = wic_benchmark.overall_score
            #         print(f"WiC Score: {wic_benchmark.overall_score:.4f}")
            #     except Exception as e:
            #         print(f"WiC evaluation failed: {e}")
            #         results['wic'] = 0.0
            
            # Run vision-based benchmark (MME)
            if 'mme' in self.benchmarks_to_run:
                try:
                    print("Running MME benchmark...")
                    mme_results = run_mme_evaluation(model, tokenizer, self.mme_max_samples)
                    if mme_results:
                        results.update(mme_results)
                        print(f"MME Overall Score: {mme_results.get('overall', 0):.2f}%")
                except Exception as e:
                    print(f"MME evaluation failed: {e}")
                    results['mme_overall'] = 0.0
            
        except Exception as e:
            print(f"Benchmark evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return results
        
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # super().on_log(args, state, control, **kwargs)
        pass
        
    def _generate_table(self):
        super()._generate_table()
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_train_end(args, state, control, **kwargs)
        
        # Save benchmark results history
        if self.benchmark_results_history:
            results_file = os.path.join(args.output_dir, "benchmark_results.json")
            try:
                with open(results_file, 'w') as f:
                    json.dump(self.benchmark_results_history, f, indent=2)
                print(f"Benchmark results saved to {results_file}")
            except Exception as e:
                print(f"Failed to save benchmark results: {e}")
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_train_begin(args, state, control, **kwargs)
