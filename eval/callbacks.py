from typing import List, Dict, Optional
from deepeval.metrics import BaseMetric
from deepeval.evaluate.execute import execute_test_cases
from deepeval.dataset import EvaluationDataset
from deepeval.integrations.hugging_face import DeepEvalHuggingFaceCallback
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from transformers import (
        Trainer,
        TrainingArguments,
        TrainerState,
        TrainerControl
    )


def get_model_eval_callback(
    trainer: Trainer,
    evaluation_dataset: Optional[EvaluationDataset] = None,
    metrics: Optional[List[BaseMetric]] = None,
    tokenizer_args: Optional[Dict] = {},
    aggregation_method: str = "avg",
    show_table: bool = False,
): 
    
    if evaluation_dataset is None:
        # evaluation_dataset = EvaluationDataset(
        #     goldens=[
        #         Golden(input="..."),
        #         Golden(input="...")
        #     ]
        # )
        from deepeval.benchmarks import MMLU
        from deepeval.benchmarks.mmlu.task import MMLUTask
        from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
        from deepeval.test_case import ArenaTestCase, LLMTestCase
        from deepeval.metrics import ArenaGEval
        # mmlu = MMLU(tasks=[MMLUTask.HIGH_SCHOOL_EUROPEAN_HISTORY], n_shots=0)
        # evaluation_dataset = EvaluationDataset(
        #     goldens=mmlu.load_benchmark_dataset(task=MMLUTask.HIGH_SCHOOL_EUROPEAN_HISTORY)
        # )
        # print("Golden dataset: ","\n",  evaluation_dataset.goldens, "\n")
        evaluation_dataset = EvaluationDataset(
            test_cases=[
                LLMTestCase(
                    name="GPT-4",
                    input="What is the capital of France?",
                    actual_output="Paris",
                ),
                LLMTestCase(
                    name="Claude",
                    input="What is the capital of France?",
                    actual_output="Paris is the capital of France.",
                )
            ]
        )
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

    return ModelEvalCallback(
        trainer=trainer,
        evaluation_dataset=evaluation_dataset,
        metrics=metrics,
        tokenizer_args=tokenizer_args,
        aggregation_method=aggregation_method,
        show_table=show_table,
    )


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
        *args, 
        **kwargs
    )->None:
        super().__init__(
            trainer=trainer, 
            evaluation_dataset=evaluation_dataset, 
            metrics=metrics,
            tokenizer_args=tokenizer_args,
            show_table=show_table)
    
    def _calculate_metric_scores(self) -> Dict[str, List[float]]:
        return super()._calculate_metric_scores()

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
        
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_epoch_end(args, state, control, **kwargs)
        
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_log(args, state, control, **kwargs)
        
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
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_train_begin(args, state, control, **kwargs)
