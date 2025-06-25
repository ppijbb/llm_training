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
    tokenizer_args: Optional[Dict] = None,
    aggregation_method: str = "avg",
    show_table: bool = False,
): 

    
    if evaluation_dataset is None:
        first_golden = Golden(input="...")
        second_golden = Golden(input="...")
        evaluation_dataset = EvaluationDataset(goldens=[first_golden, second_golden])
        # evaluation_dataset = EvaluationDataset(
        #     golden_dataset=Golden(
        #         dataset_name="",
        #         dataset_config_name="",
        #     )
        # )
    if metrics is None:
        metrics = [ GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            # NOTE: you can only provide either criteria or evaluation_steps, and not both
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        )]
    if tokenizer_args is None:
        tokenizer_args = {}
    if aggregation_method is None:
        aggregation_method = "avg"

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
