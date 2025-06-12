from deepeval.metrics import BaseMetric
from deepeval.evaluate.execute import execute_test_cases
from deepeval.dataset import EvaluationDataset
from deepeval.integrations.hugging_face import DeepEvalHuggingFaceCallback
from transformers import (
        TrainerCallback,
        ProgressCallback,
        Trainer,
        TrainingArguments,
        TrainerState,
        TrainerControl,
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
        evaluation_dataset: EvaluationDataset = None,
        metrics: List[BaseMetric] = None,
        tokenizer_args: Dict = None,
        aggregation_method: str = "avg",
        show_table: bool = False,
        *args, 
        **kwargs
    )->None:
        super().__init__(evaluation_dataset, metrics, trainer)
    
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
