from dataset.batch import Batch
from metric.metrics import TRAIN_LOSS, TRANSLATION_CHRF, Metric

from .base import BaseTrainer
from .result import TrainOutput


class InstructTrainer(BaseTrainer):
    """
    A Trainer for any instruct model from HuggingFace.
    """

    def __init__(
        self,
        *args,
        metrics: list[Metric] = [TRANSLATION_CHRF, TRAIN_LOSS],
        **kwargs,
    ):
        super().__init__(metrics=metrics, *args, **kwargs)

    def forward(self, batch: Batch) -> TrainOutput:
        inputs = batch.data.to(self.hardware.device)
        outputs = self.model(**inputs)
        # TODO: Ensure accuracy is calculated correctly
        # from debugger import breakpoint
        # breakpoint("Train Forward Predictions")
        # _, pred = torch.max(outputs.logits, dim=-1)
        return TrainOutput(
            loss=outputs.loss,
            metrics=dict(
                loss=outputs.loss.item(),
                # TODO: Wait of unsloth to fix the logits...
                # accuracy=TokenAccuracy.compute(
                #     pred,
                #     inputs["labels"],  # pyright: ignore[reportArgumentType]
                #     ignore_index=self.ignore_index,
                # )
            ),
            info=batch.info,
        )
