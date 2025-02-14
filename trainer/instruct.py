from dataset.batch import Batch
from metric.functional import classification_accuracy
from metric.metrics import CLASS_ACCURACY, CLASS_ACCURACY_UNCASED, TRAIN_LOSS, Metric

from .base import BaseTrainer
from .result import TrainOutput, ValidationOutput


class InstructTrainer(BaseTrainer):
    """
    A Trainer for any instruct model from HuggingFace.
    """

    def __init__(
        self,
        *args,
        max_new_tokens: int | None = None,
        ignore_index: int = -100,
        metrics: list[Metric] = [CLASS_ACCURACY, CLASS_ACCURACY_UNCASED, TRAIN_LOSS],
        **kwargs,
    ):
        super().__init__(metrics=metrics, *args, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.ignore_index = ignore_index

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

    def predict(self, batch: Batch) -> ValidationOutput:
        inputs = batch.data.to(self.hardware.device)
        unwrapped_model = self.unwrap_model()
        tokeniser = self.unwrap_tokeniser()
        outputs = unwrapped_model.generate(  # pyright: ignore[reportCallIssue]
            **inputs,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=tokeniser.pad_token_id,
        )
        preds = [
            tokeniser.decode(out[input_ids.size(0) :], skip_special_tokens=True)
            for input_ids, out in zip(inputs.input_ids, outputs)
        ]
        return ValidationOutput(
            metrics=dict(
                accuracy={
                    "class": classification_accuracy(
                        preds, batch.answers, ignore_case=False
                    ),
                    "class_uncased": classification_accuracy(
                        preds, batch.answers, ignore_case=True
                    ),
                },
            ),
            preds=preds,
            target=batch.answers,
            info=batch.info,
        )
