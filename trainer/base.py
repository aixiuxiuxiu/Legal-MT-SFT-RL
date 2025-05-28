import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import torch
import torch.nn as nn
import torch.optim as optim
from progrich import ProgressBar, Spinner
from rich.console import Console
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from unsloth import FastModel
from wandb import Table
from wandb.sdk.wandb_run import Run as WandbRun

from dataset.batch import Batch
from dataset.prefill import prefix_completions_with_prefill
from debugger import breakpoint
from dist import is_main, sync_dict_values
from lr_scheduler import BaseLrScheduler
from metric import (
    Metric,
    MetricTracker,
)
from metric.functional import classification_accuracy
from metric.table import table_from_metrics
from model.utils import unwrap_model, unwrap_tokeniser
from reward.classification import extract_answer
from utils import nested_dict
from utils.hardware import HardwareManager

from .result import (
    Example,
    TrainOutput,
    TrainResult,
    ValidationOutput,
    ValidationResult,
)
from .utils import set_sampler_epoch


class BaseTrainer(ABC):
    """
    A Trainer to handle the training loops and make it easier to extend it with custom
    losses. This is very similar to something like PyTorch Lightning's module, except
    that this is integrated into the Trainer class rather than the Module itself.
    This separates it completely from the Model, so not only can multiple models use the
    same training strategy, but it also means that a Model that is only used for
    inference does not need to define things that are purely for training
    e.g. Lightning requires to create the optimiser from the Module, which needs to know
    the learning rate etc., so if you want to be able to customise it, you would need to
    accept a parameter, which would be useless during inference)

    Furthermore, Lightning abstracts away a lot of things, which caused to also swallow
    up some errors, so I'd rather use the custom code (that I have been using anyway,
    but simply is this style/class) and having some non-zero overhead since it needs to
    cover all possible use cases while including a lot of checks to ensure the users
    utilise it correctly.

    Also this does not do all the multi-processing (DDP) stuff, it that is handled
    separately in the train script, this is really just for the model interactions.
    """

    def __init__(
        self,
        model: nn.Module,
        optimiser: optim.Optimizer,
        processor: PreTrainedTokenizerBase,
        save_dir: str | os.PathLike,
        metrics: list[Metric],
        hardware: HardwareManager = HardwareManager(),
        lr_scheduler: BaseLrScheduler | None = None,
        max_grad_norm: float = 1.0,
        num_epochs: int = 10,
        max_new_tokens: int | None = None,
        ignore_index: int = -100,
        prefill: str | None = None,
        wandb: WandbRun | None = None,
        console: Console = Console(),
    ):
        self.model = model
        self.optimiser = optimiser
        self.hardware = hardware
        self.processor = processor
        self.save_dir = Path(save_dir)
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.max_new_tokens = max_new_tokens
        self.ignore_index = ignore_index
        self.metrics = metrics
        self.prefill = prefill
        self.wandb = wandb
        self.console = console
        # The progress bar is defined here because the second progress bar
        # (train/validation) needs to be attached to this one, which take place in two
        # different methods.
        self.progress = ProgressBar("Total", total=num_epochs, persist=True)
        self.example_table = Table(columns=["epoch", "path", "pred", "target"])

        best_metric = None
        for metric in self.metrics:
            if metric.when != "train":
                best_metric = metric
                break
        if best_metric is None:
            raise ValueError(
                "No validation metric available to use as best metric "
                f"given: metrics={self.metrics!r}."
            )
        self.best_metric = best_metric

    def unwrap_model(self) -> nn.Module:
        return unwrap_model(self.model)

    def unwrap_tokeniser(self) -> PreTrainedTokenizerBase:
        return unwrap_tokeniser(self.processor)

    def get_lr(self) -> float:
        return (
            self.lr_scheduler.lr
            if self.lr_scheduler
            else self.optimiser.param_groups[0]["lr"]
        )

    def _epoch_text(self, epoch: int, desc: str | None = None) -> str:
        current = epoch + 1
        end = self.num_epochs
        pad = len(str(self.num_epochs))
        text = f"[{current:>{pad}}/{end}] Epoch {current}"
        if desc:
            text = f"{text} - {desc}"
        return text

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> TrainResult:
        torch.set_grad_enabled(True)
        self.model.train()
        # Needed to revert the inference mode.
        FastModel.for_training(self.unwrap_model())
        num_replicas = set_sampler_epoch(data_loader, epoch=epoch)
        # Zeroing out the gradients here, because during the backward pass the zeroing
        # happens at the end, which saves the memory from it since the
        # zero_grad(set_to_none=True) (default) will eliminate the need to have the
        # gradients in memory, hence resetting them afterwards is beneficial.
        # But for the first step it needs to be done manually.
        self.optimiser.zero_grad()

        losses = []
        start_time = time.time()
        metrics = MetricTracker(self.metrics, when="train")
        pbar = ProgressBar(
            "Train",
            total=len(data_loader.dataset),  # pyright: ignore[reportArgumentType]
            prefix=f"Epoch {epoch + 1} -",
            # Attach it to the total progress bar.
            progress=self.progress,
        )
        pbar.start()
        i = 0
        spinner = Spinner("Waiting for results of first batch...")
        spinner.start()
        for batch in data_loader:
            i += 1
            # The last batch may not be a full batch
            curr_batch_size = batch.data["input_ids"].size(0)
            with self.hardware.autocast():
                output = self.forward(batch)
            self.backward(output.loss)
            metrics.append(output.metrics)
            losses.append(output.loss.item())
            spinner.update(
                f"Current Batch: {batch.data['input_ids'].size()} • Loss {losses[-1]} • Avg loss {torch.mean(torch.tensor(losses, dtype=torch.float))}"
            )

            pbar.advance(curr_batch_size * num_replicas)
        spinner.stop()
        pbar.stop()

        mean_metrics = metrics.mean()
        # Gather the metrics onto the primary process
        mean_metrics = sync_dict_values(mean_metrics, device=self.hardware.device)
        return TrainResult(
            lr=self.get_lr(),
            metrics=mean_metrics,
            time_elapsed=time.time() - start_time,
        )

    @torch.no_grad()
    def validation_epoch(self, data_loader: DataLoader, epoch: int) -> ValidationResult:
        torch.set_grad_enabled(False)
        self.model.eval()
        # This is needed, otherwise some kv-cache issues occur.
        FastModel.for_inference(self.unwrap_model())
        num_replicas = set_sampler_epoch(data_loader, epoch=epoch)

        start_time = time.time()
        metrics = MetricTracker(self.metrics, when="validation")
        pbar = ProgressBar(
            "Validation",
            total=len(data_loader.dataset),  # pyright: ignore[reportArgumentType]
            prefix=f"Epoch {epoch + 1} -",
            # Attach it to the total progress bar.
            progress=self.progress,
        )
        pbar.start()
        for batch in data_loader:
            # The last batch may not be a full batch
            curr_batch_size = batch.data["input_ids"].size(0)
            with self.hardware.autocast():
                output = self.predict(batch)
            metrics.append(output.metrics)
            pbar.advance(curr_batch_size * num_replicas)
        pbar.stop()

        mean_metrics = metrics.mean()
        # Gather the metrics onto the primary process
        mean_metrics = sync_dict_values(mean_metrics, device=self.hardware.device)
        # TODO: Adapt this to work for more general cases.
        # The type ignores are here because pyright cannot guarantee that the loop
        # is executed at least once. But we know that is always the case, so the
        # variables are always bound.
        examples = [
            Example(
                path=str(path),
                pred=pred,
                target=target,
            )
            for pred, target, path in zip(
                output.preds,  # pyright: ignore[reportPossiblyUnboundVariable]
                output.target,  # pyright: ignore[reportPossiblyUnboundVariable]
                output.info["path"],  # pyright: ignore[reportPossiblyUnboundVariable]
            )
        ]
        return ValidationResult(
            metrics=mean_metrics,
            examples=examples,
            time_elapsed=time.time() - start_time,
        )

    @abstractmethod
    def forward(self, batch: Batch) -> TrainOutput:
        raise NotImplementedError("forward method is not implemented")

    def backward(self, loss: torch.Tensor):
        if torch.isnan(loss) or torch.isinf(loss):
            breakpoint("Loss is NaN")
        if self.lr_scheduler is not None:
            self.lr_scheduler.adjust_lr()
        if self.hardware.grad_scaler is None:
            loss.backward()
            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.optimiser.step()
        else:
            self.hardware.grad_scaler.scale(loss).backward()
            self.hardware.grad_scaler.unscale_(self.optimiser)
            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            )
            self.hardware.grad_scaler.step(self.optimiser)
            self.hardware.grad_scaler.update()
        # Zero out the gradients at the end (set_to_none=True by default) to save memory
        # of the gradients since they are now set to None.
        self.optimiser.zero_grad()

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
        preds = prefix_completions_with_prefill(preds, prefill=self.prefill)
        pred_answers = [extract_answer(pred) or pred for pred in preds]
        return ValidationOutput(
            metrics=dict(
                accuracy={
                    "class": classification_accuracy(
                        pred_answers, batch.answers, ignore_case=False
                    ),
                    "class_uncased": classification_accuracy(
                        pred_answers, batch.answers, ignore_case=True
                    ),
                },
            ),
            preds=preds,
            target=batch.answers,
            info=batch.info,
        )

    def save_pretrained(self, name: str) -> Path:
        path = self.save_dir / name
        if is_main():
            path.mkdir(parents=True, exist_ok=True)
            model = self.unwrap_model()
            # Unwrapping the module makes the type checking brittle, but this is
            # guaranteed to be any model that implements save_pretrained.
            model.save_pretrained(path, safe_serialization=True)  # pyright: ignore[reportCallIssue]
            self.processor.save_pretrained(path)
        return path

    def to(self, device: torch.device) -> Self:
        self.hardware.to(device)
        self.model.to(self.hardware.device)
        return self

    def train(
        self,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader,
    ):
        best_metric_value = None
        with self.progress:
            epoch_pad = len(str(self.num_epochs))
            for epoch in range(self.num_epochs):
                start_time = time.time()
                self.progress.update(
                    prefix=f"\\[{epoch + 1:>{epoch_pad}}/{self.num_epochs}]"
                )
                train_result = self.train_epoch(train_data_loader, epoch=epoch)

                validation_result = self.validation_epoch(
                    validation_data_loader, epoch=epoch
                )

                with Spinner("Saving model and tokeniser"):
                    self.save_pretrained("latest")
                    current_metric = nested_dict.get_recursive(
                        validation_result.metrics, self.best_metric.key
                    )
                    if not isinstance(current_metric, (int, float)):
                        raise KeyError(
                            "Cannot get value of validation metric for "
                            f"key={self.best_metric.key!r}, got {current_metric}."
                        )
                    is_new_best = self.best_metric.is_new_best(
                        best_metric_value, current_metric
                    )
                    if is_new_best:
                        best_metric_value = current_metric
                        self.save_pretrained("best")

                if self.wandb:
                    with Spinner("Logging to Weights & Biases"):
                        for example in validation_result.examples:
                            # Adding the row to the table, because only the most recent
                            # one is shown in the wandb interface, but it should be easy
                            # to compare them.
                            self.example_table.add_data(
                                epoch + 1,
                                example.path,
                                example.pred,
                                example.target,
                            )
                        self.wandb.log(
                            dict(
                                epoch=epoch + 1,
                                train=train_result.to_log_dict(),
                                validation=validation_result.to_log_dict(),
                            ),
                        )

                table = table_from_metrics(
                    train=train_result,
                    validation=validation_result,
                    metrics=self.metrics,
                    title=f"Epoch {epoch + 1}",
                    lr_scheduler=self.lr_scheduler,
                    time_elapsed=time.time() - start_time,
                )
                self.console.print(table)

                # The new best checkpoint should be printed after the table, however
                # saving it should be included in the timing, hence the reporting is
                # split off from the checkpoint above.
                if is_new_best:
                    icon = "🔔"
                    self.console.print(
                        f"{icon} New best checkpoint: Epoch {epoch + 1} — "
                        f"{self.best_metric.name} = {best_metric_value:.5f} {icon}"
                    )

                self.progress.advance()
            if self.wandb:
                # Log the example over time in a table. wandb does not support updating
                # the table at each epoch, but only a "summary", so the whole table
                # needs to be logged at once.
                self.wandb.log(dict(example=self.example_table))
