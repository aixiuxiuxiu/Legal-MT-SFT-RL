import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from unsloth import FastVisionModel
from wandb import Table
from wandb.sdk.wandb_run import Run as WandbRun

from dataset.batch import Batch
from debugger import breakpoint
from dist import sync_dict_values
from lr_scheduler import BaseLrScheduler
from metric import restore_dict_of_metrics, summarise_list_of_metrics
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
        hardware: HardwareManager = HardwareManager(),
        lr_scheduler: BaseLrScheduler | None = None,
        max_grad_norm: float = 1.0,
        num_epochs: int = 10,
        wandb: WandbRun | None = None,
    ):
        self.model = model
        self.optimiser = optimiser
        self.hardware = hardware
        self.processor = processor
        self.save_dir = Path(save_dir)
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.wandb = wandb
        self.example_table = Table(columns=["epoch", "path", "pred", "target"])

    # Unwraps a model to the core model, which can be across multiple layers with
    # wrappers such as DistributedDataParallel.
    def unwrap_model(self) -> nn.Module:
        model = self.model
        while hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
        return model

    def unwrap_tokeniser(self) -> PreTrainedTokenizerBase:
        tokeniser = (
            # FIXME: Fix this type annotation for Image/Text processors.
            # But this is at least safe.
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            and isinstance(self.processor.tokenizer, PreTrainedTokenizerBase)
            else self.processor
        )
        return tokeniser

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
        FastVisionModel.for_training(self.unwrap_model())
        num_replicas = set_sampler_epoch(data_loader, epoch=epoch)
        # Zeroing out the gradients here, because during the backward pass the zeroing
        # happens at the end, which saves the memory from it since the
        # zero_grad(set_to_none=True) (default) will eliminate the need to have the
        # gradients in memory, hence resetting them afterwards is beneficial.
        # But for the first step it needs to be done manually.
        self.optimiser.zero_grad()

        losses = []
        metrics = []
        pbar = tqdm(
            desc=self._epoch_text(epoch=epoch, desc="Train"),
            total=len(data_loader.dataset),  # pyright: ignore[reportArgumentType]
            leave=False,
            dynamic_ncols=True,
        )
        for batch in data_loader:
            # The last batch may not be a full batch
            curr_batch_size = batch.data["input_ids"].size(0)
            with self.hardware.autocast():
                output = self.forward(batch)
            losses.append(output.loss.item())
            self.backward(output.loss)
            metrics.append(output.metrics)
            pbar.update(curr_batch_size * num_replicas)
        pbar.close()

        mean_metrics = summarise_list_of_metrics(metrics)
        result = dict(
            loss=torch.mean(torch.tensor(losses)).item(),
            metrics={name: metric.to_dict() for name, metric in mean_metrics.items()},
        )
        # Gather the metrics onto the primary process
        result = sync_dict_values(result, device=self.hardware.device)
        return TrainResult(
            loss=result["loss"],
            metrics=restore_dict_of_metrics(result.get("metrics", {}), mean_metrics),
            lr=self.get_lr(),
        )

    @torch.no_grad()
    def validation_epoch(self, data_loader: DataLoader, epoch: int) -> ValidationResult:
        torch.set_grad_enabled(False)
        self.model.eval()
        # This is needed, otherwise some kv-cache issues occur.
        FastVisionModel.for_inference(self.unwrap_model())
        num_replicas = set_sampler_epoch(data_loader, epoch=epoch)

        metrics = []
        pbar = tqdm(
            desc=self._epoch_text(epoch=epoch, desc="Validation"),
            total=len(data_loader.dataset),  # pyright: ignore[reportArgumentType]
            leave=False,
            dynamic_ncols=True,
        )
        for batch in data_loader:
            # The last batch may not be a full batch
            curr_batch_size = batch.data["input_ids"].size(0)
            with self.hardware.autocast():
                output = self.predict(batch)
            metrics.append(output.metrics)
            pbar.update(curr_batch_size * num_replicas)
        pbar.close()

        mean_metrics = summarise_list_of_metrics(metrics)
        # Gather the metrics onto the primary process
        synced_metric = sync_dict_values(
            {name: metric.to_dict() for name, metric in mean_metrics.items()},
            device=self.hardware.device,
        )
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
            metrics=restore_dict_of_metrics(synced_metric, mean_metrics),
            examples=examples,
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

    @abstractmethod
    def predict(self, batch: Batch) -> ValidationOutput:
        raise NotImplementedError("predict method is not implemented")

    def save_pretrained(self, name: str) -> Path:
        path = self.save_dir / name
        if self.wandb:
            path.mkdir(parents=True, exist_ok=True)
            model = self.unwrap_model()
            # Unwrapping the module makes the type checking brittle, but this is
            # guaranteed to be any model that implements save_pretrained.
            model.save_pretrained(path, safe_serialization=True)
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
        metric_name: str = "class_accuracy",
    ):
        best_metric = None
        for epoch in range(self.num_epochs):
            train_result = self.train_epoch(train_data_loader, epoch=epoch)
            # TODO: Prettify
            print(f"## Train - Epoch {epoch + 1}")
            print(train_result)

            validation_result = self.validation_epoch(
                validation_data_loader, epoch=epoch
            )
            # TODO: Prettify
            print(f"## Validation - Epoch {epoch + 1}")
            print(validation_result.metrics)

            self.save_pretrained("latest")
            current_metric = validation_result.metrics[metric_name]
            if current_metric.is_better_than(current_metric):
                best_metric = current_metric
                self.save_pretrained("best")
                icon = "🔔"
                print(
                    f"{icon} New best checkpoint: Epoch {epoch + 1} — "
                    f"{metric_name} = {best_metric.get_value():.5f} {icon}"
                )

            if self.wandb:
                for example in validation_result.examples:
                    # Adding the row to the table, because only the most recent one is
                    # shown in the wandb interface, but it should be easy to compare
                    # them.
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
        if self.wandb:
            # Log the example over time in a table. wandb does not support updating the
            # table at each epoch, but only a "summary", so the whole table needs to be
            # logged at once.
            self.wandb.log(dict(example=self.example_table))
