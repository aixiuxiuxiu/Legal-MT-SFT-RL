from rich import box
from rich.table import Table

from lr_scheduler.base import BaseLrScheduler
from trainer.result import TrainResult, ValidationResult
from utils import nested_dict
from utils.nested_dict import NestedDict

from .metrics import Metric


def format_number(number: float | None, precision: int = 5) -> str:
    if number is None:
        return "-"
    return f"{number:.{precision}f}"


def get_formatted_number_from_nested_dict(
    d: NestedDict[float], key: str, precision: int = 5
) -> str:
    value = nested_dict.get_recursive(d, key)
    if value is not None and not isinstance(value, (int, float)):
        raise TypeError(
            "Values to log to a table must contain a single float at "
            f"got {value} at key={key!r}."
        )
    return format_number(value, precision=precision)


def table_from_metrics(
    train: TrainResult,
    validation: ValidationResult,
    metrics: list[Metric],
    title: str | None = None,
    lr_scheduler: BaseLrScheduler | None = None,
) -> Table:
    caption = "Learning Rate = {lr:.8f} ≈ {lr:.4e}".format(lr=train.lr)
    if lr_scheduler and lr_scheduler.is_warmup():
        caption += " [🌡️ {step}/{warmup_steps} - {percent:.0%}]".format(
            step=lr_scheduler.step,
            warmup_steps=lr_scheduler.warmup_steps,
            percent=lr_scheduler.step / lr_scheduler.warmup_steps,
        )
    table = Table(
        title=title,
        box=box.HORIZONTALS,
        border_style="dim",
        caption=caption,
        caption_style="",
        min_width=60,
    )
    table.add_column("Name")
    train_row = ["Train"]
    validation_row = ["Validation"]
    for metric in metrics:
        table.add_column(metric.get_short_name(), justify="right")
        train_row.append(
            get_formatted_number_from_nested_dict(train.metrics, key=metric.key)
        )
        validation_row.append(
            get_formatted_number_from_nested_dict(validation.metrics, key=metric.key)
        )
    table.add_row(*train_row)
    table.add_row(*validation_row)
    return table
