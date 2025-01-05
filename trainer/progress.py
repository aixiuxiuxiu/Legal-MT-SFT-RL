from collections.abc import Generator
from contextlib import contextmanager
from types import TracebackType
from typing import Literal, Self

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
)
from rich.status import Status

from .progress_columns import (
    CompletionRatioColumn,
    ElapsedColumn,
    ETAColumn,
    SpeedColumn,
)

type TaskName = Literal["total", "train", "validation"]


class TrainerProgress:
    """
    A progress manager for the trainer. This is mostly a wrapper around rich's
    Progress, but customised and simplified to be used for the trainer.

    Two progress bars are shown simultaneously:
        - Total: Shows the total time elapsed for the epochs
        - Train/Validation: Shows the progress in each train/validation epoch (batches)

    It looks roughly like this:

    [ 1/10]   Total   0% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    0/10 • 0:01:09 • ETA -:--:--
    Epoch 1 - Train   4% ━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━  36/800 • 0:01:09 • ETA 0:18:08
    """

    live: Live
    status: Status
    pbar: Progress
    total: TaskID
    train: TaskID
    validation: TaskID
    epoch: int

    def __init__(
        self,
        num_epochs: int,
        console: Console = Console(),
        start_epoch: int = 1,
    ):
        self.status = Status("", console=console)
        self.pbar = Progress(
            TextColumn("{task.fields[prefix]}"),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(
                bar_width=None,
                style="dim",
                complete_style="none",
            ),
            CompletionRatioColumn(),
            TextColumn("[dim]•[/dim]"),
            ElapsedColumn(),
            TextColumn("[dim]•[/dim]"),
            ETAColumn(human_readable="always"),
            TextColumn("[dim]•[/dim]"),
            SpeedColumn(),
            console=console,
        )
        self.total = self.pbar.add_task("Total", start=False, prefix="")
        self.train = self.pbar.add_task("Train", start=False, visible=False, prefix="")
        self.validation = self.pbar.add_task(
            "Validation", start=False, visible=False, prefix=""
        )
        self.epoch = start_epoch
        self.num_epochs = num_epochs
        self.live = Live(Group(self.pbar), console=console)

    def __enter__(self) -> Self:
        self.start("total", total=self.num_epochs)
        self.live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        self.live.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def _get_task(self, name: TaskName) -> TaskID:
        if name == "total":
            return self.total
        elif name == "train":
            return self.train
        elif name == "validation":
            return self.validation

    def _get_prefix(self, name: TaskName) -> str:
        current = self.epoch
        end = self.num_epochs
        pad = len(str(self.num_epochs))
        if name == "total":
            text = f"\\[{current:>{pad}}/{end}]"
        else:
            text = f"Epoch {current} -"
        return text

    def end_epoch(self):
        if self.epoch < self.num_epochs:
            self.epoch += 1
        self.pbar.advance(self.total)
        self.pbar.update(self.total, prefix=self._get_prefix("total"))

    def start(self, name: TaskName, total: int, reset: bool = True):
        task = self._get_task(name)
        prefix = self._get_prefix(name)
        if reset:
            self.pbar.reset(task, start=True, visible=True, total=total, prefix=prefix)
        else:
            self.pbar.update(task, visible=True, total=total, prefix=prefix)
            self.pbar.start_task(task)

    def stop(self, name: TaskName):
        task = self._get_task(name)
        self.pbar.update(task, visible=False)

    def advance(self, name: TaskName, num: int = 1):
        task = self._get_task(name)
        self.pbar.advance(task, advance=num)

    @contextmanager
    def spinner(self, text: str) -> Generator[Status]:
        self.start_spinner(text)
        yield self.status
        self.stop_spinner()

    def start_spinner(self, text: str):
        self.status.update(text)
        self.live.update(Group(self.pbar, self.status), refresh=True)

    def stop_spinner(self):
        self.status.stop()
        self.live.update(Group(self.pbar), refresh=True)
