from rich.console import RenderableType
from rich.progress import ProgressColumn, Task
from rich.style import Style
from rich.table import Column
from rich.text import Text

from utils.fmt import (
    DurationHumanRedable,
    format_duration,
    format_large_num,
    format_large_num_si,
    format_time_human_readable,
)


def task_elapsed_till_last_step(task: Task) -> float | None:
    if task.start_time is None:
        return None
    with task._lock:
        progress = task._progress
        if not progress:
            return None
        last_step = progress[-1].timestamp
        return last_step - task.start_time


class SpeedColumn(ProgressColumn):
    # Refresh twice a second to prevent jitter
    max_refresh = 0.5

    def __init__(
        self,
        table_column: Column | None = None,
        style: str | Style = "blue",
    ):
        self.style = style
        super().__init__(table_column=table_column or Column(justify="right"))

    def render(self, task: Task) -> RenderableType:
        completed = task.completed
        elapsed = task_elapsed_till_last_step(task)
        if elapsed is None or elapsed == 0 or completed == 0:
            return Text("󱥸 it/s", style=self.style)
        speed = completed / elapsed
        if speed < 1:
            return Text(f"{format_time_human_readable(1 / speed)}/it", style=self.style)
        return Text(f"{format_large_num_si(speed)} it/s", style=self.style)


class ETAColumn(ProgressColumn):
    # Refresh twice a second to prevent jitter
    max_refresh = 0.5
    human_readable: DurationHumanRedable

    def __init__(
        self,
        table_column: Column | None = None,
        style: str | Style = "cyan",
        human_readable: DurationHumanRedable = "when-large",
    ):
        self.style = style
        self.human_readable = human_readable
        super().__init__(table_column=table_column)

    def render(self, task: Task) -> RenderableType:
        completed = task.completed
        total = task.total
        elapsed = task_elapsed_till_last_step(task)
        if elapsed is None or elapsed == 0 or completed == 0 or total is None:
            return Text("ETA ⁇ ", style=self.style)
        remaining = elapsed * (total - completed) / completed
        return Text(
            f"ETA {format_duration(remaining, human_readable=self.human_readable)}",
            style=self.style,
        )


class CompletionRatioColumn(ProgressColumn):
    human_readable: DurationHumanRedable

    def __init__(
        self,
        table_column: Column | None = None,
        style: str | Style = "green",
        human_readable: DurationHumanRedable = "when-large",
    ):
        self.style = style
        self.human_readable = human_readable
        super().__init__(table_column=table_column or Column(justify="right"))

    def render(self, task: Task) -> RenderableType:
        completed = task.completed
        completed_str = format_large_num(completed, human_readable=self.human_readable)
        total = task.total
        if total is None:
            return Text(completed_str, style=self.style)
        total_str = format_large_num(total, human_readable=self.human_readable)
        total_digits = len(str(int(total)))
        # Pad the completed string to the same length as the total, so that it won't
        # jump around when it increases.
        # When it's human readable, it might be shorter, but it should always at least
        # have 6 characters (up to 1 million before it's shortened).
        # But don't go up to 6 chars, if the maximum will never have this many digits.
        min_len = max(len(total_str), min(total_digits, 6))
        return Text(f"{completed_str:>{min_len}}/{total_str}", style=self.style)


class ElapsedColumn(ProgressColumn):
    human_readable: DurationHumanRedable

    def __init__(
        self,
        table_column: Column | None = None,
        style: str | Style = "yellow",
        human_readable: DurationHumanRedable = "when-large",
    ):
        self.style = style
        self.human_readable = human_readable
        super().__init__(table_column=table_column or Column(justify="right"))

    def render(self, task: Task) -> RenderableType:
        elapsed = task.finished_time if task.finished else task.elapsed
        return Text(
            format_duration(elapsed or 0, human_readable=self.human_readable),
            style=self.style,
        )
