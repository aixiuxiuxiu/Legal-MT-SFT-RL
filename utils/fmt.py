import math
from typing import Literal


def format_large_num_si(num: int | float) -> str:
    """
    Formats a large number with SI order of magnitude suffixes.

    Adapted from:
    https://github.com/tqdm/tqdm/blob/0ed5d7f18fa3153834cbac0aa57e8092b217cc16/tqdm/std.py#L370-L398

    Args:
        num (float): Number to format.

    Returns:
        out (str): Number with SI order of magnitude unit suffix.
    """
    for unit in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        abs_num = abs(num)
        if abs_num < 999.5:
            if abs_num < 99.95:
                if abs_num < 9.995:
                    return f"{num:1.2f}{unit}"
                return f"{num:2.1f}{unit}"
            return f"{num:3.0f}{unit}"
        num /= 1000
    return f"{num:3.1f}Y"


type DurationHumanRedable = Literal["always", "when-large", "never"]


def format_large_num(
    num: int | float, human_readable: DurationHumanRedable = "when-large"
) -> str:
    match human_readable:
        case "always":
            return format_large_num_si(num)
        case "when-large":
            # Human readable once it is over a 100k
            return format_large_num_si(num) if num >= 1e5 else f"{num:.0f}"
        case "never":
            return f"{num:.0f}"


def format_duration(
    seconds: int | float, human_readable: DurationHumanRedable = "when-large"
) -> str:
    match human_readable:
        case "always":
            return format_time_human_readable(seconds)
        case "when-large":
            # Human readable once it is over a day.
            return (
                format_time_human_readable(seconds)
                if seconds > 60 * 60 * 24
                else format_duration_simple(seconds)
            )
        case "never":
            return format_duration_simple(seconds)


def format_duration_simple(seconds: int | float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    else:
        return f"{mins:02d}:{secs:02d}"


def format_time_human_readable(seconds: int | float) -> str:
    if seconds == 0:
        return "0s"
    exponent = math.floor(math.log10(seconds))
    #  Use ns if it is smaller than 1e-9 (should be impossible when using timestamps).
    if exponent < -9:
        time = seconds * 1e9
        return f"{time:g}ns"
    elif exponent < 0:
        units = ["ms", "μs", "ns"]
        # The index is calcualted such that every 10**(-3) it changes.
        # This needs + 1, since 1e-1 is the beginning, not 1e0.
        index = abs(exponent + 1) // 3
        # Use the seleted unit
        unit = units[index]
        # The seconds need to be multiplied by one order of maginute larger than the
        # selected unit, as it goes in the other direction.
        # For example: 1e-5 => 10μs == 1e-5 * 1e6 (which is the index = 1)
        time = seconds * 10 ** (3 * (index + 1))
        return f"{time:.0f}{unit}"
    else:
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)

        parts = []
        if days > 0:
            parts.append(f"{days:.0f}d")
        if hours > 0:
            parts.append(f"{hours:.0f}h")
        if mins > 0:
            parts.append(f"{mins:.0f}m")
        if secs > 0:
            parts.append(f"{secs:.0f}s")
        # Pad all elements except the first one to have at least 3 characters, as all of
        # them can have at most 2 digits (plus the unit), so that it keeps a consistent
        # width.
        parts = [p if i == 0 else f"{p:0>3}" for i, p in enumerate(parts)]
        return "".join(parts)
