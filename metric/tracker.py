import torch

from utils import nested_dict
from utils.nested_dict import NestedDict

from .metrics import Metric, MetricWhen


class MetricTracker:
    metrics: list[Metric]
    keys: list[str]
    metrics_dict: NestedDict[Metric]
    values: NestedDict[list[float]]

    def __init__(self, metrics: list[Metric], when: MetricWhen | None = None):
        self.metrics = metrics
        self.keys = []
        # For named access to get the metric
        self.metrics_dict = {}
        self.values = {}
        for metric in metrics:
            if when and when != metric.when:
                continue
            self.keys.append(metric.key)
            nested_dict.set_recursive(self.values, metric.key, [])
            nested_dict.set_recursive(self.metrics_dict, metric.key, metric)

    def get(
        self, key: str | None = None
    ) -> list[float] | NestedDict[list[float]] | None:
        """
        Get the value for the nested key out of the tracker. If no key is given, it
        returns the whole tracked values as a dictionary,

        Args:
            key (str, optional): Key to lookup. If not given returns the whole
                dictionary.

        Returns:
            value (list[float] | NestedDict[list[float]]): The value of the given key.
        """
        if key is None:
            return self.values
        return nested_dict.get_recursive(self.values, key)

    def append(self, values: NestedDict[float]):
        """
        Appends a set of values to the tracker. The values are given as a dictionary
        with the same hierarchy as the tracker, but with a single value.

        Args:
            values (NestedDict[float]): Nested dictionary with the stats to be added.
        """
        for key in self.keys:
            curr_values = nested_dict.get_recursive(self.values, key)
            if not isinstance(curr_values, list):
                raise KeyError(
                    f"Cannot access metric with key={key!r}, got {values}. "
                    f"Only available metrics are {self.keys}."
                )
            new_value = nested_dict.get_recursive(values, key)
            if not isinstance(new_value, (int, float)):
                raise TypeError(
                    f"Values to append must contain a single float at key={key!r}, "
                    f"got {new_value}."
                )
            curr_values.append(new_value)

    def mean(self) -> NestedDict[float]:
        """
        Calculates the mean values for all tracked metrics across all time steps.

        Returns:
            out (NestedDict[float]): Nested dictionary with the mean values for all
                tracked metrics.
        """
        out: NestedDict[float] = {}
        for key in self.keys:
            value = self.get(key)
            if not isinstance(value, list):
                raise KeyError(f"Cannot get last value for key={key!r}, got {out}.")
            # Mean cannot be calculated on an empty list, so leave it out of the dict.
            # But that is still valid.
            if len(value) == 0:
                continue
            mean_value = torch.mean(torch.tensor(value, dtype=torch.float)).item()
            nested_dict.set_recursive(out, key, mean_value)
        return out

    def last(self) -> NestedDict[float]:
        """
        Gives the last value for each tracked metric.

        Returns:
            out (NestedDict[float]): Nested dictionary with the last value for each
                tracked metrics.
        """
        out: NestedDict[float] = {}
        for key in self.keys:
            value = self.get(key)
            if not isinstance(value, list):
                raise KeyError(f"Cannot get last value for key={key!r}, got {out}.")
            # Cannot get last value from an empty list, so leave it out of the dict.
            # But that is still valid.
            if len(value) == 0:
                continue
            last_value = value[-1]
            nested_dict.set_recursive(out, key, last_value)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.keys!r})"
