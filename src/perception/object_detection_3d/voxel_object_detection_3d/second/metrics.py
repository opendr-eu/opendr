from typing import Dict


class Metric:
    def __init__(self, value=None) -> None:
        self.value = value

    def log(self, target: str, prefix="", end="\n"):

        if target == "console":
            print(prefix + str(self.get()), end=end)
        else:
            with open(target, "a") as f:
                f.write(prefix + str(self.get()) + end)

    def print(self, prefix="", end="\n"):
        self.log("console", prefix, end)

    def get(self):
        return self.value

    def update(self, value):
        self.value = value

    def clear(self):
        self.value = None


class AverageMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.value = []

    def get(self):

        if (len(self.value) <= 0):
            return None

        return sum(self.value) / len(self.value)

    def update(self, value):
        self.value.append(value)

    def clear(self):
        self.value = []


class SumMetric(Metric):
    def __init__(self, value=0) -> None:
        super().__init__()
        self.initial_value = value
        self.value = self.initial_value

    def get(self):

        return self.value

    def update(self, value):
        self.value += value

    def clear(self):
        self.value = self.initial_value



class MaxMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.value = []

    def get(self):
        return max(self.value)

    def update(self, value):
        self.value.append(value)

    def clear(self):
        self.value = []


class MinMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

        self.value = []

    def get(self):
        return min(self.value)

    def update(self, value):
        self.value.append(value)

    def clear(self):
        self.value = []


class RangeMetric(Metric):
    def __init__(self, delimiter=", ", value_separator=": ") -> None:
        super().__init__()

        self.delimiter = delimiter
        self.value_separator = value_separator
        self.value = []

    def get(self):

        if len(self.value) > 0:

            values = {
                "min": min(self.value),
                "max": max(self.value),
                "avg": sum(self.value) / len(self.value),
            }
        else:
            values = {
                "min": "no values",
                "max": "no values",
                "avg": "no values",
            }

        result = []

        for name, metric in values.items():
            result.append(name + self.value_separator + str(metric))

        return self.delimiter.join(result)

    def update(self, value):
        self.value.append(value)

    def clear(self):
        self.value = []


class MetricGroup(Metric):
    def __init__(
        self, metrics: Dict[str, Metric], delimiter=", ", value_separator=": "
    ) -> None:
        self.metrics = metrics
        self.delimiter = delimiter
        self.value_separator = value_separator

    def update(self, values):

        for key, value in values.items():
            if key in self.metrics:
                self.metrics[key].update(value)
            else:
                raise ValueError(
                    "No metric '" + key + "' in this metric group"
                )

    def get(self):

        result = []

        for name, metric in self.metrics.items():
            result.append(str(name) + self.value_separator + str(metric.get()))

        return self.delimiter.join(result)

    def metric(self, name):
        return self.metrics[name]

    def clear(self):
        for metric in self.metrics.values():
            metric.clear()
