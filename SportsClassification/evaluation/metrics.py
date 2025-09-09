from torcheval.metrics import MulticlassAccuracy, Metric
from typing import Any


EVALUATION_METRICS_DICT: dict[str, Metric[Any]] = {
    "accuracy": MulticlassAccuracy(k=1),
    "top-5-accuracy": MulticlassAccuracy(k=5)
}


def setup_metric(metric_name: str) -> Metric[Any]:
    return EVALUATION_METRICS_DICT[metric_name]
