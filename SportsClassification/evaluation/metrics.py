from torcheval.metrics import MulticlassAccuracy, Metric


EVALUATION_METRICS_DICT = {
    "accuracy": MulticlassAccuracy(k=1),
    "top-5-accuracy": MulticlassAccuracy(k=5)
}


def setup_metric(metric_name: str) -> Metric:
    return EVALUATION_METRICS_DICT[metric_name]
