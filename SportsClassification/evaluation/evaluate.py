import torch as T
from torch import nn
from torch.utils.data import DataLoader
from evaluation.metrics import setup_metric


class Evaluator:
    def __init__(self, metrics: list[str], device: str | None = None):
        self.metrics = metrics
        self.metrics_fn = {k: setup_metric(k) for k in metrics}
        self.device = device or "cpu"

    def _reset_metrics(self):
        for values in self.metrics_fn.values():
            values.reset()

    def evaluate(self, model: nn.Module, dl: DataLoader, loss_fn: nn.modules.loss._Loss | None = None):
        model.eval()
        self._reset_metrics()
        loss = 0
        
        for input_, labels in dl:
            input_, labels = input_.to(self.device), labels.to(self.device)
            with T.no_grad():
                outputs = model(input_)
            
            self._calc_one_step(outputs=outputs, labels=labels)
            if loss_fn:
                loss += loss_fn(outputs, labels).item()
        metrics = self._compute_metrics()
        if loss_fn:
            metrics["loss"] = loss / len(dl)
        return metrics

    def _calc_one_step(self, outputs: T.Tensor, labels: T.Tensor):
        for metric_fn in self.metrics_fn.values():
            metric_fn.update(outputs, labels)

    def _compute_metrics(self):
        return {key: value.compute().item() for key, value in self.metrics_fn.items()}

    def calc_metrics(self, outputs: T.Tensor, labels: T.Tensor):
        self._reset_metrics()
        self._calc_one_step(outputs=outputs, labels=labels)
        metrics = self._compute_metrics()
        return metrics
