import os
import pandas as pd
from tqdm import tqdm
from typing import Any, Tuple
import torch as T
from torch import nn
from torch.utils.data import DataLoader
from datasets.dataset_utils import DatasetHandler
from evaluation.metrics import setup_metric
from models.model_factory import BuildModel
from models.utils import ModelHandler
from utils.config_parser import ConfigParser
from utils.filesystem import extract_file_paths
from utils.helpers import filter_kwargs
from torcheval.metrics import Metric


class Evaluator:
    def __init__(self, metrics: list[str], device: str | None = None):
        self.metrics = metrics
        self.metrics_fn: dict[str, Metric[Any]] = {k: setup_metric(k) for k in metrics}
        self.device = device or "cpu"

    def _reset_metrics(self):
        for values in self.metrics_fn.values():
            values.reset()

    def evaluate(self, model: nn.Module, dl: DataLoader[Tuple[T.Tensor, int]], loss_fn: nn.Module | None = None) -> dict[str, Any]:
        model.eval()
        self._reset_metrics()
        loss = 0
        
        for input_, labels in tqdm(dl):
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

    def _compute_metrics(self) -> dict[str, float]:
        return {key: value.compute().item() for key, value in self.metrics_fn.items()}

    def calc_metrics(self, outputs: T.Tensor, labels: T.Tensor) -> dict[str, float]:
        self._reset_metrics()
        self._calc_one_step(outputs=outputs, labels=labels)
        metrics = self._compute_metrics()
        return metrics


class EvaluateProjectModels:
    def __init__(self, project_name: str):
        self.project_name = project_name

        self.df_list: list[dict[str, float]] = []
        self.configs_list: list[str] = []
        self.logs_dir = os.path.join("logs", project_name)
        self.checkpoints_dir = os.path.join("checkpoints", project_name)
        self.model_handler = ModelHandler()
        self.configs_list = extract_file_paths(self.logs_dir, "config.yaml")

    def _evaluate_single_model(self, config: ConfigParser):
        experiment_name = config["experiment_name"]
        device = config["misc"].get("device", "cpu") if T.cuda.is_available() else "cpu"
        evaluator = Evaluator(**filter_kwargs(Evaluator, config["evaluation"]), device=device)

        # build model
        model = BuildModel(**filter_kwargs(BuildModel, config["model"])).to(device)
        # load weights
        self.model_handler.load_weights(
            model=model,
            source="s3",
            version_name=experiment_name + "/model_best.pth",
            model_part="all",
            checkpoint_dir=config["checkpoint_dir"],
            project_name=self.project_name,
            s3_bucket_name=config["s3_bucket_name"]
        )
        
        for dataset in ["train", "val", "test"]:
            metrics = self._evaluate_dataset(
                model=model,
                evaluator=evaluator,
                dataset_config=config["dataset"],
                sub_dataset=dataset,
                experiment_name=experiment_name
            )
            self.df_list.append(metrics)

    @staticmethod
    def _evaluate_dataset(model: nn.Module, evaluator: Evaluator, dataset_config: dict[str, Any], sub_dataset: str, experiment_name: str) -> dict[str, float]:
        dataset_handler = DatasetHandler(dataset_config)
        dataloader = dataset_handler.create_dataloader(sub_dataset=sub_dataset, use_augmentations=False, shuffle=False)

        metrics = evaluator.evaluate(model=model, dl=dataloader)
        metrics["sub_dataset"] = sub_dataset
        metrics["experiment_name"] = experiment_name
        return metrics

    def evaluate(self):

        for config in self.configs_list:
            config_parser = ConfigParser(config)
            self._evaluate_single_model(config=config_parser)
        
        df = pd.DataFrame(self.df_list)
        return df
