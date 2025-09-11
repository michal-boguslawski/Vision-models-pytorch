import json
import logging
import os
import torch as T
from torch import nn
from typing import Any
from utils.aws_handler import AWSHandler
from utils.filesystem import remove_dir_with_content
from utils.metaclass import SingletonMeta


aws_handler = AWSHandler()


class SingletonLogger(metaclass=SingletonMeta):
    def __init__(self):
        self._initialized = False
        self.checkpoints_dir: str = "checkpoints"

    def initialize(
        self,
        project_name: str = "project_default",
        experiment_name: str = "experiment_default",
        checkpoints_dir: str = "checkpoints",
        logs_dir: str = "logs"
    ):
        if self._initialized:
            return  # already initialized

        self.checkpoints_dir = os.path.join(checkpoints_dir, project_name, experiment_name)
        log_file = os.path.join(logs_dir, project_name, experiment_name, "app.log")
        metrics_file = os.path.join(logs_dir, project_name, experiment_name, "metrics.json")
        
        remove_dir_with_content(self.checkpoints_dir)
        remove_dir_with_content(os.path.join(logs_dir, project_name, experiment_name))

        # create directories
        for path in [os.path.dirname(log_file), os.path.dirname(metrics_file), self.checkpoints_dir]:
            os.makedirs(path, exist_ok=True)

        self._setup_logger(log_file, metrics_file)
        self._initialized = True

    def _setup_logger(
        self,
        log_file: str,
        metrics_file: str
    ):
        self._logger = logging.getLogger("AppLogger")
        self._logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            # console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            self._logger.addHandler(console_handler)
            
            # file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self._logger.addHandler(file_handler)

        self.metrics_logger = logging.getLogger("MetricsLogger")
        self.metrics_logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.metrics_logger.handlers:
            
            metrics_handler = logging.FileHandler(metrics_file)
            metrics_handler.setLevel(logging.INFO)
            # Simple formatter: just the message (JSON)
            metrics_formatter = logging.Formatter("%(message)s")
            metrics_handler.setFormatter(metrics_formatter)
            self.metrics_logger.addHandler(metrics_handler)

    def get_logger(self) -> logging.Logger:
        if not hasattr(self, "logger"):
            raise RuntimeError("Logger not initialized. Call .initialize() first.")
        return self._logger

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            raise RuntimeError("Logger not initialized. Call .initialize() first.")
        return self._logger

    def log_metrics(self, metrics: dict[str, float]):
        """Write metrics only to metrics file."""
        self.metrics_logger.info(json.dumps(metrics))

    def log_artifact(self, filename: str, target: str, artifact: Any = None):
        if target == "s3":
            local_path = os.path.join(self.checkpoints_dir, filename)
            self.logger.info(f"Uploading {local_path} to S3")
            aws_handler.upload_file_to_s3(local_path=local_path)
        elif target == "local" and isinstance(artifact, nn.Module):
            from models.utils import ModelHandler
            model_handler = ModelHandler()
            model_handler.save_weights(model=artifact, filename=filename)
        