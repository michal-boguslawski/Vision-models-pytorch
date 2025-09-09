import json
import logging
import numpy as np
import os
import sys
import torch as T
from torch import nn
from utils.config_parser import ConfigParser
from utils.filesystem import make_dirs, remove_dir_with_content, flush_cache
from utils.helpers import load_dict_and_append
from utils.aws_handler import AWSHandler


class Logger:
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        checkpoint_dir: str = "checkpoints",
        save_best_only: bool = True,
        save_checkpoint: bool = True,
        log_dir: str = "logs",
        log_subdirs: dict[str, str] | None = None,
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.save_checkpoint = save_checkpoint
        self.log_dir = log_dir
        self.log_subdirs: dict[str, str] = log_subdirs or {}

        self.best_val_loss = None
        self.experiment_log_dir = ""
        self.experiment_checkpoint_dir = ""
        
        self._setup()
        self._setup_logger()
        flush_cache() 

    def _setup_logger(self):
        log_path = os.path.join(self.experiment_log_dir, "train", "training.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)  # still prints to terminal
            ]
        )

    def _setup(self):
        # create directories for logs
        self.experiment_log_dir = os.path.join(self.log_dir, self.project_name, self.experiment_name)
        remove_dir_with_content(self.experiment_log_dir)
        make_dirs([self.experiment_log_dir])
        
        # create directories for checkpoints
        if self.save_checkpoint:
            self.experiment_checkpoint_dir = os.path.join(self.checkpoint_dir, self.project_name, self.experiment_name)
            remove_dir_with_content(self.experiment_checkpoint_dir)
            make_dirs([self.experiment_checkpoint_dir])

    def log_config(self, config: ConfigParser):
        config_path = os.path.join(self.experiment_log_dir, "config.yaml")
        config.save(config_path)

    def log_metrics(self, new_metrics: dict[str, list[float] | float ], sub_dataset: str = "train"):
        log_subdirs: str = self.log_subdirs[sub_dataset]
        metrics_path = os.path.join(self.experiment_log_dir, log_subdirs, "metrics.json")
        metrics = load_dict_and_append(metrics_path, new_metrics)
        
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        self.log_info(f"New metrics saved to {metrics_path}")
        self.log_info(" ".join([f"{key}: {np.mean(value):.6f}" for key, value in new_metrics.items()]))

    def log_info(self, message: str):
        logging.info(message)

    def log_artifact(self, target: str = "s3", s3_bucket_name: str | None = None):
        if target == "s3" and s3_bucket_name:
            self._save_weights_to_s3(s3_bucket_name=s3_bucket_name)

    def _save_weights(self, model: nn.Module, path: str):
        T.save(model.state_dict(), path)        

    def on_epoch_end_log(self, model: nn.Module, val_loss: float | None = None, epoch: int | None = None, end: bool = False):
        if self.save_best_only and self.save_checkpoint and val_loss:
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(self.experiment_checkpoint_dir, "model_best.pth")
                self._save_weights(model=model, path=checkpoint_path)
                self.log_info(f"Model saved. New best value {val_loss:.6f}")
        elif ( not self.save_checkpoint ) and end:
            checkpoint_path = os.path.join(self.experiment_checkpoint_dir, "model.pth")
            self._save_weights(model=model, path=checkpoint_path)
            self.log_info(f"Model saved.")
        elif self.save_checkpoint:
            checkpoint_path = os.path.join(self.experiment_checkpoint_dir, f"model_epoch_{epoch}.pth")
            self._save_weights(model=model, path=checkpoint_path)
            self.log_info(f"Model saved for epoch {epoch}.")

    def _save_weights_to_s3(self, s3_bucket_name: str):
        aws_handler = AWSHandler(s3_bucket_name=s3_bucket_name)
        for root, _, files in os.walk(self.experiment_checkpoint_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                s3_path = local_path.replace("\\", "/")
                
                aws_handler.upload_file(local_path=local_path, to_path=s3_path)
