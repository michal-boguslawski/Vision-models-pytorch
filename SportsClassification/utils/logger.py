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


class Logger:
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        checkpoint_dir: str = "checkpoints",
        save_best_only: bool = True,
        save_checkpoint: bool = True,
        log_dir: str = "logs",
        log_subdirs: dict | None = None,
        *args,
        **kwargs,
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.save_checkpoint = save_checkpoint
        self.log_dir = log_dir
        self.log_subdirs = log_subdirs

        self.best_val_loss = None
        self.experiment_log_dir = None
        self.experiment_checkpoint_dir = None
        
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
        make_dirs([self.experiment_log_dir], self.log_subdirs)
        
        # create directories for checkpoints
        if self.save_checkpoint:
            self.experiment_checkpoint_dir = os.path.join(self.checkpoint_dir, self.project_name, self.experiment_name)
            remove_dir_with_content(self.experiment_checkpoint_dir)
            make_dirs([self.experiment_checkpoint_dir])

    def log_config(self, config: ConfigParser):
        config_path = os.path.join(self.experiment_log_dir, "config.yaml")
        config.save(config_path)

    def log_metrics(self, new_metrics: dict, sub_dataset: str = "train"):
        metrics_path = os.path.join(self.experiment_log_dir, self.log_subdirs[sub_dataset], "metrics.json")
        metrics = load_dict_and_append(metrics_path, new_metrics)
        
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        self.log_info(f"New metrics saved to {metrics_path}")
        self.log_info(" ".join([f"{key}: {np.mean(value):.6f}" for key, value in new_metrics.items()]))

    def log_info(self, message: str):
        logging.info(message)

    def log_artifact(self):
        pass

    def _save_weights(self, model: nn.Module, path: str):
        T.save(model.state_dict(), path)        

    def on_epoch_end_log(self, model: nn.Module, val_loss: float | None = None, epoch: int | None = None, end: bool = False):
        if self.save_best_only and self.save_checkpoint:
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, "model_best.pth")
                self._save_weights(model=model, path=checkpoint_path)
                self.log_info(f"Model saved. New best value {val_loss:.6f}")
        elif ( not self.save_checkpoint ) and end:
            checkpoint_path = os.path.join(self.checkpoint_dir, "model.pth")
            self._save_weights(model=model, path=checkpoint_path)
            self.log_info(f"Model saved.")
        elif self.save_checkpoint:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
            self._save_weights(model=model, path=checkpoint_path)
            self.log_info(f"Model saved for epoch {epoch}.")
