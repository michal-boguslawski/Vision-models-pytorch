import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from evaluation.evaluate import Evaluator
from models.model_factory import build_model
from training.optimizers import setup_optimizer
from training.losses import setup_loss
from training.schedulers import setup_scheduler
from utils.config_parser import ConfigParser
from utils.logger import Logger
from utils.helpers import filter_kwargs, append_dict_to_dict


class Trainer:
    def __init__(
        self,
        config: ConfigParser | None = None
    ):
        self.config = config
        self.project_name = self.config.get("project_name")
        self.experiment_name = self.config.get("experiment_name")
        self.device = self.config["misc"].get("device")

        self._setup()
        self.evaluator = Evaluator(device=self.device, **filter_kwargs(Evaluator, self.config.get("evaluation")))
        self.logger = Logger(**filter_kwargs(Logger, self.config.config))

    def _setup(self):
        self._setup_directories()

        # setup model
        model_config = self.config["model"]
        self.model = build_model(device=self.device, **filter_kwargs(build_model, model_config))

        # setup training related objects
        training_config = self.config["training"]
        self.optimizer = setup_optimizer(self.model, training_config.get("optimizer"))

        self.loss_fn = setup_loss(training_config.get("loss"))

        self.scheduler = setup_scheduler(self.optimizer, training_config.get("scheduler"))

        self.num_epochs = training_config["num_epochs"]
        self.log_interval = training_config.get("log_interval")

    def _setup_directories(self):
        self.checkpoint_dir = self.config.get("checkpoint_dir")
        self.log_dir = self.config.get("log_dir")
        self.log_subdirs = self.config.get("log_subdirs")

    def train_one_epoch(self, epoch: int, train_dataloader: DataLoader):
        self.model.train()
        total_metrics = {
            "loss": []
        }
        pbar = tqdm(train_dataloader, desc=f"Training epoch {epoch}", unit="batch")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_metrics["loss"].append(loss.item())
            metrics = self.evaluator.calc_metrics(outputs=outputs, labels=labels)
            total_metrics = append_dict_to_dict(total_metrics, metrics)

            pbar.set_postfix({key: np.mean(value) for key, value in total_metrics.items()})
            if self.log_interval and ( (batch_idx - self.log_interval + 1) % self.log_interval == 0 ):
                self.logger.log_metrics({key: value[-self.log_interval:] for key, value in total_metrics.items()}, "train")
        
        log_interval = self.log_interval or len(train_dataloader)
        last_idx = batch_idx % log_interval

        log_dict = {}
        for key, value in total_metrics.items():
            log_dict[key] = value[-last_idx:]
            log_dict[f"epoch_{key}"] = np.mean(value)

        self.logger.log_metrics(log_dict, "train")
        self.logger.log_info("Epoch summary " + " ".join([f"{key}: {np.mean(value):.6f}" for key, value in total_metrics.items()]))

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None
    ):
        self.logger.log_config(self.config)
        self.logger.log_info("Training started...")
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch, train_dataloader)
            if self.scheduler:
                self.scheduler.step()
                self.logger.log_info(f"Current lr: {self.scheduler.get_last_lr()}")
            
            val_metrics = {}
            if val_dataloader:
                val_metrics = self.validate(val_dataloader)
            self.logger.on_epoch_end_log(
                model=self.model,
                val_loss=val_metrics.get("loss"),
                epoch=epoch,
                end=(epoch == (self.num_epochs - 1))
            )

    def validate(self, dl: DataLoader):
        metrics = self.evaluator.evaluate(self.model, dl, self.loss_fn)
        self.logger.log_metrics(metrics, "val")
        return metrics
