import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple, cast
import torch as T
from torch.utils.data import DataLoader

from evaluation.evaluate import Evaluator
from models.model_factory import BuildModel
from models.utils import ModelHandler
from training.early_stopping import EarlyStopping
from training.optimizers import setup_optimizer
from training.losses import setup_loss
from training.schedulers import setup_scheduler
from utils.config_parser import ConfigParser
from utils.logger import Logger
from utils.helpers import filter_kwargs, append_dict_to_dict
from typing import Mapping


class Trainer:
    """
    Trainer class for managing the training, validation, and evaluation of a deep learning model.

    Responsibilities:
        - Initialize model, optimizer, loss function, scheduler, and logging.
        - Handle training loop, validation, and metric logging.
        - Manage early stopping and learning rate scheduling.
        - Maintain directories for checkpoints and logs.
        - Provide a high-level fit() method to train the model for multiple epochs with optional validation.

    Attributes:
        config (ConfigParser): Configuration object containing all settings.
        project_name (str | None): Name of the project, from configuration.
        experiment_name (str | None): Name of the experiment, from configuration.
        device (str | None): Device to run the model on (e.g., "cuda" or "cpu").
        model (nn.Module): The deep learning model being trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (Callable): Loss function used for training.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        log_interval (int | None): Interval for logging training metrics.
        checkpoint_dir (str | None): Directory path for saving model checkpoints.
        log_dir (str | None): Base directory path for logs.
        log_subdirs (dict | None): Dictionary of subdirectories for logging (e.g., "train", "val").
        evaluator (Evaluator): Object for calculating metrics.
        logger (Logger): Object for logging metrics and information.
        early_stopper (EarlyStopping): Object for handling early stopping based on validation metrics.

    Methods:
        train_one_epoch(epoch, train_dataloader): Train the model for a single epoch.
        validate(dl): Run validation on a given dataloader and return metrics.
        _scheduler_step(metric): Step the scheduler, handling ReduceLROnPlateau if used.
        _early_stopping_step(metric): Update early stopper and return True if training should stop.
        _setup(): Initialize model, optimizer, loss, scheduler, and training settings.
        _setup_directories(): Initialize checkpoint and log directory paths.
        fit(train_dataloader, val_dataloader=None): 
            High-level method to train the model for multiple epochs. 
            Optionally evaluates on a validation dataloader, updates the scheduler, 
            handles early stopping, and logs metrics after each epoch.
    """
    def __init__(
        self,
        config: ConfigParser
    ):
        self.config = config
        self.project_name = self.config.get("project_name")
        self.experiment_name = self.config.get("experiment_name")
        self.device = self.config["misc"].get("device")
        self.s3_bucket_name: str | None = self.config.get("s3_bucket_name")

        self.evaluator = Evaluator(device=self.device, **filter_kwargs(Evaluator, self.config.get("evaluation")))
        self.logger = Logger(**filter_kwargs(Logger, self.config.config))

        self.scaler = T.GradScaler(self.device)
        self.model_handler = ModelHandler()

        self._setup()

    def _setup(self):
        """
        Set up the training environment, including directories, model, optimizer, loss function, 
        scheduler, and training parameters.

        Steps:
            1. Create necessary directories by calling `self._setup_directories()`.
            2. Initialize the model using configuration from `self.config["model"]` and move it to the device.
            3. Set up the optimizer using `setup_optimizer` and the training configuration.
            4. Set up the loss function using `setup_loss`.
            5. Set up the learning rate scheduler using `setup_scheduler`.
            5. Set up the early stopper.
            6. Load training parameters such as number of epochs and logging interval from the configuration.

        Attributes Set:
            - self.model (nn.Module): The neural network model.
            - self.optimizer (Optimizer): The optimizer for training.
            - self.loss_fn (Callable): The loss function used for training.
            - self.scheduler (Optional[LR Scheduler]): Learning rate scheduler, if defined.
            - self.early_stopper (EarlyStopping): Early stopping of training.
            - self.num_epochs (int): Number of training epochs.
            - self.log_interval (int | None): Interval (in batches) for logging metrics.

        Notes:
            - Assumes that `self.device` is already defined.
            - Configuration dictionaries are filtered for valid parameters using `filter_kwargs`.
        """
        self._setup_directories()

        # setup model
        model_config = self.config["model"]
        model = BuildModel(**filter_kwargs(BuildModel, model_config)).to(self.device)

        # load pretrained weights
        pretrained: dict[str, dict[str, Any]] = (model_config.get("pretrained") or {})

        self.model_handler.load_weights(
            model=model,
            checkpoint_dir = self.checkpoint_dir,
            project_name = self.config["project_name"],
            **filter_kwargs(self.model_handler.load_weights, pretrained)
        )

        self.model = model

        # setup training related objects
        training_config = self.config["training"]
        self.optimizer = setup_optimizer(self.model, training_config.get("optimizer"))

        self.loss_fn = setup_loss(training_config.get("loss"))

        self.scheduler = setup_scheduler(self.optimizer, training_config.get("scheduler"))
        
        self.early_stopper = EarlyStopping(**filter_kwargs(EarlyStopping, training_config.get("early_stopping")))  # should be assigned as a function

        self.num_epochs = training_config["num_epochs"]
        self.log_interval = training_config.get("log_interval")
        self.freeze_time = cast(int, pretrained.get("freeze_time", self.num_epochs))

    def _setup_directories(self):
        """
        Set up directory paths for checkpoints and logging based on the configuration.

        Attributes Set:
            - self.checkpoint_dir (str | None): Directory path where model checkpoints will be saved.
            - self.log_dir (str | None): Base directory path for logging.
            - self.log_subdirs (dict | None): Dictionary of subdirectories for different logging types (e.g., "train", "val").

        Notes:
            - The directory paths are retrieved from `self.config`.
            - This method does not create the directories on the filesystem; it only sets the paths.
        """
        self.checkpoint_dir: str = str(self.config.get("checkpoint_dir", ""))
        self.log_dir = str(self.config.get("log_dir", ""))
        self.log_subdirs = str(self.config.get("log_subdirs", ""))

    def train_one_epoch(self, epoch: int, train_dataloader: DataLoader[Tuple[T.Tensor, int]]) -> None:
        """
        Train the model for one epoch on the provided training DataLoader.

        Iterates over all batches in the DataLoader, performs forward and backward passes,
        updates model parameters, calculates metrics for each batch, and logs both batch-level
        and epoch-level metrics.

        Args:
            epoch (int): The current epoch number (used for logging).
            train_dataloader (DataLoader): PyTorch DataLoader containing the training dataset.

        Logs:
            - Batch-level metrics every `log_interval` batches.
            - Epoch-level metrics after completing all batches.
            - Epoch summary with average metrics.

        Notes:
            - The model is set to training mode (`self.model.train()`).
            - The optimizer and loss function defined in the trainer are used.
            - Metrics are calculated using `self.evaluator.calc_metrics`.
        """
        self.model.train()
        total_metrics: dict[str, list[float]] = {
            "loss": []
        }
        batch_idx = -1
        pbar = tqdm(train_dataloader, desc=f"Training epoch {epoch}", unit="batch")
        accumulate_steps = 1
        max_norm = 5.0
        self.optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass with autocast
            with T.autocast(device_type=self.device, dtype=T.float16):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss /= accumulate_steps

            # Backward with scaled loss
            self.scaler.scale(loss).backward()  # type: ignore

            if (batch_idx + 1) % accumulate_steps == 0:
                self.scaler.unscale_(self.optimizer)
                T.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_metrics["loss"].append(accumulate_steps * loss.item())
            metrics = self.evaluator.calc_metrics(outputs=outputs, labels=labels)
            total_metrics = append_dict_to_dict(total_metrics, metrics)

            stats: Mapping[str, Any] = {key: np.mean(value) for key, value in total_metrics.items()}
            pbar.set_postfix(ordered_dict=stats)  # type: ignore
            if self.log_interval and ( (batch_idx - self.log_interval + 1) % self.log_interval == 0 ):
                self.logger.log_metrics({key: value[-self.log_interval:] for key, value in total_metrics.items()}, "train")
        
        log_interval = self.log_interval or len(train_dataloader)
        last_idx = batch_idx % log_interval

        log_dict: dict[str, Any] = {}
        for key, value in total_metrics.items():
            log_dict[key] = value[-last_idx:]
            log_dict[f"epoch_{key}"] = np.mean(value)

        self.logger.log_metrics(log_dict, "train")
        self.logger.log_info("Epoch summary " + " ".join([f"{key}: {np.mean(value):.6f}" for key, value in total_metrics.items()]))
        
    def _scheduler_step(self, metric: float) -> None:
        """
        Perform a scheduler step to adjust the learning rate.

        If the scheduler is of type `ReduceLROnPlateau`, it requires a validation 
        metric (e.g., validation loss) to determine when to reduce the learning rate.
        Otherwise, a standard `.step()` call is made.

        Args:
            metric (float): The performance metric (typically validation loss) 
                used for learning rate scheduling when applicable.
        """
        if self.scheduler:
            if isinstance(self.scheduler, T.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics=metric)  # type: ignore
            else:
                self.scheduler.step()
            self.logger.log_info(f"Current lr: {self.scheduler.get_last_lr()[0]}")

    def _early_stopping_step(self, metric: float) -> bool:
        """
        Evaluate early stopping criteria based on the given metric.

        Calls the early stopper with the metric (typically validation loss). 
        If the early stopping condition is met, training should be stopped.

        Args:
            metric (float): The performance metric (e.g., validation loss) 
                used to evaluate whether to stop training early.

        Returns:
            bool: True if training should stop (early stopping triggered), 
            False otherwise.
        """
        if self.early_stopper:
            self.early_stopper(metric)
            self.logger.log_info(f"Last best score {self.early_stopper.best_score} - {self.early_stopper.counter} steps ago")
            return self.early_stopper.should_stop
        return False

    def fit(
        self,
        train_dataloader: DataLoader[Tuple[T.Tensor, int]],
        val_dataloader: DataLoader[Tuple[T.Tensor, int]] | None = None
    ):
        """
        High-level method to train the model for multiple epochs.

        Performs training on the provided train_dataloader and optionally evaluates
        on the val_dataloader at the end of each epoch. Handles learning rate
        scheduling, early stopping, and logging.

        Args:
            train_dataloader (DataLoader): Dataloader for training data.
            val_dataloader (DataLoader | None, optional): Dataloader for validation data.
                If provided, validation metrics are computed after each epoch. Defaults to None.

        Behavior:
            - Logs configuration at the start of training.
            - Iterates over `num_epochs` and calls `train_one_epoch` for training.
            - Optionally calls `validate` on the validation dataloader.
            - Updates the learning rate scheduler with validation loss if applicable.
            - Checks early stopping condition using validation loss.
            - Logs epoch metrics, including training and validation metrics.
            - Stops training early if early stopping is triggered.
        """
        self.logger.log_config(self.config)
        self.logger.log_info("Training started...")
        self.logger.log_info(self.model.__str__())
        for epoch in range(self.num_epochs):
            if epoch == self.freeze_time:
                self.model_handler.unfreeze_weights(self.model)

            self.train_one_epoch(epoch, train_dataloader)
            
            val_metrics = {}
            if val_dataloader:
                val_metrics = self.validate(val_dataloader)

            val_loss = val_metrics.get("loss", 0.0)
            self._scheduler_step(val_loss)
            
            if self._early_stopping_step(val_loss):
                self.logger.log_info(f"Early stopping triggered at epoch {epoch}")
                break

            self.logger.on_epoch_end_log(
                model=self.model,
                val_loss=val_metrics.get("loss"),
                epoch=epoch,
                end=(epoch == (self.num_epochs - 1))
            )

        self.logger.log_artifact(target="s3", s3_bucket_name=self.s3_bucket_name)

    def validate(self, dl: DataLoader[Tuple[T.Tensor, int]]) -> Dict[str, float]:
        """
        Evaluate the model on a validation dataset.

        Runs the model on all batches in the provided DataLoader, calculates 
        metrics using the evaluator, logs them, and returns the results.

        Args:
            dl (DataLoader): PyTorch DataLoader containing the validation dataset.

        Returns:
            Dict[str, float]: Dictionary of computed metrics (e.g., loss, accuracy).
        """
        metrics = self.evaluator.evaluate(self.model, dl, self.loss_fn)
        self.logger.log_metrics({f"epoch_{k}": v for k, v in metrics.items()}, "val")
        return metrics
