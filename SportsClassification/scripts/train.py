from copy import deepcopy
import os
from datasets.dataset_utils import DatasetHandler
from training.trainer import Trainer
from utils.aws_handler import AWSHandler
from utils.config_parser import ConfigParser
from utils.logger import SingletonLogger
from utils.seed import set_seed


if __name__ == "__main__":
    logger_instance = SingletonLogger()
    aws_handler = AWSHandler()
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config = ConfigParser("configs/default.yaml")

    logger_instance.initialize(
        project_name=config.get("project_name"),
        experiment_name=config.get("experiment_name"),
        checkpoints_dir=config.get("checkpoints_dir"),
        logs_dir=config.get("logs_dir")
    )
    aws_handler.initialize()
    
    misc_config = config["misc"]
    set_seed(misc_config["seed"], misc_config["deterministic"])

    dataset_handler = DatasetHandler(config=config["dataset"])
    config_to_save = deepcopy(config.config)
    
    train_dl = dataset_handler.create_dataloader("train", use_augmentations=True, shuffle=True)
    val_dl = dataset_handler.create_dataloader("val", use_augmentations=False, shuffle=False)

    config_to_save["label_to_id"] = dataset_handler.label_to_id
    aws_handler.put_item_to_dynamodb(
        config_to_save
    )

    trainer = Trainer(config=config)
    trainer.fit(
        train_dataloader=train_dl,
        val_dataloader=val_dl
    )
