import os
from datasets.dataset_utils import DatasetHandler
from training.trainer import Trainer
from utils.aws_handler import AWSHandler
from utils.config_parser import ConfigParser
from utils.seed import set_seed


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config = ConfigParser("configs/default.yaml")
    
    misc_config = config["misc"]
    set_seed(misc_config["seed"], misc_config["deterministic"])
    dataset_handler = DatasetHandler(config=config["dataset"])

    trainer = Trainer(config=config, dataset_handler=dataset_handler)
    trainer.fit()
