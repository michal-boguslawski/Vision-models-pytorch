import os
from utils.config_parser import ConfigParser
from datasets.dataset_utils import create_dataloader
from training.trainer import Trainer
from utils.seed import set_seed


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config = ConfigParser("configs/default.yaml")
    
    misc_config = config["misc"]
    set_seed(misc_config["seed"], misc_config["deterministic"])
    
    train_dl = create_dataloader(
        config=config["dataset"],
        sub_dataset="train",
        use_augmentations=True,
        shuffle=True
    )
    val_dl = create_dataloader(
        config=config["dataset"],
        sub_dataset="val",
        use_augmentations=True,
        shuffle=False
    )

    trainer = Trainer(config=config)
    trainer.fit(train_dataloader=train_dl, val_dataloader=val_dl)
