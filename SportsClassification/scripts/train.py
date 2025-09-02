from utils.config_parser import ConfigParser
from datasets.dataset_utils import create_dataloader
from training.trainer import Trainer
from utils.seed import set_seed


if __name__ == "__main__":
    config = ConfigParser("configs/default.yaml")
    
    misc_config = config["misc"]
    set_seed(misc_config["seed"], misc_config["deterministic"])
    
    train_dl = create_dataloader(
        config=config["dataset"],
        sub_dataset="train"
    )
    val_dl = create_dataloader(
        config=config["dataset"],
        sub_dataset="val"
    )

    trainer = Trainer(config=config)
    trainer.fit(train_dataloader=train_dl, val_dataloader=val_dl)
