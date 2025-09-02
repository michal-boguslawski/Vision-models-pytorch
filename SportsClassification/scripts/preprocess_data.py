from utils.config_parser import ConfigParser
from utils.preprocessing import DataPreprocessor


if __name__ == "__main__":
    config = ConfigParser(config_path="configs/dataset_config.yaml")
    
    data_preprocessor = DataPreprocessor(
        root_dir=config["root_dir"],
        raw_subdir=config["raw_subdir"],
        processed_subdir=config["processed_subdir"],
        annotations_subdir=config["annotations_subdir"],
        train_subdir=config["train_subdir"],
        val_subdir=config["val_subdir"],
        test_subdir=config["test_subdir"],
        annotations_file=config.get("annotations_file"),
        use_existing_train_test_split=config.get("use_existing_train_test_split"),
        input_size=config.get("input_size"),
    )
    data_preprocessor.run()
