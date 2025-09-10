from utils.config_parser import ConfigParser
from utils.preprocessing import DataPreprocessor
from utils.helpers import filter_kwargs
from utils.logger import SingletonLogger


if __name__ == "__main__":
    config = ConfigParser(config_path="configs/default.yaml")
    logger_instance = SingletonLogger()
    logger_instance.initialize(
        project_name=config.get("project_name"),
        experiment_name=config.get("experiment_name"),
        checkpoints_dir=config.get("checkpoints_dir"),
        logs_dir=config.get("logs_dir")
    )
    
    data_preprocessor = DataPreprocessor(**filter_kwargs(DataPreprocessor, config["dataset"]))
    data_preprocessor.run()
