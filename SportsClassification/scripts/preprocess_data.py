from utils.config_parser import ConfigParser
from utils.preprocessing import DataPreprocessor
from utils.helpers import filter_kwargs


if __name__ == "__main__":
    config = ConfigParser(config_path="configs/dataset_config.yaml")
    
    data_preprocessor = DataPreprocessor(**filter_kwargs(DataPreprocessor, config.config))
    data_preprocessor.run()
