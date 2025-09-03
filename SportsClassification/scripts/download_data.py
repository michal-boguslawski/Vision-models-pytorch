from utils.config_parser import ConfigParser
from utils.data_utils import download_data
from utils.helpers import filter_kwargs


if __name__ == "__main__":
    config = ConfigParser(config_path="configs/dataset_config.yaml")
    
    download_data(**filter_kwargs(download_data, config.config))
