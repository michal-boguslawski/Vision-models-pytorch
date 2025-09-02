from utils.config_parser import ConfigParser
from utils.data_utils import download_data


if __name__ == "__main__":
    config = ConfigParser(config_path="configs/dataset_config.yaml")
    
    download_data(
        source=config["source"],
        root_dir=config.get("root_dir"),
        raw_subdir=config.get("raw_subdir"),
        force_download=config.get("force_download", False)
    )
