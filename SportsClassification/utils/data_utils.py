import kagglehub
import os
from utils.helpers import remove_dir_with_content, check_data_exists


def _download_data_kaggle(
    kaggle_dataset_name: str,
    root_dir: str = "data",
    raw_subdir: str = "raw",
    *args,
    **kwargs
):
    path = os.path.join(root_dir, raw_subdir)
    dataset_path = kagglehub.dataset_download(kaggle_dataset_name, force_download=True)

    remove_dir_with_content(path)

    print(f"Path to dataset files: {dataset_path}")
    
    os.rename(dataset_path, path)
    print(f"Folder moved from {dataset_path} to {path}")


def download_data(
    source: str,
    root_dir: str = "data",
    raw_subdir: str = "raw",
    force_download: bool = False,
    *args,
    **kwargs
):
    if source == "kaggle":
        if (
            not check_data_exists(root_dir=root_dir, subdir=raw_subdir)
            or ( force_download )
        ):
            _download_data_kaggle(
                root_dir=root_dir,
                raw_subdir=raw_subdir,
                *args,
                **kwargs
            )
        else:
            print("Dataset already exists in location {root_dir}")
