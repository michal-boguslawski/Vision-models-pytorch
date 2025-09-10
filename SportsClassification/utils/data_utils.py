import kagglehub
import os
import shutil
from typing import Tuple
import torch as T
from torch.utils.data import Dataset, DataLoader
from utils.filesystem import remove_dir_with_content, check_data_exists
from utils.logger import SingletonLogger


logger_instance = SingletonLogger()


def _download_data_kaggle(
    kaggle_dataset_name: str,
    root_dir: str = "data",
    raw_subdir: str = "raw",
):
    logger_instance.logger.info("Downloading dataset from kaggle...")
    path = os.path.join(root_dir, raw_subdir)
    dataset_path = kagglehub.dataset_download(kaggle_dataset_name, force_download=True)

    remove_dir_with_content(path)

    logger_instance.logger.info(f"Dataset downloaded. Path to dataset files: {dataset_path}")

    shutil.move(dataset_path, path)
    logger_instance.logger.info(f"Folder moved from {dataset_path} to {path}")


def download_data(
    source: str,
    kaggle_dataset_name: str,
    root_dir: str = "data",
    raw_subdir: str = "raw",
    force_download: bool = False,
):
    if source == "kaggle":
        if (
            not check_data_exists(root_dir=root_dir, subdir=raw_subdir)
            or ( force_download )
        ):
            _download_data_kaggle(
                kaggle_dataset_name=kaggle_dataset_name,
                root_dir=root_dir,
                raw_subdir=raw_subdir,
            )
        else:
            logger_instance.logger.info("Dataset already exists in location {root_dir}")

def compute_mean_std(dt: Dataset[Tuple[T.Tensor, int]]) -> Tuple[float, float]:
    loader = DataLoader(dt, batch_size=64, shuffle=False, num_workers=0)
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for images, _ in loader:
        images = images.float() / 255.0
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std
