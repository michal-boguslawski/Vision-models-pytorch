import os
import pandas as pd
from PIL import Image
from typing import Tuple, Any, cast, Optional
import torch as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from utils.config_parser import ConfigParser
from utils.seed import seed_worker
from datasets.augmentations import AUGMENTATIONS_DICT


def create_target_encoders(df: pd.DataFrame) -> Tuple[dict[int, str], dict[str, int]]:
    unique_df = df[["class_id", "label"]].drop_duplicates()
    id_to_label = dict(zip(unique_df["class_id"], unique_df["label"]))
    label_to_id = dict(zip(unique_df["label"], unique_df["class_id"]))
    return id_to_label, label_to_id


class ImageDataset(Dataset[Tuple[T.Tensor, int]]):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str = "data",
        processed_subdir: str = "processed",
        transform: v2.Compose | None = None,
    ):
        super().__init__()
        self.annotations_df = df.copy()
        
        self.root_dir = root_dir
        self.processed_subdir = processed_subdir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations_df)

    def __getitem__(self, idx: int) -> Tuple[T.Tensor, int]:
        row = self.annotations_df.iloc[idx]
        img_path = os.path.join(self.root_dir, self.processed_subdir, row["filepath"])
        image = decode_image(img_path)
        label = row["class_id"]
        if self.transform:
            image = self.transform(image)
        return image, label


class DatasetHandler:
    def __init__(
        self,
        config: ConfigParser | dict[str, Any],
    ):
        self.config = config
        
        self._load_df()
        self._create_target_encoders(self.df_dict["train"])
        self.transforms = None

    def _load_df(self):
        root_dir = str(self.config["root_dir"])
        annotations_subdir = str(self.config.get("annotations_subdir", "annotations"))

        df_dict: dict[str, pd.DataFrame] = {}
        annotations_file_path_dict: dict[str, str] = {}
        for sub_dataset in ["train", "val", "test"]:
            annotations_file_path_dict[sub_dataset] = os.path.join(
                root_dir, annotations_subdir, sub_dataset + ".csv"
            )
            df_dict[sub_dataset] = pd.read_csv(annotations_file_path_dict[sub_dataset])  # type: ignore

        self.annotation_file_path = annotations_file_path_dict
        self.df_dict = df_dict

    def _create_target_encoders(self, df: pd.DataFrame):
        unique_df = df[["class_id", "label"]].drop_duplicates()
        self.id_to_label = dict(zip(unique_df["class_id"], unique_df["label"]))
        self.label_to_id = dict(zip(unique_df["label"], unique_df["class_id"]))

    def _create_transforms(self, use_augmentations: bool = True, size: Optional[Tuple[int, int]] = None) -> v2.Compose:
        augmentations = self.config.get("augmentations", {}) if use_augmentations else None
        normalize_mean = self.config.get("normalize_mean", None)
        normalize_std = self.config.get("normalize_std", None)
        
        transforms_list: list[v2.Transform] = []
        random_apply_list: list[v2.Transform] = []

        transforms_list.append(v2.ToImage())
        if size:
            transforms_list.append(v2.Resize(size=size))

        if augmentations:
            for key, value in augmentations.items():
                if key in AUGMENTATIONS_DICT and value:
                    apply_type, transform = AUGMENTATIONS_DICT[key]
                    if apply_type == "compose":
                        transforms_list.append(transform)
                    elif apply_type == "random":
                        random_apply_list.append(transform)

        if random_apply_list:
            transforms_list.append(v2.RandomApply(random_apply_list))

        transforms_list.append(v2.ToDtype(T.float32, scale=True))

        if normalize_mean and normalize_std:
            transforms_list.append(
                v2.Normalize(
                    mean=normalize_mean,
                    std=normalize_std
                )
            )

        transforms = v2.Compose(transforms_list)
        return transforms

    def preprocess_single_image(self, image: Image.Image | T.Tensor, size: Optional[Tuple[int, int]] = None) -> T.Tensor:
        if not self.transforms:
            self.transforms = self._create_transforms(use_augmentations=False, size=size)

        image_tensor = self.transforms(image)
        return image_tensor

    def _create_image_dataset(
        self,
        sub_dataset: str = "train",
        use_augmentations: bool = True,
    ):
        transform = self._create_transforms(use_augmentations)

        dt = ImageDataset(
            df=self.df_dict[sub_dataset],
            root_dir=self.config["root_dir"],
            processed_subdir=self.config["processed_subdir"],
            transform=transform,
        )
        return dt

    def create_dataloader(
        self,
        sub_dataset: str = "train",
        use_augmentations: bool = True,
        shuffle: bool = True,
    ) -> DataLoader[Tuple[T.Tensor, int]]:
        num_workers: int = cast(int, self.config.get("num_workers", 1))

        dl = DataLoader(
            dataset=self._create_image_dataset(sub_dataset, use_augmentations),
            batch_size=self.config["batch_size"],
            shuffle=self.config["shuffle"] and shuffle,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        return dl
