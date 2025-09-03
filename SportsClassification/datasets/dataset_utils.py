import os
import pandas as pd
import torch as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from utils.config_parser import ConfigParser
from utils.seed import seed_worker
from datasets.augmentations import AUGMENTATIONS_DICT


def create_transforms(
    augmentations: dict | None = None,
    normalize_mean: list | None = None,
    normalize_std: list | None = None
) -> v2.Compose:
    transforms_list = []
    if augmentations:
        for key, value in augmentations.items():
            if key in AUGMENTATIONS_DICT and value:
                transforms_list.append(AUGMENTATIONS_DICT[key])

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


def create_target_encoders(df: pd.DataFrame) -> tuple[dict, dict]:
    unique_df = df[["class_id", "label"]].drop_duplicates()
    id_to_label = dict(zip(unique_df["class_id"], unique_df["label"]))
    label_to_id = dict(zip(unique_df["label"], unique_df["class_id"]))
    return id_to_label, label_to_id


class ImageDataset(Dataset):
    def __init__(
        self,
        annotations_file_path: str,
        root_dir: str = "data",
        processed_subdir: str = "processed",
        transform=None,
    ):
        super().__init__()
        self.annotations_df = pd.read_csv(annotations_file_path)
        self.id_to_label, self.label_to_id = create_target_encoders(self.annotations_df)
        
        self.root_dir = root_dir
        self.processed_subdir = processed_subdir
        self.transform = transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx: int):
        row = self.annotations_df.iloc[idx]
        img_path = os.path.join(self.root_dir, self.processed_subdir, row["filepath"])
        image = decode_image(img_path)
        label = row["class_id"]
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloader(
    config: ConfigParser | dict,
    sub_dataset: str = "train",
):
    transform = create_transforms(
        augmentations=config.get("augmentations") if sub_dataset == "train" else None,
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std"),
    )
    root_dir = str(config["root_dir"])
    annotations_subdir = str(config.get("annotations_subdir", "annotations"))
    processed_subdir = str(config.get("processed_subdir", "processed"))
    
    annotations_file_path = os.path.join(root_dir, annotations_subdir, sub_dataset + ".csv")

    dt = ImageDataset(
        annotations_file_path=annotations_file_path,
        root_dir=root_dir,
        processed_subdir=processed_subdir,
        transform=transform,
    )
    dl = DataLoader(
        dataset=dt,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"] if sub_dataset == "train" else False,
        num_workers=config.get("num_workers", 1),
        worker_init_fn=seed_worker
    )
    return dl
