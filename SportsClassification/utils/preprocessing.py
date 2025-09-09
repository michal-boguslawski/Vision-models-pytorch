import numpy as np
import os
from PIL import Image, UnidentifiedImageError
import pandas as pd
from tqdm import tqdm
from typing import Any, cast
from datasets.dataset_utils import ImageDataset
from utils.filesystem import remove_dir_with_content, make_dirs
from utils.data_utils import compute_mean_std


default_annotation_dict = {
    "labels": "labels",
    "class_id": "class id",
    "train_test_split": "data set",
    "filepaths": "filepaths",
}


class DataPreprocessor:
    def __init__(
        self,
        root_dir: str = "data",
        raw_subdir: str = "raw",
        processed_subdir: str = "processed",
        annotations_subdir: str = "annotations",
        train_subdir: str = "train",
        val_subdir: str = "val",
        test_subdir: str = "test",
        annotations_file: str = "sports.csv",
        use_existing_train_test_split: bool = True,
        input_size: tuple[int, int] = (224, 224),
        annotations_config: dict[str, Any] = default_annotation_dict,
    ):
        self.root_dir = root_dir
        self.raw_subdir = raw_subdir
        self.processed_subdir = processed_subdir
        self.annotations_subdir = annotations_subdir
        self.train_subdir = train_subdir
        self.val_subdir = val_subdir
        self.test_subdir = test_subdir
        self.annotations_file = annotations_file
        self.use_existing_train_test_split = use_existing_train_test_split
        self.input_size = input_size
        self.annotations_config = annotations_config

        self._generate_paths()

    def _generate_paths(self):
        self.processed_dir = os.path.join(self.root_dir, self.processed_subdir)
        self.annotations_dir = os.path.join(self.root_dir, self.annotations_subdir)
        self.annotations_file_path = os.path.join(self.root_dir, self.raw_subdir, self.annotations_file)
        self.split_subdirs = [self.train_subdir, self.val_subdir, self.test_subdir]

    def _clean_folders(self):
        remove_dir_with_content(self.processed_dir)
        remove_dir_with_content(self.annotations_dir)

    def _make_dirs(self, labels: list[str] | np.ndarray):
        # make directories for each train, test, val datasets
        labels_list = labels.tolist() if isinstance(labels, np.ndarray) else labels
        make_dirs([self.processed_dir], self.split_subdirs, labels_list)
        # make directory for annotations
        make_dirs([self.annotations_dir])

    def _preprocessing_loop(self, df: pd.DataFrame, data_set: str):
        print("Start loop")
        temp_annotations_list: list[tuple[int, str, str]] = []
        
        for _, row in tqdm(df.iterrows(), desc=data_set, total=len(df)):
            image_path = os.path.join(self.root_dir, self.raw_subdir, str(row[self.annotations_config["filepaths"]]))
            label = str(row[self.annotations_config["labels"]])
            image_name = os.path.basename(image_path)
            processed_image_path = os.path.join(self.processed_dir, data_set, label, image_name)
            
            try:
                with Image.open(image_path) as im:
                    im = im.convert("RGB")
                    im = im.resize(size=self.input_size)
                    im.save(processed_image_path)
                class_id: int = row[str(self.annotations_config["class_id"])]
                annotation_row = (
                    class_id,
                    label,
                    os.path.join(data_set, label, image_name)
                )
                temp_annotations_list.append(annotation_row)
            except UnidentifiedImageError:
                print(f"Skipped non-image file {image_path}")
            except Exception as e:
                raise e

        temp_annotations_df = pd.DataFrame(
            temp_annotations_list,
            columns=["class_id", "label", "filepath"]
        )

        temp_annotations_df.to_csv(
            os.path.join(self.annotations_dir, data_set + ".csv"),
            index=False
        )

    def _run_preprocessing(self, df: pd.DataFrame) -> None:
        print("Use existing split")
        processed_data_set_name: str = ""
        split_column: str = str(self.annotations_config["train_test_split"])
        data_sets: np.ndarray = df[split_column].unique()
        for data_set in data_sets:
            data_set_df = cast(pd.DataFrame, df[df[split_column] == data_set])
            
            check_ = False
            for processed_data_set_name in self.split_subdirs:
                if processed_data_set_name in data_set:
                    check_ = True
                    break
            
            if check_:
                self._preprocessing_loop(df=data_set_df, data_set=processed_data_set_name)
        print("Data preprocessed")

    def _compute_mean_and_std(self):
        file_path: str = os.path.join(self.annotations_dir, "train.csv")
        train_df: pd.DataFrame = pd.read_csv( # type: ignore
            file_path,
            dtype={"class_id": int, "label": str, "filepath": str},
        )
        print("Compute mean and std")
        dt = ImageDataset(
            df=train_df,
            root_dir=self.root_dir,
            processed_subdir=self.processed_subdir,
            transform=None,
        )
        mean, std = compute_mean_std(dt)
        print(f"Mean: {mean}, Std: {std}")

    def run(self) -> None:
        print("Start preprocessing")
        annotations_df: pd.DataFrame = pd.read_csv(self.annotations_file_path)  # type: ignore
        labels = annotations_df[str(self.annotations_config["labels"])].unique()
        self._clean_folders()
        self._make_dirs(labels=labels)
        
        if self.use_existing_train_test_split:
            self._run_preprocessing(annotations_df)

        self._compute_mean_and_std()
