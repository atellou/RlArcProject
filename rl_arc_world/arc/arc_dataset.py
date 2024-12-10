import os
import json

import numpy as np

import torch
from torch.utils.data import Dataset


class ArcDataset(Dataset):
    """ARC dataset."""

    def __init__(
        self,
        arc_dataset_dir: str,
        training_subdir: str = "training",
        testing_subdir: str = "evaluation",
        transform=None,
    ):
        """
        Args:
            arc_dataset_dir (string): Directory with all the dataset samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.arc_dir = arc_dataset_dir
        self.training_subdir = training_subdir
        self.testing_subdir = testing_subdir
        self.transform = transform
        self.load_dataset_indexes()

    def load_dataset_indexes(self):
        """
        Create list of indexes for training and testing.
        """
        # Get training file indexes
        self.training_file_indexes = os.listdir(
            os.path.join(self.arc_dir, self.training_subdir)
        )
        # Get testing file indexes
        self.testing_file_indexes = os.listdir(
            os.path.join(self.arc_dir, self.testing_subdir)
        )

    def __len__(self):
        return len(self.training_file_indexes) + len(self.testing_file_indexes)

    def open_file(self, file_key: str):
        """
        Open file and return train and test sample with grids as numpy arrays.
        """
        with open(file_key, "r") as file:
            sample_json = json.load(file)
            sample = {}
            sample["train"] = [
                (np.array(p_grids["input"]), np.array(p_grids["output"]))
                for p_grids in sample_json["train"]
            ]
            sample["test"] = [
                (np.array(p_grids["input"]), np.array(p_grids["output"]))
                for p_grids in sample_json["test"]
            ]
        return sample

    def __getitem__(self, idx, train: bool = True):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if train:
            self.file_key = os.path.join(
                self.arc_dir, self.training_subdir, self.training_file_indexes[idx]
            )
        else:
            self.file_key = os.path.join(
                self.arc_dir, self.testing_subdir, self.testing_file_indexes[idx]
            )

        sample = self.open_file(self.file_key)

        if self.transform:
            sample = self.transform(sample)

        return sample
