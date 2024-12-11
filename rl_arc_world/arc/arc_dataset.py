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
        keep_in_memory: bool = False,
        transform=None,
    ):
        """
        Args:
            arc_dataset_dir (string): Directory with all the dataset samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.arc_dir = arc_dataset_dir
        self.keep_in_memory = keep_in_memory
        self.transform = transform
        self.load_dataset()

    def open_file(self, file_key: str, test_index: int = None):
        """
        Open file and return train and test samples.
        """
        with open(file_key, "r") as file:
            sample = json.load(file)
        if test_index is not None:
            sample["test"] = sample["test"][test_index]
        return sample

    def load_dataset(self):
        """
        Creates a dictionary of file location or samples from the dataset.
        For the samples with more than one test pair,
        the key for the dictionary is additioned and undersocere plus test pair index.

        Two behaviours:
        If "self.keep_in_memory" is False (default) create dictionary of file_id[key]:file_dir[value] pairs.
        If "self.keep_in_memory" is True create dictionary of file_id[key]:[value] pairs.

        Returns:
            None
        """
        # Get training file indexes
        file_indexes = os.listdir(self.arc_dir)
        self.samples = {}
        for i, file in enumerate(file_indexes):
            sample = self.open_file(os.path.join(self.arc_dir, file))
            for i, s_test in enumerate(sample["test"]):
                key = "{file_id}{underscore}{sample_number}".format(
                    file_id=file.split(".")[0],
                    underscore="_" * int(i > 0),
                    sample_number=i * int(i > 0),
                )
                if self.keep_in_memory:
                    self.samples[key] = {"train": sample["train"], "test": s_test}
                else:
                    self.samples[key] = os.path.join(self.arc_dir, file)

    def __len__(self):
        return len(self.file_indexes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]

        if isinstance(sample, str):
            test_index = int(sample.split("_")[-1]) if "_" in sample else 0
            sample = self.open_file(sample, test_index=test_index)

        if self.transform:
            sample = self.transform(sample)

        return sample
