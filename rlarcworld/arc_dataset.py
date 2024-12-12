import os
import json

import numpy as np

import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ArcSampleManager(object):
    """
    Resize and Concatenate the grids in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, examples_stack_dim=10):
        assert isinstance(
            output_size, (tuple)
        ), "The output size should be tuple with the last two elements being the height, width."
        self.output_size = output_size
        self.examples_stack_dim = examples_stack_dim

    def pad_to_size(self, grid):
        height, width = [
            target - current
            for current, target in zip(grid.shape, self.output_size[-2:])
        ]
        assert height >= 0 and width >= 0, "The output size is smaller than the grid."
        logger.debug(
            "Shape original grid={}, Padded height={}, Padded width={}.".format(
                np.shape(grid), height, width
            )
        )
        return torch.nn.functional.pad(grid, (0, width, 0, height), value=-1)

    def concat_input_output(self, input_grid, output_grid):
        return torch.cat((input_grid.unsqueeze(0), output_grid.unsqueeze(0)), dim=0)

    def concat_examples(self, train_examples):
        return torch.cat(
            [
                self.concat_input_output(
                    self.pad_to_size(torch.tensor(example["input"])),
                    self.pad_to_size(torch.tensor(example["output"])),
                ).unsqueeze(0)
                for example in train_examples
            ],
            dim=0,
        )

    def __call__(self, sample):
        sample["train"] = self.concat_examples(sample["train"])
        n_examples = sample["train"].shape[0]
        if n_examples < self.examples_stack_dim:
            pad_size = np.zeros(len(sample["train"].shape) * 2, dtype=int)
            pad_size[-1] = self.examples_stack_dim - n_examples
            logger.debug(
                "Train Examples shape: {},Padding size: {}".format(
                    sample["train"].shape, pad_size
                )
            )
            sample["train"] = torch.nn.functional.pad(
                sample["train"], tuple(pad_size), value=-1
            )

        sample["test"]["input"] = self.pad_to_size(
            torch.tensor(sample["test"]["input"])
        )
        sample["test"]["output"] = self.pad_to_size(
            torch.tensor(sample["test"]["output"])
        )
        return sample


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
        self.file_ids = list(self.samples.keys())

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[self.file_ids[idx]]

        if isinstance(sample, str):
            test_index = int(sample.split("_")[-1]) if "_" in sample else 0
            sample = self.open_file(sample, test_index=test_index)

        if self.transform:
            sample = self.transform(sample)

        sample["examples"] = sample.pop("train")
        sample["task"] = sample.pop("test")
        return sample
