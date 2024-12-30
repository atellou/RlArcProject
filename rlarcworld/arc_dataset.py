import os
import json

import numpy as np

from tensordict import TensorDict
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ArcSampleTransformer(object):
    """
    Resize and Concatenate the grids in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(
        self,
        output_size: tuple[int],
        examples_stack_dim: int = 10,
        padding_constant_value: int = -1,
        zero_based_correction: int = 1,
    ):
        """
        Transformations for ARC samples dataset, specific to dimension transformations.
        Args:
            output_size (tuple): The desired output size for the grids.
            examples_stack_dim (int, optional): The dimension of the stack of examples. Defaults to 10.
            padding_constant_value (int, optional): The constant value to pad the grids. Defaults to -1.
            zero_based_correction (int, optional): The correction to apply to make the grids minimum value 0. Defaults to 1 (grid+1).
        """
        assert isinstance(
            output_size, (tuple)
        ), "The output size should be tuple with the last two elements being the height, width."
        self.zero_based_correction = zero_based_correction
        self.constant_value = padding_constant_value
        self.output_size = output_size
        self.examples_stack_dim = examples_stack_dim

    def pad_to_size(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Pad the grid to the given size.
        Args:
            grid (torch.Tensor): The grid to be padded.
        Returns:
            torch.Tensor: The padded grid.
        """
        assert len(grid.shape) == 2, "The grid should be a 2D array."
        height, width = [
            target - state for state, target in zip(grid.shape, self.output_size[-2:])
        ]
        assert height >= 0 and width >= 0, "The output size is smaller than the grid."
        logger.debug(
            "Shape original grid={}, Padded height={}, Padded width={}.".format(
                np.shape(grid), height, width
            )
        )
        return torch.nn.functional.pad(
            grid, (0, width, 0, height), value=self.constant_value
        )

    def concat_unsqueezed(
        self, grid_one: torch.Tensor, grid_two: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate two grids in a third dimension.
        Args:
            grid_one (torch.Tensor): The first grid.
            output_grid (torch.Tensor): The second grid.
        Returns:
            torch.Tensor: The concatenated grid.
        """
        assert grid_one.shape == grid_two.shape, (
            "The input and output grids should have the same shape." ""
        )
        return torch.cat((grid_one.unsqueeze(0), grid_two.unsqueeze(0)), dim=0)

    def concat_examples(self, train_examples: list[dict]) -> torch.Tensor:
        """
        Stack the train examples into a single tensor.
        Args:
            train_examples (list): A list of dictionaries containing the input and output grids.
        Returns:
            torch.Tensor: The concatenated tensor of train examples of shape (Examples, Stages, Grid Height, Grid Width).
        """
        logger.debug("Concat examples...")
        return torch.cat(
            [
                self.concat_unsqueezed(
                    self.pad_to_size(torch.as_tensor(example["input"])),
                    self.pad_to_size(torch.as_tensor(example["output"])),
                ).unsqueeze(0)
                for example in train_examples
            ],
            dim=0,
        )

    def __call__(self, sample: dict[list]) -> torch.Tensor:
        """
        Apply the transformations to the sample.
        Args:
            sample (dict): A dictionary containing the input and output grids.
        Returns:
            dict: The transformed sample.
        """
        logger.debug("Perform {} transformations...".format(self.__class__.__name__))
        sample["examples"] = self.concat_examples(sample["examples"])
        n_examples = sample["examples"].shape[0]
        if n_examples < self.examples_stack_dim:
            pad_size = torch.zeros(len(sample["examples"].shape) * 2, dtype=int)
            pad_size[-1] = self.examples_stack_dim - n_examples
            logger.debug(
                "Train Examples shape: {},Padding size: {}".format(
                    sample["examples"].shape, pad_size
                )
            )
            sample["examples"] = (
                torch.nn.functional.pad(
                    sample["examples"], tuple(pad_size), value=self.constant_value
                )
                + self.zero_based_correction
            )

        sample["task"]["input"] = (
            self.pad_to_size(torch.as_tensor(sample["task"]["input"]))
            + self.zero_based_correction
        )
        sample["task"]["output"] = (
            self.pad_to_size(torch.as_tensor(sample["task"]["output"]))
            + self.zero_based_correction
        )
        return sample


class ArcDataset(Dataset):
    """ARC dataset."""

    def __init__(
        self,
        arc_dataset_dir: str,
        keep_in_memory: bool = False,
        transform: callable = None,
    ):
        """
        Creates a Torch Dataset from the ARC dataset.
        Args:
            arc_dataset_dir (string): Directory with all the dataset samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.arc_dir = arc_dataset_dir
        self.keep_in_memory = keep_in_memory
        self.transform = transform
        self.transformed = set()
        self.load_dataset()

    def open_file(self, file_key: str, test_index: int = None) -> dict:
        """
        Open file and return train and test samples.
        """
        logger.debug("Open file: {}".format(file_key))
        with open(file_key, "r") as file:
            sample = json.load(file)
        if test_index is not None:
            sample["test"] = {
                "input": torch.as_tensor(sample["test"][test_index]["input"]),
                "output": torch.as_tensor(sample["test"][test_index]["output"]),
            }
        else:
            sample["test"] = [
                {
                    "input": torch.as_tensor(t["input"]),
                    "output": torch.as_tensor(t["output"]),
                }
                for t in sample["test"]
            ]
        sample["train"] = [
            {
                "input": torch.as_tensor(t["input"]),
                "output": torch.as_tensor(t["output"]),
            }
            for t in sample["train"]
        ]
        return {"train": sample["train"], "test": sample["test"]}

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
        logger.debug("Load Dataset")
        file_indexes = os.listdir(self.arc_dir)
        self.samples = {}
        for i, file in enumerate(file_indexes):
            sample = self.open_file(os.path.join(self.arc_dir, file))
            for i, s_test in enumerate(sample["test"]):
                key = (file.split(".")[0], i)
                if self.keep_in_memory:
                    self.samples[key] = {"examples": sample["train"], "task": s_test}
                else:
                    self.samples[key] = os.path.join(self.arc_dir, file)
        self.file_ids = list(self.samples.keys())

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns extracted and transformed sample from the dataset.
        Args:
            idx (int): Index of the sample to be returned.
        Returns:
            dict: The transformed sample.
        """
        logger.debug("Sample index: {}".format(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_id = self.file_ids[idx]
        logger.debug("File id: {}".format(file_id))
        sample = self.samples[file_id]

        if isinstance(sample, str):
            test_index = file_id[-1]
            sample = self.open_file(sample, test_index=test_index)

        sample.setdefault("examples", sample.pop("train", None))
        sample.setdefault("task", sample.pop("test", None))
        if self.transform:
            logger.debug("Transform data")
            if self.keep_in_memory and file_id not in self.transformed:
                sample = self.transform(sample)
                self.transformed.add(file_id)
            elif not self.keep_in_memory:
                sample = self.transform(sample)

        return sample
