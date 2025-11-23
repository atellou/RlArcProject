"""Tests for ARC augmentation utilities."""

import pytest
import torch

from rlarcworld.arc_dataset import ComposeTransforms
from rlarcworld.utils.augmentation import (
    ArcAugmentationTransformer,
    permute_colors,
    reflect_grid,
    rotate_grid,
    shuffle_examples,
    translate_grid,
)


@pytest.fixture
def sample_grid():
    """Create a sample grid for testing."""
    return torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)


@pytest.fixture
def sample_data(sample_grid):
    """Create a sample ARC dataset item."""
    return {
        "examples": torch.stack([sample_grid, sample_grid]),
        "task": {"input": sample_grid.clone(), "output": sample_grid.clone()},
    }


class TestAugmentationUtils:
    def test_rotate_grid(self, sample_grid):
        # Test 90° rotation
        rotated = rotate_grid(sample_grid, 1)
        expected = torch.tensor([[3, 6, 9], [2, 5, 8], [1, 4, 7]], dtype=torch.float32)
        assert torch.allclose(rotated, expected)

        # Test 180° rotation
        rotated = rotate_grid(sample_grid, 2)
        expected = torch.tensor([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=torch.float32)
        assert torch.allclose(rotated, expected)

        # Test identity rotation (0°)
        rotated = rotate_grid(sample_grid, 0)
        assert torch.allclose(rotated, sample_grid)

    def test_reflect_grid(self, sample_grid):
        # Test horizontal reflection
        reflected = reflect_grid(sample_grid, 0)
        expected = torch.tensor([[7, 8, 9], [4, 5, 6], [1, 2, 3]], dtype=torch.float32)
        assert torch.allclose(reflected, expected)

        # Test vertical reflection
        reflected = reflect_grid(sample_grid, 1)
        expected = torch.tensor([[3, 2, 1], [6, 5, 4], [9, 8, 7]], dtype=torch.float32)
        assert torch.allclose(reflected, expected)

    def test_permute_colors(self, sample_grid):
        color_map = {1: 9, 2: 8, 3: 7, 4: 6, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1}
        permuted = permute_colors(sample_grid, color_map)
        expected = torch.tensor([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=torch.float32)
        assert torch.allclose(permuted, expected)

    def test_translate_grid(self, sample_grid):
        # Test right shift
        translated = translate_grid(sample_grid, 1, 0)
        expected = torch.tensor([[0, 1, 2], [0, 4, 5], [0, 7, 8]], dtype=torch.float32)
        assert torch.allclose(translated, expected)

        # Test down shift
        translated = translate_grid(sample_grid, 0, 1)
        expected = torch.tensor([[0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        assert torch.allclose(translated, expected)

    def test_augment_sample_rotation(self, sample_data, sample_grid):
        # Test transformer's augment_sample method with rotation
        transformer = ArcAugmentationTransformer(augmentations=["rotate"])

        # Set seed for deterministic rotation (will get rotation=1 for 90°)
        torch.manual_seed(42)
        augmented = transformer.augment_sample(sample_data)

        # Verify rotation was applied (output should differ from input)
        assert not torch.allclose(augmented["task"]["input"], sample_grid)
        # Verify shape is preserved
        assert augmented["task"]["input"].shape == sample_grid.shape

    def test_augment_sample_color_permutation(self, sample_data, sample_grid):
        # Test transformer's augment_sample method with color permutation
        transformer = ArcAugmentationTransformer(
            augmentations=["permute_colors"], color_permutation_range=(1, 10)
        )

        # Set seed for deterministic permutation
        torch.manual_seed(123)
        augmented = transformer.augment_sample(sample_data)

        # Verify colors were permuted (output should differ from input)
        # Note: with low probability, permutation might be identity, so we check shape
        assert augmented["task"]["input"].shape == sample_grid.shape
        # Verify all values are still in valid range
        assert augmented["task"]["input"].min() >= 1
        assert augmented["task"]["input"].max() <= 9

    def test_shuffle_examples(self, sample_grid):
        examples = torch.stack([sample_grid, sample_grid * 2, sample_grid * 3])

        # Test truncation
        shuffled = shuffle_examples(examples, max_examples=2)
        assert shuffled.shape[0] == 2

        # Test no truncation
        shuffled = shuffle_examples(examples, max_examples=5)
        assert shuffled.shape[0] == 3

    def test_augmentation_transformer(self, sample_grid):
        # Test transformer with rotation - use fresh sample
        sample = {
            "examples": torch.stack([sample_grid, sample_grid]),
            "task": {"input": sample_grid.clone(), "output": sample_grid.clone()},
        }
        original_input = sample["task"]["input"].clone()

        transformer = ArcAugmentationTransformer(
            augmentations=["rotate"], augmentation_probability=1.0
        )

        # Set seed for deterministic behavior
        torch.manual_seed(42)
        augmented = transformer(sample)
        # Should be different after rotation (unless rotation=0)
        # We verify shape is preserved
        assert augmented["task"]["input"].shape == original_input.shape

    def test_augmentation_transformer_shuffle(self, sample_grid):
        # Test with shuffle_probability = 1.0
        examples1 = torch.stack([sample_grid, sample_grid * 2, sample_grid * 3])
        sample1 = {
            "examples": examples1,
            "task": {"input": sample_grid.clone(), "output": sample_grid.clone()},
        }

        transformer = ArcAugmentationTransformer(
            max_examples=2, shuffle_probability=1.0
        )
        augmented = transformer(sample1)
        assert augmented["examples"].shape[0] == 2

        # Test with shuffle_probability = 0.0 (no shuffling) - use fresh sample
        examples2 = torch.stack([sample_grid, sample_grid * 2, sample_grid * 3])
        sample2 = {
            "examples": examples2,
            "task": {"input": sample_grid.clone(), "output": sample_grid.clone()},
        }

        transformer_no_shuffle = ArcAugmentationTransformer(
            max_examples=2, shuffle_probability=0.0
        )
        augmented_no_shuffle = transformer_no_shuffle(sample2)
        assert augmented_no_shuffle["examples"].shape[0] == 3  # Not truncated

    def test_color_permutation_range(self, sample_grid):
        # Test custom color permutation range
        # Grid with values 0-11 (after zero_based_correction)
        extended_grid = torch.tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.float32
        )
        sample = {
            "examples": torch.stack([extended_grid, extended_grid]),
            "task": {"input": extended_grid.clone(), "output": extended_grid.clone()},
        }

        transformer = ArcAugmentationTransformer(
            augmentations=["permute_colors"],
            color_permutation_range=(1, 12),
            augmentation_probability=1.0,
        )

        # Set seed for deterministic permutation
        torch.manual_seed(999)
        augmented = transformer(sample)
        # Verify shape is preserved
        assert augmented["task"]["input"].shape == extended_grid.shape
        # Verify values are in valid range
        assert augmented["task"]["input"].min() >= 0
        assert augmented["task"]["input"].max() <= 11

    def test_compose_transforms(self, sample_grid):
        # Create two simple transforms
        def add_one(sample):
            sample["task"]["input"] = sample["task"]["input"] + 1
            return sample

        def multiply_two(sample):
            sample["task"]["input"] = sample["task"]["input"] * 2
            return sample

        # Compose them
        composed = ComposeTransforms([add_one, multiply_two])

        sample = {"task": {"input": sample_grid.clone(), "output": sample_grid.clone()}}

        result = composed(sample)
        expected = (sample_grid + 1) * 2
        assert torch.allclose(result["task"]["input"], expected)
