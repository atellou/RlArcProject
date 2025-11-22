"""Tests for ARC augmentation utilities."""

import unittest.mock

import pytest
import torch

from rlarcworld.utils.augmentation import (
    augment_sample,
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
        # Mock rotation to always be 90°
        with unittest.mock.patch(
            "rlarcworld.utils.augmentation.random.randint", return_value=1
        ):
            augmented = augment_sample(sample_data, ["rotate"])

            # Verify examples were rotated
            rotated = rotate_grid(sample_grid, 1)
            expected_examples = torch.stack([rotated, rotated])
            assert torch.allclose(augmented["examples"], expected_examples)

            # Verify task input was rotated
            assert torch.allclose(augmented["task"]["input"], rotated)

    def test_augment_sample_color_permutation(self, sample_data):
        # Mock color permutation to reverse colors
        with unittest.mock.patch(
            "rlarcworld.utils.augmentation.random.shuffle",
            side_effect=lambda x: x.reverse(),
        ):
            augmented = augment_sample(sample_data, ["permute_colors"])

            # Original colors: 1-9, reversed mapping: 9->1, 8->2, etc.
            expected = torch.tensor(
                [[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=torch.float32
            )

            assert torch.allclose(augmented["task"]["input"], expected)

    def test_shuffle_examples(self, sample_grid):
        examples = torch.stack([sample_grid, sample_grid * 2, sample_grid * 3])

        # Test truncation
        shuffled = shuffle_examples(examples, max_examples=2)
        assert shuffled.shape[0] == 2

        # Test no truncation
        shuffled = shuffle_examples(examples, max_examples=5)
        assert shuffled.shape[0] == 3
