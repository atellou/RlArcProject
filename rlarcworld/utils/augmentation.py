"""ARC Dataset Augmentation Utilities !!!

Provides transformations for ARC grid data including:
- Geometric transformations (rotations, reflections)
- Color value permutations
- Spatial translations
- Example shuffling/modification

Module for augmenting ARC dataset samples.
"""

import random
from typing import Dict, List

import numpy as np
import torch


def rotate_grid(grid: torch.Tensor, rotations: int) -> torch.Tensor:
    """Rotate grid by 90° increments (0=0°, 1=90°, 2=180°, 3=270°)."""
    return torch.rot90(grid, rotations, dims=(-2, -1))


def reflect_grid(grid: torch.Tensor, axis: int) -> torch.Tensor:
    """Reflect grid horizontally (axis=0) or vertically (axis=1)."""
    return torch.flip(grid, dims=(axis,))


def permute_colors(grid: torch.Tensor, color_map: Dict[int, int]) -> torch.Tensor:
    """Remap color values using a permutation mapping."""
    remapped = np.vectorize(color_map.get)(grid.cpu().numpy())
    return torch.tensor(remapped, dtype=grid.dtype, device=grid.device)


def translate_grid(grid: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Shift grid with zero-padding."""
    translated = torch.zeros_like(grid)
    h, w = grid.shape[-2:]

    # Calculate source and target slices
    src_x = max(0, -dx)
    src_y = max(0, -dy)
    tgt_x = max(0, dx)
    tgt_y = max(0, dy)

    copy_w = w - abs(dx)
    copy_h = h - abs(dy)

    if copy_w > 0 and copy_h > 0:
        translated[..., tgt_y : tgt_y + copy_h, tgt_x : tgt_x + copy_w] = grid[
            ..., src_y : src_y + copy_h, src_x : src_x + copy_w
        ]

    return translated


def augment_sample(sample: Dict, augmentations: List[str]) -> Dict:
    """Apply augmentations to an ARC sample.

    Args:
        sample: Dictionary with 'examples' and 'task' keys
        augmentations: List of augmentations to apply:
          - 'rotate': Random rotation (0°, 90°, 180°, 270°)
          - 'reflect': Random reflection (horizontal/vertical)
          - 'permute_colors': Random color permutation
          - 'translate': Random translation

    Returns:
        Augmented sample
    """
    augmented = sample.copy()

    # Apply geometric transformations to all grids
    if "rotate" in augmentations:
        rotations = random.randint(0, 3)
        augmented["examples"] = rotate_grid(augmented["examples"], rotations)
        augmented["task"]["input"] = rotate_grid(augmented["task"]["input"], rotations)
        augmented["task"]["output"] = rotate_grid(
            augmented["task"]["output"], rotations
        )

    if "reflect" in augmentations:
        axis = random.choice([0, 1])
        augmented["examples"] = reflect_grid(augmented["examples"], axis)
        augmented["task"]["input"] = reflect_grid(augmented["task"]["input"], axis)
        augmented["task"]["output"] = reflect_grid(augmented["task"]["output"], axis)

    # Apply color permutation
    if "permute_colors" in augmentations:
        colors = list(range(1, 10))
        random.shuffle(colors)
        color_map = {i: colors[i - 1] for i in range(1, 10)}
        color_map[0] = 0  # Keep background as 0

        augmented["examples"] = permute_colors(augmented["examples"], color_map)
        augmented["task"]["input"] = permute_colors(
            augmented["task"]["input"], color_map
        )
        augmented["task"]["output"] = permute_colors(
            augmented["task"]["output"], color_map
        )

    # Apply translation
    if "translate" in augmentations:
        dx, dy = random.randint(-3, 3), random.randint(-3, 3)
        augmented["examples"] = translate_grid(augmented["examples"], dx, dy)
        augmented["task"]["input"] = translate_grid(augmented["task"]["input"], dx, dy)
        augmented["task"]["output"] = translate_grid(
            augmented["task"]["output"], dx, dy
        )

    return augmented


def shuffle_examples(examples: torch.Tensor, max_examples: int = 10) -> torch.Tensor:
    """Shuffle and select a subset of examples.

    Args:
        examples: Tensor of shape (num_examples, 2, H, W)
        max_examples: Maximum number of examples to keep

    Returns:
        Shuffled and truncated examples tensor
    """
    num_examples = examples.shape[0]
    if num_examples > max_examples:
        indices = torch.randperm(num_examples)[:max_examples]
        return examples[indices]
    return examples
