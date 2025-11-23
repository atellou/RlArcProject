"""ARC Dataset Augmentation Utilities !!!

Provides transformations for ARC grid data including:
- Geometric transformations (rotations, reflections)
- Color value permutations
- Spatial translations
- Example shuffling/modification

Module for augmenting ARC dataset samples.
"""

from typing import Dict, List

import torch


def rotate_grid(grid: torch.Tensor, rotations: int) -> torch.Tensor:
    """Rotate grid by 90° increments (0=0°, 1=90°, 2=180°, 3=270°)."""
    return torch.rot90(grid, rotations, dims=(-2, -1))


def reflect_grid(grid: torch.Tensor, axis: int) -> torch.Tensor:
    """Reflect grid horizontally (axis=0) or vertically (axis=1)."""
    return torch.flip(grid, dims=(axis,))


def permute_colors(grid: torch.Tensor, color_map: Dict[int, int]) -> torch.Tensor:
    """Remap color values using a permutation mapping.

    Uses tensor indexing for efficient remapping without numpy.
    """
    # Create a lookup tensor for fast remapping
    max_val = max(max(color_map.keys()), int(grid.max().item())) + 1
    lookup = torch.arange(max_val, dtype=grid.dtype, device=grid.device)

    # Update lookup table with color map
    for old_color, new_color in color_map.items():
        lookup[old_color] = new_color

    # Apply mapping using indexing
    return lookup[grid.long()].to(grid.dtype)


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


class ArcAugmentationTransformer:
    """Apply augmentations to ARC samples.

    This transformer can be passed to ArcDataset to apply augmentations
    during data loading.

    Args:
        augmentations: List of augmentation names to apply:
            - 'rotate': Random rotation (0°, 90°, 180°, 270°)
            - 'reflect': Random reflection (horizontal/vertical)
            - 'permute_colors': Random color permutation
            - 'translate': Random translation
        probability: Probability of applying augmentations (0.0 to 1.0)
        max_examples: Maximum number of examples to keep after shuffling
        shuffle_probability: Probability of shuffling examples (0.0 to 1.0)
    """

    def __init__(
        self,
        augmentations: List[str] | None = None,
        max_examples: int | None = None,
        color_permutation_range: tuple[int, int] = (1, 12),
        augmentation_probability: float = 1.0,
        shuffle_probability: float = 1.0,
    ):
        """Initialize the augmentation transformer.

        Args:
            augmentations: List of augmentation names. If None, applies all.
            max_examples: Maximum examples to keep. If None, keeps all.
            color_permutation_range: Range of colors to permute.
            augmentation_probability:
                Probability of applying augmentations (default: 1.0)
            shuffle_probability:
                Probability of shuffling examples (default: 1.0)
        """
        self.augmentations = augmentations or []
        self.max_examples = max_examples
        self.color_permutation_range = color_permutation_range
        self.augmentation_probability = augmentation_probability
        self.shuffle_probability = shuffle_probability

        # Validate augmentation names
        valid_augmentations = {"rotate", "reflect", "permute_colors", "translate"}
        for aug in self.augmentations:
            if aug not in valid_augmentations:
                raise ValueError(
                    f"Invalid augmentation '{aug}'. "
                    f"Valid options: {valid_augmentations}"
                )

    def augment_sample(self, sample: Dict) -> Dict:
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
        if "rotate" in self.augmentations:
            rotations = torch.randint(0, 4, (1,)).item()  # 0-3 inclusive
            augmented["examples"] = rotate_grid(augmented["examples"], rotations)
            augmented["task"]["input"] = rotate_grid(
                augmented["task"]["input"], rotations
            )
            augmented["task"]["output"] = rotate_grid(
                augmented["task"]["output"], rotations
            )

        if "reflect" in self.augmentations:
            axis = torch.randint(0, 2, (1,)).item()  # 0 or 1
            augmented["examples"] = reflect_grid(augmented["examples"], axis)
            augmented["task"]["input"] = reflect_grid(augmented["task"]["input"], axis)
            augmented["task"]["output"] = reflect_grid(
                augmented["task"]["output"], axis
            )

        # Apply color permutation
        if "permute_colors" in self.augmentations:
            colors = list(range(*self.color_permutation_range))
            # Permute the colors themselves, not indices
            perm_colors = [colors[i] for i in torch.randperm(len(colors)).tolist()]
            color_map = {colors[i]: perm_colors[i] for i in range(len(colors))}

            augmented["examples"] = permute_colors(augmented["examples"], color_map)
            augmented["task"]["input"] = permute_colors(
                augmented["task"]["input"], color_map
            )
            augmented["task"]["output"] = permute_colors(
                augmented["task"]["output"], color_map
            )

        # Apply translation
        if "translate" in self.augmentations:
            dx = torch.randint(-3, 4, (1,)).item()  # -3 to 3 inclusive
            dy = torch.randint(-3, 4, (1,)).item()
            augmented["examples"] = translate_grid(augmented["examples"], dx, dy)
            augmented["task"]["input"] = translate_grid(
                augmented["task"]["input"], dx, dy
            )
            augmented["task"]["output"] = translate_grid(
                augmented["task"]["output"], dx, dy
            )

        return augmented

    def __call__(self, sample: Dict) -> Dict:
        """Apply augmentations to the sample.

        Args:
            sample: Dictionary with 'examples' and 'task' keys

        Returns:
            Augmented sample
        """
        # Apply augmentations with probability
        if self.augmentations and torch.rand(1).item() < self.augmentation_probability:
            sample = self.augment_sample(sample)

        # Shuffle and truncate examples with probability
        if (
            self.max_examples is not None
            and "examples" in sample
            and torch.rand(1).item() < self.shuffle_probability
        ):
            sample["examples"] = shuffle_examples(sample["examples"], self.max_examples)

        return sample
