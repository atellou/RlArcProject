import torch
from tensordict import TensorDict

from rlarcworld.common.enums import StateKeys


def make_state_tensordict(
    batch_size: int,
    grid_size: int,
    color_values: int,
    num_examples: int = 10,
    device: torch.device | str = "cpu",
) -> TensorDict:
    """Create a synthetic state TensorDict for testing.

    Args:
        batch_size: Number of samples in batch.
        grid_size: Size of the grid (H=W).
        color_values: Number of color values.
        num_examples: Number of example pairs.
        device: Device to create tensors on.

    Returns:
        TensorDict with all required state keys.
    """
    return TensorDict(
        {
            StateKeys.LAST_GRID.value: torch.randint(
                0, color_values, (batch_size, 1, grid_size, grid_size), device=device
            ),
            StateKeys.GRID.value: torch.randint(
                0, color_values, (batch_size, 1, grid_size, grid_size), device=device
            ),
            StateKeys.EXAMPLES.value: torch.randint(
                0,
                color_values,
                (batch_size, num_examples, 2, grid_size, grid_size),
                device=device,
            ),
            StateKeys.INITIAL.value: torch.randint(
                0, color_values, (batch_size, 1, grid_size, grid_size), device=device
            ),
            StateKeys.INDEX.value: torch.randint(
                0, grid_size, (batch_size, 1, grid_size, grid_size), device=device
            ),
        }
    )


def make_synthetic_step_on_device(
    device: torch.device, grid_size: int, color_values: int
):
    """Returns a function that builds a synthetic training step.

    The returned function has signature: (batch_size: int, actor) -> TensorDict
    and constructs keys matching what the trainer expects.
    """

    def _make(batch_size: int, actor) -> TensorDict:
        state = TensorDict(
            {
                "last_grid": torch.randint(
                    0,
                    color_values,
                    (batch_size, 1, grid_size, grid_size),
                    device=device,
                ),
                "grid": torch.randint(
                    0,
                    color_values,
                    (batch_size, 1, grid_size, grid_size),
                    device=device,
                ),
                "examples": torch.randint(
                    0,
                    color_values,
                    (batch_size, 3, 2, grid_size, grid_size),
                    device=device,
                ),
                "initial": torch.randint(
                    0,
                    color_values,
                    (batch_size, 1, grid_size, grid_size),
                    device=device,
                ),
                "index": torch.randint(
                    0, grid_size, (batch_size, 1, grid_size, grid_size), device=device
                ),
                "terminated": torch.zeros(
                    (batch_size, 1), device=device, dtype=torch.long
                ),
            }
        ).to(device)
        with torch.no_grad():
            actions = actor(state)
        reward = TensorDict(
            {
                "pixel_wise": torch.randint(
                    -40, 2, size=(batch_size, 1), device=device
                ).to(torch.float32),
                "binary": torch.randint(0, 2, size=(batch_size, 1), device=device).to(
                    torch.float32
                ),
            }
        )
        next_state = state.clone()
        return TensorDict(
            {
                "state": state,
                "actions": actions,
                "reward": reward,
                "next_state": next_state,
                "terminated": torch.zeros((batch_size, 1), device=device),
            },
            batch_size=[batch_size],
        )

    return _make
