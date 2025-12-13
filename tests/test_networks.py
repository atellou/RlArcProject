import logging

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rlarcworld.common.enums import (
    ActionKeys,
    GridEncoderIdsType,
    RewardKeys,
    StateKeys,
)
from rlarcworld.models.actor import ViTARCActor
from rlarcworld.models.critic import VitARCCritic
from rlarcworld.models.grid_encoder import GridEncoder, ObjectIndexExtractor
from rlarcworld.models.reward_world_model import RewardWorldModel
from rlarcworld.nn_modules import RlArcWorldViTARCConfig

logger = logging.getLogger(__name__)


# ---------------- Fixtures ----------------


@pytest.fixture()
def batch_size() -> int:
    # Randomize to keep test variability but cast to int
    return int(torch.randint(1, 20, size=(1,)).item())


# ---------------- Helpers ----------------
def train_arc_critic_network(batch_size: int):
    grid_size = 16
    color_values = 11
    logger.info(
        "Testing VitARCCritic with batch size: %s, grid size: %s and color values: %s",
        batch_size,
        grid_size,
        color_values,
    )
    num_atoms = {
        RewardKeys.PIXEL_WISE.value: int(torch.randint(50, 100, size=(1,)).item()),
        RewardKeys.BINARY.value: 1,
    }
    v_min = {RewardKeys.PIXEL_WISE.value: -40, RewardKeys.BINARY.value: 0}
    v_max = {RewardKeys.PIXEL_WISE.value: 2, RewardKeys.BINARY.value: 1}

    rl_vit_arc_config = RlArcWorldViTARCConfig(
        grid_size=grid_size,
        d_model=64,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        use_ope=False,
        grids_to_encode=[StateKeys.GRID.value],
    )
    network = VitARCCritic(color_values, rl_vit_arc_config, num_atoms, v_min, v_max)

    input_sample = TensorDict(
        {
            StateKeys.LAST_GRID.value: torch.randint(
                0, color_values, size=(batch_size, 1, grid_size, grid_size)
            ),
            StateKeys.GRID.value: torch.randint(
                0, color_values, size=(batch_size, 1, grid_size, grid_size)
            ),
            StateKeys.EXAMPLES.value: torch.randint(
                0, color_values, size=(batch_size, 10, 2, grid_size, grid_size)
            ),
            StateKeys.INITIAL.value: torch.randint(
                0, color_values, size=(batch_size, 1, grid_size, grid_size)
            ),
            StateKeys.INDEX.value: torch.randint(
                0, grid_size, size=(batch_size, 1, grid_size, grid_size)
            ),
        }
    )

    network.input_val(input_sample)
    for key in input_sample.keys():
        dc = input_sample.clone()
        import pytest

        with pytest.raises(AssertionError):
            dc.pop(key)
            network.input_val(dc)

    action_probs = TensorDict(
        {
            ActionKeys.X_LOCATION.value: torch.softmax(
                torch.randn(batch_size, grid_size), dim=-1
            ),
            ActionKeys.Y_LOCATION.value: torch.softmax(
                torch.randn(batch_size, grid_size), dim=-1
            ),
            ActionKeys.COLOR_VALUES.value: torch.softmax(
                torch.randn(batch_size, color_values), dim=-1
            ),
            ActionKeys.SUBMIT.value: torch.softmax(torch.randn(batch_size, 2), dim=-1),
        }
    )

    to_zero = []
    for key in input_sample.keys():
        if torch.rand(1).item() < 0.2:
            input_sample[key] = input_sample[key] * 0.0
            to_zero.append(key)
    for key in action_probs.keys():
        if torch.rand(1).item() < 0.2:
            action_probs[key] = action_probs[key] * 0.0
            to_zero.append(key)

    org_sample = input_sample.clone()
    output = network(input_sample, action=action_probs)

    torch.testing.assert_close(input_sample, org_sample)

    network.output_val(output)

    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.RMSprop(network.parameters())

    target = TensorDict(
        {
            RewardKeys.PIXEL_WISE.value: torch.softmax(
                torch.randint(
                    -40,
                    2,
                    size=(batch_size, num_atoms[RewardKeys.PIXEL_WISE.value]),
                ).float(),
                dim=-1,
            ),
            RewardKeys.BINARY.value: torch.randint(
                0, 2, size=(batch_size, num_atoms[RewardKeys.BINARY.value])
            ).float(),
        }
    ).to(network.device)

    for key, dist in output.items():
        (
            torch.testing.assert_close(
                torch.sum(dist, dim=1), torch.ones(batch_size).to(network.device)
            ),
            f"Probability mass function not normalized for key: {key}",
        )
        assert torch.all(dist >= 0), f"Negative probability values for key: {key}"
        assert torch.all(dist <= 1), f"Probability values greater than 1 for key: {key}"

    loss = criterion(
        output[RewardKeys.PIXEL_WISE.value], target[RewardKeys.PIXEL_WISE.value]
    ) + criterion(output[RewardKeys.BINARY.value], target[RewardKeys.BINARY.value])

    assert not torch.isnan(loss), f"{criterion._get_name()} is NaN for Critic network"

    optimizer.zero_grad()
    loss.backward()

    # Check gradients flow (same pattern as ViTARCActor test)
    grad_count = 0
    for name, param in network.named_parameters():
        if param.grad is not None:
            grad_count += 1

    assert grad_count > 0, "No gradients flowing in VitARCCritic"

    optimizer.step()

    return loss


def train_vitarc_actor(batch_size: int):
    """Train ViTARCActor for one step and return loss."""
    grid_size = 16
    color_values = 11
    num_examples = 2

    logger.info(
        "Testing ViTARCActor with batch size: %s, grid_size: %s",
        batch_size,
        grid_size,
    )

    rl_vit_arc_config = RlArcWorldViTARCConfig(
        grid_size=grid_size,
        d_model=64,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        use_ope=False,
        grids_to_encode=[StateKeys.GRID.value],
    )
    # Use minimal config for faster training
    network = ViTARCActor(
        color_values=color_values,
        rl_vit_arc_config=rl_vit_arc_config,
    )

    input_sample = TensorDict(
        {
            StateKeys.LAST_GRID.value: torch.randint(
                0, color_values, size=(batch_size, 1, grid_size, grid_size)
            ),
            StateKeys.GRID.value: torch.randint(
                0, color_values, size=(batch_size, 1, grid_size, grid_size)
            ),
            StateKeys.EXAMPLES.value: torch.randint(
                0,
                color_values,
                size=(batch_size, num_examples, 2, grid_size, grid_size),
            ),
            StateKeys.INITIAL.value: torch.randint(
                0, color_values, size=(batch_size, 1, grid_size, grid_size)
            ),
            StateKeys.INDEX.value: torch.randint(
                0, grid_size, size=(batch_size, 1, grid_size, grid_size)
            ),
        }
    )

    output = network(input_sample)

    # Use KLDivLoss since output is softmax probabilities, not logits
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    target = TensorDict(
        {
            ActionKeys.X_LOCATION.value: torch.softmax(
                torch.rand(size=(batch_size, grid_size)), dim=1
            ),
            ActionKeys.Y_LOCATION.value: torch.softmax(
                torch.rand(size=(batch_size, grid_size)), dim=1
            ),
            ActionKeys.COLOR_VALUES.value: torch.softmax(
                torch.rand(size=(batch_size, color_values)), dim=1
            ),
            ActionKeys.SUBMIT.value: torch.softmax(
                torch.rand(size=(batch_size, 2)), dim=1
            ),
        }
    ).to(network.device)

    # KLDivLoss expects log-probabilities as input
    loss = sum([criterion(output[k].log(), target[k]) for k in output.keys()])

    assert not torch.isnan(loss), "Loss is NaN for ViTARCActor"

    optimizer.zero_grad()
    loss.backward()

    # Check gradients flow
    grad_count = 0
    for name, param in network.named_parameters():
        if param.grad is not None:
            grad_count += 1

    assert grad_count > 0, "No gradients flowing in ViTARCActor"

    optimizer.step()
    return loss


def train_reward_world_model(batch_size: int):
    """Train reward world model for one step and return loss."""
    size = 16
    color_values = 11
    logger.info(
        "Testing RewardWorldModel with batch size: %s, size: %s and color values: %s",
        batch_size,
        size,
        color_values,
    )

    network = RewardWorldModel(
        grid_size=size,
        color_values=color_values,
        embedding_size=128,
    )

    # Create input state
    input_sample = TensorDict(
        {
            StateKeys.LAST_GRID.value: torch.randint(
                0, color_values, size=(batch_size, 1, size, size)
            ),
            StateKeys.GRID.value: torch.randint(
                0, color_values, size=(batch_size, 1, size, size)
            ),
            StateKeys.EXAMPLES.value: torch.randint(
                0, color_values, size=(batch_size, 10, 2, size, size)
            ),
            StateKeys.INITIAL.value: torch.randint(
                0, color_values, size=(batch_size, 1, size, size)
            ),
            StateKeys.INDEX.value: torch.randint(
                0, size, size=(batch_size, 1, size, size)
            ),
        }
    )

    # Test input validation
    network.input_val(input_sample)
    for key in input_sample.keys():
        dc = input_sample.clone()
        with pytest.raises(AssertionError):
            dc.pop(key)
            network.input_val(dc)

    # Forward pass
    optimizer = torch.optim.RMSprop(network.parameters())
    criterion = torch.nn.MSELoss()

    output = network(input_sample)

    # Validate output structure
    network.output_val(output)
    assert RewardKeys.BINARY.value in output
    assert RewardKeys.PIXEL_WISE.value in output

    # Check output shapes and ranges
    assert output[RewardKeys.BINARY.value].shape == (
        batch_size,
        2,
    )  # [P(incorrect), P(correct)]
    assert output[RewardKeys.PIXEL_WISE.value].shape == (batch_size,)

    # Binary should be probability distribution (from softmax)
    assert torch.all(output[RewardKeys.BINARY.value] >= 0)
    assert torch.all(output[RewardKeys.BINARY.value] <= 1)
    # Should sum to 1 across classes
    assert torch.allclose(
        output[RewardKeys.BINARY.value].sum(dim=-1),
        torch.ones(batch_size, device=network.device),
        atol=1e-6,
    )

    # Pixel-wise should be percentage in [0, 1] (from sigmoid)
    # Represents percentage of wrong pixels (grid_diff / grid_size²)
    assert torch.all(output[RewardKeys.PIXEL_WISE.value] >= 0)
    assert torch.all(output[RewardKeys.PIXEL_WISE.value] <= 1)

    # Create synthetic targets
    target = TensorDict(
        {
            RewardKeys.BINARY.value: torch.randint(
                0, 2, size=(batch_size,)
            ),  # Class indices
            RewardKeys.PIXEL_WISE.value: torch.rand(batch_size),  # Percentage in [0, 1]
        }
    ).to(network.device)

    # Compute loss
    # Binary: cross-entropy (classification)
    binary_loss = F.cross_entropy(
        output[RewardKeys.BINARY.value], target[RewardKeys.BINARY.value]
    )
    # Pixel-wise: MSE (regression)
    pixel_wise_loss = criterion(
        output[RewardKeys.PIXEL_WISE.value], target[RewardKeys.PIXEL_WISE.value]
    )
    loss = binary_loss + pixel_wise_loss

    assert not torch.isnan(loss), "Loss is NaN for RewardWorldModel"

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Check gradients flow
    to_zero: list[str] = []  # Keys that might have zero gradients due to masking
    for name, param in network.named_parameters():
        input_key = name.split(".")[1] if "." in name else name
        if "base_model" not in name and param.grad is None:
            raise ValueError(f"Gradient not flowing in RewardWorldModel for: {name}")
        elif "base_model" not in name and input_key not in to_zero:
            assert not torch.all(param.grad.abs().sum() == 0) or torch.all(
                input_sample.get(input_key, torch.tensor(0)) == 0
            ), f"Gradient of zero for RewardWorldModel for: {name}"

    optimizer.step()
    return loss


# ---------------- GridEncoder Tests ----------------
class TestGridEncoder:
    def test_grid_tokenizer_encode_grid(self):
        """Test GridEncoder.encode_grid produces correct shapes and values."""
        logger.info("Testing GridEncoder.encode_grid")
        batch_size = 4
        grid_size = 30
        color_values = 11

        tokenizer = GridEncoder(grid_size=grid_size)

        # Create random grid [B, 1, H, W]
        grid = torch.randint(
            0, color_values, size=(batch_size, 1, grid_size, grid_size)
        )

        input_ids, attention_mask, object_idx = tokenizer.encode_grid(grid)

        # Check shapes
        # Expected seq_len: (grid_size + 1) * (grid_size + 1) + 2 (BOS + grid + EOS)
        expected_grid_tokens = (grid_size + 1) * (grid_size + 1)
        expected_seq_len = 1 + expected_grid_tokens + 1  # BOS + grid + EOS

        assert input_ids.shape == (batch_size, expected_seq_len)
        assert attention_mask.shape == (batch_size, expected_seq_len)
        assert object_idx is None  # No object extractor provided

        # Check BOS and EOS tokens
        assert torch.all(input_ids[:, 0] == GridEncoderIdsType.BOS_ID)
        assert torch.all(input_ids[:, -1] == GridEncoderIdsType.EOS_ID)

        # Check attention mask is 1 for non-padding
        assert torch.all(attention_mask[input_ids != GridEncoderIdsType.PAD_ID] == 1)

    def test_grid_tokenizer_with_object_extractor(self):
        """Test GridEncoder with ObjectIndexExtractor returns object_idx."""
        logger.info("Testing GridEncoder with ObjectIndexExtractor")
        batch_size = 2
        grid_size = 10  # Smaller for faster test
        color_values = 11

        tokenizer = GridEncoder(grid_size=grid_size, use_ope=True)

        grid = torch.randint(
            0, color_values, size=(batch_size, 1, grid_size, grid_size)
        )

        input_ids, attention_mask, object_idx = tokenizer.encode_grid(grid)

        assert object_idx is not None
        assert object_idx.shape == input_ids.shape
        # Object IDs should be non-negative
        assert torch.all(object_idx >= 0)

    def test_grid_tokenizer_encode_temporal_grids(self):
        """Test GridEncoder.encode_temporal_grids with all grids."""
        logger.info("Testing GridEncoder.encode_temporal_grids")
        batch_size = 2
        grid_size = 10
        color_values = 11

        tokenizer = GridEncoder(grid_size=grid_size)

        current = torch.randint(
            0, color_values, size=(batch_size, 1, grid_size, grid_size)
        )
        initial = torch.randint(
            0, color_values, size=(batch_size, 1, grid_size, grid_size)
        )
        last_grid = torch.randint(
            0, color_values, size=(batch_size, 1, grid_size, grid_size)
        )

        input_ids, attention_mask, temporal_ids, object_idx = (
            tokenizer.encode_temporal_grids(
                current=current,
                initial=initial,
                last_grid=last_grid,
            )
        )

        # Check shapes match
        assert input_ids.shape == attention_mask.shape
        assert input_ids.shape == temporal_ids.shape

        # Check temporal IDs: 0=initial, 1=last, 2=current
        assert torch.all(temporal_ids >= 0)
        assert torch.all(temporal_ids <= 2)

        # Check BOS token
        assert torch.all(input_ids[:, 0] == GridEncoderIdsType.BOS_ID)

    def test_grid_tokenizer_encode_example_pair(self):
        """Test GridEncoder.encode_example_pair."""
        logger.info("Testing GridEncoder.encode_example_pair")
        batch_size = 2
        grid_size = 10
        color_values = 11

        tokenizer = GridEncoder(grid_size=grid_size)

        # Example pair: [B, 2, H, W] where dim 1 is (input, output)
        example = torch.randint(
            0, color_values, size=(batch_size, 2, grid_size, grid_size)
        )

        input_ids, attention_mask, object_idx = tokenizer.encode_example_pair(example)

        # Check shapes
        assert input_ids.shape == attention_mask.shape

        # Check BOS and EOS
        assert torch.all(input_ids[:, 0] == GridEncoderIdsType.BOS_ID)
        assert torch.all(input_ids[:, -1] == GridEncoderIdsType.EOS_ID)

        # Check SEP token exists (between input and output grids)
        assert torch.any(input_ids == GridEncoderIdsType.SEP_ID)

    def test_object_index_extractor(self):
        """Test ObjectIndexExtractor produces valid object indices."""
        logger.info("Testing ObjectIndexExtractor")
        batch_size = 2
        grid_size = 10

        extractor = ObjectIndexExtractor(padding_id=0)

        # Create grid with distinct colored regions
        grid = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.long)
        # Add some colored objects
        grid[:, 2:5, 2:5] = 1  # Red square
        grid[:, 6:8, 6:8] = 2  # Blue square

        object_idx = extractor(grid)

        assert object_idx.shape == (batch_size, grid_size * grid_size)
        # Background should be 0
        # Objects should have positive IDs
        assert torch.all(object_idx >= 0)


# ---------------- ViTARCActor Tests ----------------


class TestViTARCActor:
    def test_vitarc_actor_forward(self):
        """Test ViTARCActor forward pass produces valid action probabilities."""
        logger.info("Testing ViTARCActor forward pass")
        batch_size = 2
        grid_size = 30
        color_values = 11
        num_examples = 3

        rl_vit_arc_config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            grids_to_encode=[StateKeys.GRID.value],
        )
        # Create actor with minimal config for faster test
        network = ViTARCActor(
            color_values=color_values,
            rl_vit_arc_config=rl_vit_arc_config,
        )

        input_sample = TensorDict(
            {
                StateKeys.LAST_GRID.value: torch.randint(
                    0, color_values, size=(batch_size, 1, grid_size, grid_size)
                ),
                StateKeys.GRID.value: torch.randint(
                    0, color_values, size=(batch_size, 1, grid_size, grid_size)
                ),
                StateKeys.EXAMPLES.value: torch.randint(
                    0,
                    color_values,
                    size=(batch_size, num_examples, 2, grid_size, grid_size),
                ),
                StateKeys.INITIAL.value: torch.randint(
                    0, color_values, size=(batch_size, 1, grid_size, grid_size)
                ),
                StateKeys.INDEX.value: torch.randint(
                    0, grid_size, size=(batch_size, 1, grid_size, grid_size)
                ),
            }
        )

        output = network(input_sample)

        # Check output keys
        assert ActionKeys.X_LOCATION.value in output
        assert ActionKeys.Y_LOCATION.value in output
        assert ActionKeys.COLOR_VALUES.value in output
        assert ActionKeys.SUBMIT.value in output

        # Check shapes
        assert output[ActionKeys.X_LOCATION.value].shape == (batch_size, grid_size)
        assert output[ActionKeys.Y_LOCATION.value].shape == (batch_size, grid_size)
        assert output[ActionKeys.COLOR_VALUES.value].shape == (batch_size, color_values)
        assert output[ActionKeys.SUBMIT.value].shape == (batch_size, 2)

        # Check probabilities sum to 1
        for key, probs in output.items():
            torch.testing.assert_close(
                probs.sum(dim=-1),
                torch.ones(batch_size, device=network.device),
                atol=1e-5,
                rtol=1e-5,
            )
            assert torch.all(probs >= 0), f"Negative probabilities for {key}"
            assert torch.all(probs <= 1), f"Probabilities > 1 for {key}"

    def test_train_vitarc_actor(self, batch_size: int):
        """Validate ViTARCActor is trainable by observing loss variability."""
        losses = [train_vitarc_actor(batch_size) for _ in range(3)]
        variability = sum(
            [
                not torch.isclose(losses[i], losses[i + 1])
                for i in range(len(losses) - 1)
            ]
        ) / max(1, (len(losses) - 1))
        assert variability >= 0.3  # Lower threshold due to smaller model


class TestViTARCCritic:
    def test_train_arc_critic_network(self, batch_size: int):
        """Validate the critic network is trainable by observing loss variability."""
        losses = [train_arc_critic_network(batch_size) for _ in range(3)]
        variability = sum(
            [
                not torch.isclose(losses[i], losses[i + 1])
                for i in range(len(losses) - 1)
            ]
        ) / max(1, (len(losses) - 1))
        assert variability >= 0.5


class TestRewardWorldModel:
    def test_train_reward_world_model(self, batch_size: int):
        """Validate the reward world model is trainable by loss variability."""
        losses = [train_reward_world_model(batch_size) for _ in range(3)]
        variability = sum(
            [
                not torch.isclose(losses[i], losses[i + 1])
                for i in range(len(losses) - 1)
            ]
        ) / max(1, (len(losses) - 1))
        assert variability >= 0.5
