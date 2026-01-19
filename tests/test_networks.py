import logging

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rlarcworld.common.configs import RlArcWorldViTARCConfig
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
from rlarcworld.utils.device import CudaConfig

logger = logging.getLogger(__name__)


# ---------------- Fixtures ----------------


@pytest.fixture()
def batch_size() -> int:
    # Randomize to keep test variability but cap at 8 to avoid OOM on smaller GPUs
    return int(torch.randint(1, 9, size=(1,)).item())


# ---------------- Helpers ----------------
def train_arc_critic_network(batch_size: int, device: str = "cpu"):
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
    )
    device_config = {"device": device, "amp_enabled": False, "use_checkpointing": False}
    network = VitARCCritic(
        color_values,
        rl_vit_arc_config,
        num_atoms,
        v_min,
        v_max,
        device_config=device_config,
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

    network.output_val(output.q_dist)

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

    for key, dist in output.q_dist.items():
        (
            torch.testing.assert_close(
                torch.sum(dist, dim=1), torch.ones(batch_size).to(network.device)
            ),
            f"Probability mass function not normalized for key: {key}",
        )
        assert torch.all(dist >= 0), f"Negative probability values for key: {key}"
        assert torch.all(dist <= 1), f"Probability values greater than 1 for key: {key}"

    loss = criterion(
        output.q_log_dist[RewardKeys.PIXEL_WISE.value],
        target[RewardKeys.PIXEL_WISE.value],
    ) + criterion(
        output.q_log_dist[RewardKeys.BINARY.value], target[RewardKeys.BINARY.value]
    )

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


def train_vitarc_actor(batch_size: int, device: str = "cpu"):
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
    )
    device_config = CudaConfig(
        device=device, amp_enabled=False, use_checkpointing=False
    )
    # Use minimal config for faster training
    network = ViTARCActor(
        color_values=color_values,
        rl_vit_arc=rl_vit_arc_config,
        device_config=device_config,
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


def train_reward_world_model(batch_size: int, device: str = "cpu"):
    """Train reward world model for one step and return loss."""
    grid_size = 16
    color_values = 11
    num_examples = 3
    logger.info(
        "Testing ViTARCRewardModel with batch size: %s, grid_size: %s",
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
    )
    device_config = CudaConfig(
        device=device, amp_enabled=False, use_checkpointing=False
    )
    network = RewardWorldModel(
        color_values=color_values,
        rl_vit_arc_config=rl_vit_arc_config,
        device_config=device_config,
    )

    # Create input state
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

    # Forward pass
    optimizer = torch.optim.RMSprop(network.parameters())

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
    assert output[RewardKeys.PIXEL_WISE.value].shape == (batch_size, 3)

    # Binary should be probability distribution (from softmax)
    assert torch.all(output[RewardKeys.BINARY.value] >= 0)
    assert torch.all(output[RewardKeys.BINARY.value] <= 1)
    # Should sum to 1 across classes
    assert torch.allclose(
        output[RewardKeys.BINARY.value].sum(dim=-1),
        torch.ones(batch_size, device=network.device),
        atol=1e-6,
    )

    assert torch.isfinite(output[RewardKeys.PIXEL_WISE.value]).all()

    # Create synthetic targets
    target = TensorDict(
        {
            RewardKeys.BINARY.value: torch.randint(
                0, 2, size=(batch_size,)
            ),  # Class indices
            RewardKeys.PIXEL_WISE.value: torch.randint(0, 3, size=(batch_size,)),
        }
    ).to(network.device)

    # Compute loss
    # Binary: cross-entropy (classification)
    log_binary_pred = torch.log(output[RewardKeys.BINARY.value].clamp_min(1e-10))
    binary_loss = F.nll_loss(log_binary_pred, target[RewardKeys.BINARY.value])
    pixel_wise_loss = F.cross_entropy(
        output[RewardKeys.PIXEL_WISE.value], target[RewardKeys.PIXEL_WISE.value]
    )
    loss = binary_loss + pixel_wise_loss

    assert not torch.isnan(loss), "Loss is NaN for RewardWorldModel"

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Check gradients flow (same pattern as ViTARCActor test)
    grad_count = 0
    for name, param in network.named_parameters():
        if param.grad is not None:
            grad_count += 1

    assert grad_count > 0, "No gradients flowing in ViTARCRewardModel"

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_object_index_extractor_cuda(self):
        """Test ObjectIndexExtractor works with CUDA tensors."""
        logger.info("Testing ObjectIndexExtractor with CUDA tensors")
        batch_size = 2
        grid_size = 10

        extractor = ObjectIndexExtractor(padding_id=0)

        # Create grid with distinct colored regions on CUDA
        grid = torch.zeros(
            batch_size, grid_size, grid_size, dtype=torch.long, device="cuda"
        )
        # Add some colored objects
        grid[:, 2:5, 2:5] = 1  # Red square
        grid[:, 6:8, 6:8] = 2  # Blue square

        object_idx = extractor(grid)

        assert object_idx.shape == (batch_size, grid_size * grid_size)
        # Result should be on the same device as input
        assert object_idx.device.type == "cuda"
        # Objects should have valid IDs
        assert torch.all(object_idx >= 0)


# ---------------- RlArcWorldViTARC Tests ----------------
class TestRlArcWorldViTARC:
    """Tests for the RlArcWorldViTARC encoder module."""

    def test_init_with_default_config(self):
        """Test RlArcWorldViTARC initialization with default best config."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC

        config = RlArcWorldViTARCConfig.best_config(grid_size=10)
        model = RlArcWorldViTARC(config)

        # Check default best config values are applied
        assert model.use_ope == RlArcWorldViTARC.BEST_USE_OPE
        assert model.t5_config.ape_type == RlArcWorldViTARC.BEST_APE_TYPE
        assert model.t5_config.rpe_type == RlArcWorldViTARC.BEST_RPE_TYPE
        assert model.t5_config.ape_mixer == RlArcWorldViTARC.BEST_APE_MIXER

    def test_init_with_custom_config(self):
        """Test RlArcWorldViTARC initialization with custom config."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC

        config = RlArcWorldViTARCConfig(
            grid_size=8,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            ape_type="SinusoidalAPE2D",
            rpe_type="Two-slope-Alibi",
            ape_mixer="default",
        )
        model = RlArcWorldViTARC(config)

        assert model.use_ope is False
        assert model.t5_config.d_model == 64
        assert model.t5_config.num_layers == 2

    def test_init_with_temporal_grids(self):
        """Test RlArcWorldViTARC with temporal grid encoding enabled."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC

        config = RlArcWorldViTARCConfig(
            grid_size=8,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            use_temporal=True,
        )
        model = RlArcWorldViTARC(config)

        assert model.use_temporal is True
        assert hasattr(model, "temporal_embed")

    def test_init_with_examples(self):
        """Test RlArcWorldViTARC with example cross-attention enabled."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC

        config = RlArcWorldViTARCConfig(
            grid_size=8,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            use_examples=True,
        )
        model = RlArcWorldViTARC(config)

        assert model.use_examples is True
        assert hasattr(model, "example_cross_attention")

    def test_forward_single_grid(self):
        """Test forward pass with single grid input."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC, ViTARCEncoderOutput

        batch_size = 2
        grid_size = 8
        color_values = 11

        config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            use_temporal=False,
            use_examples=False,
        )
        model = RlArcWorldViTARC(config)

        state = TensorDict(
            {
                StateKeys.GRID.value: torch.randint(
                    0, color_values, (batch_size, 1, grid_size, grid_size)
                ),
            }
        )

        output = model(state)

        assert isinstance(output, ViTARCEncoderOutput)
        assert output.hidden_states is not None
        assert output.hidden_states.shape[0] == batch_size
        assert output.hidden_states.shape[2] == 64  # d_model
        assert output.attention_mask is not None
        assert output.temporal_ids is None  # No temporal encoding

    def test_forward_with_temporal_grids(self):
        """Test forward pass with temporal grid inputs."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC, ViTARCEncoderOutput

        batch_size = 2
        grid_size = 8
        color_values = 11

        config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            use_temporal=True,
            use_examples=False,
        )
        model = RlArcWorldViTARC(config)

        state = TensorDict(
            {
                StateKeys.GRID.value: torch.randint(
                    0, color_values, (batch_size, 1, grid_size, grid_size)
                ),
                StateKeys.INITIAL.value: torch.randint(
                    0, color_values, (batch_size, 1, grid_size, grid_size)
                ),
                StateKeys.LAST_GRID.value: torch.randint(
                    0, color_values, (batch_size, 1, grid_size, grid_size)
                ),
            }
        )

        output = model(state)

        assert isinstance(output, ViTARCEncoderOutput)
        assert output.hidden_states is not None
        assert output.temporal_ids is not None

    def test_forward_with_examples(self):
        """Test forward pass with example cross-attention."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC, ViTARCEncoderOutput

        batch_size = 2
        grid_size = 8
        color_values = 11
        num_examples = 3

        config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            use_temporal=False,
            use_examples=True,
        )
        model = RlArcWorldViTARC(config)

        state = TensorDict(
            {
                StateKeys.GRID.value: torch.randint(
                    0, color_values, (batch_size, 1, grid_size, grid_size)
                ),
                StateKeys.EXAMPLES.value: torch.randint(
                    0, color_values, (batch_size, num_examples, 2, grid_size, grid_size)
                ),
            }
        )

        output = model(state)

        assert isinstance(output, ViTARCEncoderOutput)
        assert output.hidden_states is not None
        assert output.example_hidden is not None
        # No padding so example mask is None
        assert output.example_mask is None

    def test_forward_with_index_grid(self):
        """Test forward pass when using the index grid (timestamp grid)."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC, ViTARCEncoderOutput

        batch_size = 2
        grid_size = 8
        color_values = 11

        config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            use_index_grid=True,
            use_temporal=False,
            use_examples=False,
        )
        model = RlArcWorldViTARC(config)

        # Build state with grid and matching index grid (timestamps)
        grid = torch.randint(0, color_values, (batch_size, 1, grid_size, grid_size))
        # Simple synthetic timestamp grid: increasing integers per cell
        index_values = (
            torch.arange(grid_size * grid_size)
            .view(1, 1, grid_size, grid_size)
            .repeat(batch_size, 1, 1, 1)
        )

        state = TensorDict(
            {
                StateKeys.GRID.value: grid,
                StateKeys.INDEX.value: index_values,
            }
        )

        output = model(state)

        assert isinstance(output, ViTARCEncoderOutput)
        assert output.hidden_states is not None
        # index_ids present and aligned with input_ids when use_index_grid=True
        assert output.index_ids is not None
        assert output.index_ids.shape == output.input_ids.shape

    def test_trainable_ape_parameter_count(self):
        """Test that trainable_ape affects parameter count."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC

        grid_size = 8

        # Frozen APE
        config_frozen = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            trainable_ape=False,
        )
        model_frozen = RlArcWorldViTARC(config_frozen)
        trainable_frozen = sum(
            p.numel() for p in model_frozen.parameters() if p.requires_grad
        )

        # Trainable APE
        config_trainable = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            trainable_ape=True,
        )
        model_trainable = RlArcWorldViTARC(config_trainable)
        trainable_trainable = sum(
            p.numel() for p in model_trainable.parameters() if p.requires_grad
        )

        # Trainable APE should have more trainable parameters
        assert trainable_trainable > trainable_frozen

    def test_encoder_output_structure(self):
        """Test that ViTARCEncoderOutput contains all expected fields."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC

        batch_size = 2
        grid_size = 8
        color_values = 11

        config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            use_examples=False,
            use_temporal=False,
            use_index_grid=False,
        )
        model = RlArcWorldViTARC(config)

        state = TensorDict(
            {
                StateKeys.GRID.value: torch.randint(
                    0, color_values, (batch_size, 1, grid_size, grid_size)
                ),
            }
        )

        output = model(state)

        # Check all fields exist
        assert hasattr(output, "hidden_states")
        assert hasattr(output, "attention_mask")
        assert hasattr(output, "encoder_outputs")
        assert hasattr(output, "input_ids")
        assert hasattr(output, "temporal_ids")
        assert hasattr(output, "object_idx")
        assert hasattr(output, "example_hidden")
        assert hasattr(output, "example_mask")

    def test_best_config_class_method(self):
        """Test RlArcWorldViTARCConfig.best_config() creates valid config."""
        config = RlArcWorldViTARCConfig.best_config(grid_size=10)

        assert config.grid_size == 10
        assert config.d_model == 128
        assert config.d_ff == 512
        assert config.num_layers == 4
        assert config.num_heads == 8
        assert config.ape_type == "SinusoidalAPE2D"
        assert config.rpe_type == "Two-slope-Alibi"
        assert config.use_ope is True
        assert config.ape_mixer == "weighted_sum_no_norm_vec"
        assert config.trainable_ape is False

    def test_forward_with_ope(self):
        """Test forward pass with OPE (Object Positional Encoding) enabled."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC, ViTARCEncoderOutput

        batch_size = 2
        grid_size = 8
        color_values = 11

        config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=True,  # OPE enabled
            use_examples=False,
            use_temporal=False,
            use_index_grid=False,
        )
        model = RlArcWorldViTARC(config)

        state = TensorDict(
            {
                StateKeys.GRID.value: torch.randint(
                    0, color_values, (batch_size, 1, grid_size, grid_size)
                ),
            }
        )

        output = model(state)

        assert isinstance(output, ViTARCEncoderOutput)
        assert output.hidden_states is not None
        assert output.object_idx is not None

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from rlarcworld.models.nn_modules import RlArcWorldViTARC

        batch_size = 2
        grid_size = 8
        color_values = 11

        config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
            use_examples=False,
            use_temporal=False,
            use_index_grid=False,
        )
        model = RlArcWorldViTARC(config)

        state = TensorDict(
            {
                StateKeys.GRID.value: torch.randint(
                    0, color_values, (batch_size, 1, grid_size, grid_size)
                ),
            }
        )

        output = model(state)

        # Compute dummy loss and backprop
        loss = output.hidden_states.mean()
        loss.backward()

        # Check gradients exist for trainable parameters
        has_grad = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients found in model parameters"


# ---------------- ViTARCActor/Critic Tests ----------------
class TestViTARCActor:
    def test_vitarc_actor_forward(self):
        """Test ViTARCActor forward pass produces valid action probabilities."""
        logger.info("Testing ViTARCActor forward pass")
        batch_size = 2
        grid_size = 16  # Reduced from 30 to avoid OOM on smaller GPUs
        color_values = 11
        num_examples = 3

        rl_vit_arc_config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=64,
            d_ff=128,
            num_layers=2,
            num_heads=4,
            use_ope=False,
        )
        # Create actor with minimal config for faster test
        network = ViTARCActor(
            color_values=color_values,
            rl_vit_arc=rl_vit_arc_config,
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
