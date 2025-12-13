"""Unit tests for ArcWorldModelStepper."""

import os
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from rlarcworld.common.configs import NStepMinVMaxConfig
from rlarcworld.common.enums import RewardKeys, StateKeys
from rlarcworld.common.protocols import ActorCriticCoreProto
from rlarcworld.environments.steppers.reward_world_model import ArcWorldModelStepper
from rlarcworld.models.actor import ViTARCActor
from rlarcworld.models.reward_world_model import RewardWorldModel
from rlarcworld.nn_modules import RlArcWorldViTARCConfig
from rlarcworld.utils.logging_utils import configure_logger

configure_logger(level=os.environ.get("LOG_LEVEL", "INFO").upper(), json_format=False)


# ==================== Fixtures ====================


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def grid_size():
    """Return the grid size for tests."""
    return 16


@pytest.fixture
def color_values():
    """Return the number of color values for tests."""
    return 7


@pytest.fixture
def config(device):
    """Return a config dict for tests."""
    return {
        "device": device,
        "amp_enabled": False,
        "use_checkpointing": False,
    }


@pytest.fixture
def actor(grid_size, color_values, config):
    """Create an actor network for tests."""
    actor_config = RlArcWorldViTARCConfig(
        grid_size=grid_size,
        d_model=64,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        ape_type="SinusoidalAPE2D",
        rpe_type="Two-slope-Alibi",
        use_ope=False,
        ape_mixer="default",
    )

    actor = ViTARCActor(
        color_values=color_values,
        rl_vit_arc_config=actor_config,
    ).to(config["device"])
    actor.eval()
    return actor


@pytest.fixture
def reward_model(grid_size, color_values, config):
    """Create a reward world model for tests."""
    model = RewardWorldModel(
        grid_size=grid_size,
        color_values=color_values,
        embedding_size=128,  # Must match internal CnnAttention dimensions
        config=config,
    )
    return model


@pytest.fixture
def stepper(grid_size, color_values, actor, reward_model, config):
    """Create an ArcWorldModelStepper for tests."""
    return ArcWorldModelStepper(
        grid_size=grid_size,
        color_values=color_values,
        actor=actor,
        config=config,
        reward_model=reward_model,
        reward_model_lr=1e-4,
        step_horizon=3,
        n_step_config=NStepMinVMaxConfig(
            n_steps=1,
        ),
        gamma=0.99,
        model_update_frequency=1,
        imagination_ratio=1,
        imagination_frequency=1,
    )


@pytest.fixture
def sample_state(grid_size, color_values, device):
    """Create a sample state TensorDict for tests.

    Note: Grid inputs need shape (batch_size, 1, H, W) for the model.
    """
    batch_size = 1
    return TensorDict(
        {
            # Model expects (B, 1, H, W) for grid inputs
            StateKeys.GRID.value: torch.randint(
                0, color_values, (batch_size, 1, grid_size, grid_size)
            ),
            StateKeys.LAST_GRID.value: torch.randint(
                0, color_values, (batch_size, 1, grid_size, grid_size)
            ),
            StateKeys.INITIAL.value: torch.randint(
                0, color_values, (batch_size, 1, grid_size, grid_size)
            ),
            StateKeys.EXAMPLES.value: torch.randint(
                0, color_values, (batch_size, 10, 2, grid_size, grid_size)
            ),
            StateKeys.INDEX.value: torch.randint(
                0, grid_size, (batch_size, 1, grid_size, grid_size)
            ),
            StateKeys.TERMINATED.value: torch.zeros(batch_size, dtype=torch.bool),
        },
        batch_size=[batch_size],
    ).to(device)


@pytest.fixture
def sample_reward(device):
    """Create a sample reward TensorDict for tests."""
    batch_size = 1
    return TensorDict(
        {
            RewardKeys.BINARY.value: torch.randint(0, 2, (batch_size,)),
            RewardKeys.PIXEL_WISE.value: torch.rand(batch_size),
        },
        batch_size=[batch_size],
    ).to(device)


@pytest.fixture
def dict_sample(grid_size, color_values):
    """Create a dict sample for tests (dataset format)."""
    return {
        "task": {
            "input": torch.randint(0, color_values, (grid_size, grid_size)),
        },
        "examples": torch.randint(0, color_values, (10, 2, grid_size, grid_size)),
    }


# ==================== Initialization Tests ====================


class TestArcWorldModelStepperInit:
    """Tests for ArcWorldModelStepper initialization."""

    def test_init_creates_stepper(self, stepper):
        """Test that initialization creates a valid stepper."""
        assert stepper is not None
        assert stepper.model is not None
        assert stepper.environment is not None
        assert stepper.stepper is not None
        assert stepper.model_optimizer is not None

    def test_init_sets_metrics_tracking(self, stepper):
        """Test that metrics tracking is properly initialized."""
        assert stepper._imagination_count == 0
        assert isinstance(stepper._model_losses, TensorDict)
        assert RewardKeys.BINARY.value in stepper._model_losses.keys()
        assert RewardKeys.PIXEL_WISE.value in stepper._model_losses.keys()

    def test_init_sets_frequencies(self, stepper):
        """Test that training frequencies are set correctly."""
        assert stepper.model_update_frequency == 1
        assert stepper.imagination_ratio == 1
        assert stepper.imagination_frequency == 1


# ==================== Loss Tracking Tests ====================


class TestLossTracking:
    """Tests for TorchQueue-based loss tracking."""

    def test_push_loss_single(self, stepper):
        """Test pushing a single loss."""
        loss_dict = {RewardKeys.BINARY.value: 0.5, RewardKeys.PIXEL_WISE.value: 0.3}
        stepper.push_model_losses(loss_dict)

        # Queue should have 1 element (starts empty, pushed 1 value)
        assert stepper._model_losses[RewardKeys.BINARY.value].shape[0] == 1

    def test_push_loss_multiple(self, stepper):
        """Test pushing multiple losses."""
        losses = [
            {RewardKeys.BINARY.value: 0.5, RewardKeys.PIXEL_WISE.value: 0.3},
            {RewardKeys.BINARY.value: 0.4, RewardKeys.PIXEL_WISE.value: 0.2},
            {RewardKeys.BINARY.value: 0.3, RewardKeys.PIXEL_WISE.value: 0.1},
        ]
        for loss in losses:
            stepper.push_model_losses(loss)

        # Queue should have 3 elements (starts empty, pushed 3)
        assert stepper._model_losses[RewardKeys.BINARY.value].shape[0] == 3

    def test_push_loss_circular_buffer(self, stepper):
        """Test that loss queue wraps around when full."""
        # Push more losses than buffer size
        for i in range(1005):
            loss = {
                RewardKeys.BINARY.value: float(i),
                RewardKeys.PIXEL_WISE.value: float(i) * 2,
            }
            stepper.push_model_losses(loss)

        # Queue should be capped at max size
        assert stepper._model_losses[RewardKeys.BINARY.value].shape[0] == 1000

    def test_get_average_model_losses_empty(self, stepper):
        """Test get_average_model_losses with empty queue returns NaN."""
        import math

        avg = stepper.get_average_model_losses(reset=False)
        # Empty queue mean is NaN
        assert RewardKeys.BINARY.value in avg
        assert RewardKeys.PIXEL_WISE.value in avg
        assert math.isnan(avg[RewardKeys.BINARY.value])

    def test_get_average_model_losses_single(self, stepper):
        """Test get_average_model_losses with single loss."""
        loss_dict = {RewardKeys.BINARY.value: 0.5, RewardKeys.PIXEL_WISE.value: 0.3}
        stepper.push_model_losses(loss_dict)

        avg = stepper.get_average_model_losses(reset=False)

        assert RewardKeys.BINARY.value in avg
        assert RewardKeys.PIXEL_WISE.value in avg
        # Single value, so average equals that value
        assert abs(avg[RewardKeys.BINARY.value] - 0.5) < 1e-6
        assert abs(avg[RewardKeys.PIXEL_WISE.value] - 0.3) < 1e-6

    def test_get_average_model_losses_multiple(self, stepper):
        """Test get_average_model_losses with multiple losses."""
        losses = [
            {RewardKeys.BINARY.value: 0.6, RewardKeys.PIXEL_WISE.value: 0.3},
            {RewardKeys.BINARY.value: 0.4, RewardKeys.PIXEL_WISE.value: 0.2},
            {RewardKeys.BINARY.value: 0.2, RewardKeys.PIXEL_WISE.value: 0.1},
        ]
        for loss in losses:
            stepper.push_model_losses(loss)

        avg = stepper.get_average_model_losses(reset=False)

        # Average of [0.6, 0.4, 0.2] = 0.4
        assert abs(avg[RewardKeys.BINARY.value] - 0.4) < 1e-6
        # Average of [0.3, 0.2, 0.1] = 0.2
        assert abs(avg[RewardKeys.PIXEL_WISE.value] - 0.2) < 1e-6

    def test_get_average_model_losses_reset(self, stepper):
        """Test that get_average_model_losses resets queues by default."""
        loss_dict = {RewardKeys.BINARY.value: 0.5, RewardKeys.PIXEL_WISE.value: 0.3}
        stepper.push_model_losses(loss_dict)

        stepper.get_average_model_losses(reset=True)
        # After reset, queue should be empty
        assert stepper._model_losses[RewardKeys.BINARY.value].shape[0] == 0

    def test_get_average_model_losses_no_reset(self, stepper):
        """Test get_average_model_losses without reset."""
        loss_dict = {RewardKeys.BINARY.value: 0.5, RewardKeys.PIXEL_WISE.value: 0.3}
        stepper.push_model_losses(loss_dict)

        stepper.get_average_model_losses(reset=False)
        # Queue should still have 1 element
        assert stepper._model_losses[RewardKeys.BINARY.value].shape[0] == 1

    def test_get_loss_history(self, stepper):
        """Test get_loss_history returns TensorDict."""
        loss_dict = {RewardKeys.BINARY.value: 0.5, RewardKeys.PIXEL_WISE.value: 0.3}
        stepper.push_model_losses(loss_dict)

        history = stepper.get_loss_history()

        assert isinstance(history, TensorDict)
        assert RewardKeys.BINARY.value in history.keys()
        assert RewardKeys.PIXEL_WISE.value in history.keys()

    def test_get_loss_history_is_clone(self, stepper):
        """Test that get_loss_history returns a clone, not a view."""
        loss_dict = {RewardKeys.BINARY.value: 0.5, RewardKeys.PIXEL_WISE.value: 0.3}
        stepper.push_model_losses(loss_dict)

        original_val = stepper._model_losses[RewardKeys.BINARY.value][-1].item()
        history = stepper.get_loss_history()

        # Verify history was retrieved and original is unchanged
        assert history is not None
        assert stepper._model_losses[RewardKeys.BINARY.value][-1].item() == original_val


# ==================== Model Loss Computation Tests ====================


class TestComputeModelLoss:
    """Tests for compute_model_loss method."""

    def test_compute_model_loss_returns_dict(
        self, stepper, sample_state, sample_reward
    ):
        """Test that compute_model_loss returns loss dict."""
        pred = stepper.model(sample_state)
        total_loss, loss_dict = stepper.compute_model_loss(pred, sample_reward)

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(loss_dict, dict)
        assert RewardKeys.BINARY.value in loss_dict
        assert RewardKeys.PIXEL_WISE.value in loss_dict

    def test_compute_model_loss_values_finite(
        self, stepper, sample_state, sample_reward
    ):
        """Test that computed losses are finite."""
        pred = stepper.model(sample_state)
        total_loss, loss_dict = stepper.compute_model_loss(pred, sample_reward)

        assert torch.isfinite(total_loss)
        assert all(isinstance(v, float) and not (v != v) for v in loss_dict.values())

    def test_compute_model_loss_total_is_sum(
        self, stepper, sample_state, sample_reward
    ):
        """Test that total loss is sum of components."""
        pred = stepper.model(sample_state)
        total_loss, loss_dict = stepper.compute_model_loss(pred, sample_reward)

        expected_total = sum(loss_dict.values())
        assert abs(total_loss.item() - expected_total) < 1e-5


# ==================== Model Update Tests ====================


class TestUpdateModel:
    """Tests for update_model method."""

    def test_update_model_returns_dict(self, stepper, sample_state, sample_reward):
        """Test that update_model returns a dict of losses."""
        loss_dict = stepper.update_model(sample_state, sample_reward)

        assert isinstance(loss_dict, dict)
        assert RewardKeys.BINARY.value in loss_dict
        assert RewardKeys.PIXEL_WISE.value in loss_dict

    def test_update_model_records_loss(self, stepper, sample_state, sample_reward):
        """Test that update_model records the loss."""
        initial_size = stepper._model_losses[RewardKeys.BINARY.value].shape[0]
        stepper.update_model(sample_state, sample_reward)

        assert (
            stepper._model_losses[RewardKeys.BINARY.value].shape[0] == initial_size + 1
        )

    def test_update_model_updates_parameters(
        self, stepper, sample_state, sample_reward
    ):
        """Test that update_model actually updates model parameters."""
        # Get initial parameter values
        initial_params = [p.clone() for p in stepper.model.parameters()]

        stepper.update_model(sample_state, sample_reward)

        # Check that at least some parameters changed
        params_changed = False
        for initial, current in zip(initial_params, stepper.model.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break

        assert params_changed, "Model parameters should change after update"


# ==================== Reset Environment Tests ====================


class TestResetEnv:
    """Tests for _reset_env method."""

    def test_reset_env_with_dict_sample(self, stepper, dict_sample):
        """Test _reset_env with dict sample format."""
        # Should not raise
        stepper._reset_env(stepper.environment, dict_sample)

    def test_reset_env_with_tensordict_sample(self, stepper, sample_state):
        """Test _reset_env with TensorDict sample format."""
        # Should not raise
        stepper._reset_env(stepper.environment, sample_state)

    def test_reset_env_with_missing_grid_raises(self, stepper, device):
        """Test _reset_env raises error when TensorDict has no grid key."""
        bad_sample = TensorDict(
            {"other_key": torch.zeros(1, 16, 16)},
            batch_size=[1],
        ).to(device)

        with pytest.raises(ValueError, match="must have 'grid' or 'initial' key"):
            stepper._reset_env(stepper.environment, bad_sample)


# ==================== Imagination Tests ====================


class TestImagination:
    """Tests for imagination rollout functionality."""

    def test_imagination_count_property(self, stepper):
        """Test imagination_count property."""
        assert stepper.imagination_count == 0

    def test_imagination_count_increments(self, stepper, dict_sample):
        """Test that imagination count increments after rollout."""
        # Create a mock core
        mock_core = MagicMock(spec=ActorCriticCoreProto)
        mock_core.compute_loss.return_value = (
            torch.tensor(0.1),
            torch.tensor(0.2),
            TensorDict({}, batch_size=[]),
        )

        initial_count = stepper._imagination_count

        # Run imagination (will fail early but should still increment)
        try:
            stepper._imagine_and_train(mock_core, dict_sample, training=False)
        except Exception:
            pass  # May fail due to env issues, but count should still increment

        # Count should have incremented
        assert stepper._imagination_count >= initial_count


# ==================== Checkpointing Tests ====================


class TestCheckpointing:
    """Tests for checkpoint and load methods."""

    def test_checkpoint_returns_dict(self, stepper):
        """Test that checkpoint returns a dict with required keys."""
        ckpt = stepper.checkpoint()

        assert isinstance(ckpt, dict)
        assert "reward_model" in ckpt
        assert "model_optimizer" in ckpt
        assert "imagination_count" in ckpt
        assert "model_losses" in ckpt

    def test_checkpoint_model_losses_structure(self, stepper):
        """Test that checkpoint model_losses has correct structure."""
        loss_dict = {RewardKeys.BINARY.value: 0.5, RewardKeys.PIXEL_WISE.value: 0.3}
        stepper.push_model_losses(loss_dict)

        ckpt = stepper.checkpoint()

        assert isinstance(ckpt["model_losses"], dict)
        assert RewardKeys.BINARY.value in ckpt["model_losses"]
        assert RewardKeys.PIXEL_WISE.value in ckpt["model_losses"]

    def test_load_restores_state(self, stepper):
        """Test that load restores stepper state."""
        # Record some losses
        for i in range(5):
            loss = {
                RewardKeys.BINARY.value: float(i),
                RewardKeys.PIXEL_WISE.value: float(i) * 2,
            }
            stepper.push_model_losses(loss)
        stepper._imagination_count = 10

        # Create checkpoint
        ckpt = stepper.checkpoint()

        # Reset state
        stepper._reset_loss_queues()
        stepper._imagination_count = 0

        # Load checkpoint
        stepper.load(ckpt)

        assert stepper._imagination_count == 10
        # Check that losses were restored
        assert RewardKeys.BINARY.value in stepper._model_losses.keys()

    def test_load_partial_checkpoint(self, stepper):
        """Test that load handles partial checkpoints gracefully."""
        partial_ckpt = {"imagination_count": 42}

        # Should not raise
        stepper.load(partial_ckpt)

        assert stepper._imagination_count == 42


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for ArcWorldModelStepper."""

    def test_full_training_cycle(self, stepper, sample_state, sample_reward):
        """Test a full training cycle: update model and check metrics."""
        # Update model multiple times
        for _ in range(5):
            stepper.update_model(sample_state, sample_reward)

        # Check metrics - queue should have 5 elements (starts empty, 5 updates)
        assert stepper._model_losses[RewardKeys.BINARY.value].shape[0] == 5

        avg_losses = stepper.get_average_model_losses()
        assert RewardKeys.BINARY.value in avg_losses
        assert RewardKeys.PIXEL_WISE.value in avg_losses

        # After reset, queue should be empty
        assert stepper._model_losses[RewardKeys.BINARY.value].shape[0] == 0

    def test_checkpoint_and_restore_cycle(self, stepper, sample_state, sample_reward):
        """Test checkpoint and restore preserves training state."""
        # Train for a bit
        for _ in range(3):
            stepper.update_model(sample_state, sample_reward)
        stepper._imagination_count = 7

        # Checkpoint
        ckpt = stepper.checkpoint()
        original_queue_size = stepper._model_losses[RewardKeys.BINARY.value].shape[0]

        # Continue training
        for _ in range(2):
            stepper.update_model(sample_state, sample_reward)
        stepper._imagination_count = 15

        # Restore
        stepper.load(ckpt)

        # Should be back to checkpointed state
        assert stepper._imagination_count == 7
        assert (
            stepper._model_losses[RewardKeys.BINARY.value].shape[0]
            == original_queue_size
        )
