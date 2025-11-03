"""Unit tests for TensorBoard logging module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict

from rlarcworld.logging.tensorboard import (
    TensorBoardWriterManager,
    make_tensorboard_callbacks,
)
from rlarcworld.runners.core import OnStepContext


@pytest.fixture
def temp_experiment_path(tmp_path):
    """Temporary directory for TensorBoard logs."""
    return tmp_path / "test_experiment"


@pytest.fixture
def mock_writer():
    """Mock SummaryWriter to avoid actual file I/O."""
    with patch("rlarcworld.logging.tensorboard.SummaryWriter") as mock:
        writer_instance = MagicMock()
        mock.return_value = writer_instance
        yield writer_instance


@pytest.fixture
def mock_core():
    """Mock core with actor/critic optimizers and hyperparameters."""
    core = MagicMock()

    # Mock optimizers with learning rates
    core.actor_optimizer.param_groups = [{"lr": 0.001}]
    core.critic_optimizer.param_groups = [{"lr": 0.003}]

    # Mock entropy coefficient
    core.entropy_coef = 0.01

    # Mock actor and critic for weight norms
    core.actor.parameters.return_value = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0]),
    ]
    core.critic.parameters.return_value = [
        torch.tensor([2.0, 3.0]),
        torch.tensor([1.0, 1.0, 1.0]),
    ]

    # Mock critic z_atoms for Q estimation
    core.critic.z_atoms = {
        "pixel_wise": torch.linspace(-10, 10, 51),
        "binary": torch.linspace(-1, 1, 3),
    }

    return core


@pytest.fixture
def mock_replay_buffer():
    """Mock replay buffer with storage and sampler."""
    buffer = MagicMock()
    buffer.storage.__len__.return_value = 250
    buffer.storage.max_size = 1000
    buffer.sampler.alpha = 0.6
    buffer.sampler.beta = 0.4
    return buffer


@pytest.fixture
def sample_context(mock_core):
    """Sample OnStepContext for testing."""
    batch = TensorDict(
        {
            "state": TensorDict(
                {
                    "grid": torch.randn(4, 1, 16, 16),
                    "examples": torch.randn(4, 10, 2, 16, 16),
                    "terminated": torch.zeros(4, 1),
                },
                batch_size=[4],
            ),
            "actions": TensorDict(
                {
                    "x_location": torch.randn(4, 16),
                    "y_location": torch.randn(4, 16),
                    "color_values": torch.randn(4, 11),
                },
                batch_size=[4],
            ),
            "reward": TensorDict(
                {
                    "pixel_wise": torch.tensor([0.5, 0.3, 0.8, 0.2]),
                    "binary": torch.tensor([1.0, 0.0, 1.0, 0.0]),
                },
                batch_size=[4],
            ),
            "next_state": TensorDict(
                {
                    "grid": torch.randn(4, 1, 16, 16),
                    "examples": torch.randn(4, 10, 2, 16, 16),
                    "terminated": torch.zeros(4, 1),
                },
                batch_size=[4],
            ),
            "terminated": torch.zeros(4, 1),
        },
        batch_size=[4],
    )

    td_error = TensorDict(
        {
            "pixel_wise": torch.tensor([0.5, 0.3, 0.8, 0.2]),
            "binary": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        },
        batch_size=[4],
    )

    return OnStepContext(
        epoch=5,
        step_idx=100,
        iteration=500,
        batch=batch,
        loss_actor=torch.tensor(0.25),
        loss_critic=torch.tensor(0.42),
        td_error=td_error,
        core=mock_core,
        hyperparameters={"gamma": 0.99, "tau": 0.005},
    )


class TestTensorBoardWriterManager:
    """Test TensorBoardWriterManager initialization and lifecycle."""

    def test_init_creates_directories(self, temp_experiment_path, mock_writer):
        """Test that manager creates experiment and tensorboard directories."""
        manager = TensorBoardWriterManager(temp_experiment_path)

        assert temp_experiment_path.exists()
        assert (temp_experiment_path / "tensorboard").exists()
        assert manager.writer is not None

    def test_init_with_custom_logdir_name(self, temp_experiment_path, mock_writer):
        """Test custom logdir name."""
        TensorBoardWriterManager(temp_experiment_path, logdir_name="custom_logs")

        assert (temp_experiment_path / "custom_logs").exists()

    def test_init_with_replay_buffer(
        self, temp_experiment_path, mock_writer, mock_replay_buffer
    ):
        """Test initialization with replay buffer."""
        manager = TensorBoardWriterManager(
            temp_experiment_path, replay_buffer=mock_replay_buffer
        )

        assert manager.replay_buffer is mock_replay_buffer

    def test_init_with_medium_frequency(self, temp_experiment_path, mock_writer):
        """Test custom medium frequency."""
        manager = TensorBoardWriterManager(temp_experiment_path, medium_frequency=200)

        assert manager.medium_frequency == 200

    def test_close_flushes_and_closes_writer(self, temp_experiment_path, mock_writer):
        """Test that close() flushes and closes the writer."""
        manager = TensorBoardWriterManager(temp_experiment_path)
        manager.close()

        mock_writer.flush.assert_called_once()
        mock_writer.close.assert_called_once()

    def test_close_handles_exceptions_gracefully(
        self, temp_experiment_path, mock_writer
    ):
        """Test that close() handles exceptions without raising."""
        manager = TensorBoardWriterManager(temp_experiment_path)
        mock_writer.flush.side_effect = RuntimeError("Flush failed")

        # Should not raise
        manager.close()


class TestMinimalMetricsLogging:
    """Test minimal metrics logged every step."""

    def test_log_losses(self, temp_experiment_path, mock_writer, sample_context):
        """Test that actor and critic losses are logged."""
        manager = TensorBoardWriterManager(temp_experiment_path)
        manager.on_train_step(sample_context)

        # Check actor loss
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "loss/actor" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][0] == "train/loss/actor"
        assert abs(calls[0][0][1] - 0.25) < 1e-6

        # Check critic loss
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "loss/critic" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][0] == "train/loss/critic"
        assert abs(calls[0][0][1] - 0.42) < 1e-6

    def test_log_learning_rates(
        self, temp_experiment_path, mock_writer, sample_context
    ):
        """Test that optimizer learning rates are logged."""
        manager = TensorBoardWriterManager(temp_experiment_path)
        manager.on_train_step(sample_context)

        # Check actor LR
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "optim/actor_lr" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][0] == "train/optim/actor_lr"
        assert abs(calls[0][0][1] - 0.001) < 1e-6

        # Check critic LR
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "optim/critic_lr" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][0] == "train/optim/critic_lr"
        assert abs(calls[0][0][1] - 0.003) < 1e-6

    def test_log_entropy_coef(self, temp_experiment_path, mock_writer, sample_context):
        """Test that entropy coefficient is logged."""
        manager = TensorBoardWriterManager(temp_experiment_path)
        manager.on_train_step(sample_context)

        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "algo/entropy_coef" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][0] == "train/algo/entropy_coef"
        assert abs(calls[0][0][1] - 0.01) < 1e-6

    def test_log_rewards_by_key(
        self, temp_experiment_path, mock_writer, sample_context
    ):
        """Test that rewards are logged per key with batch average."""
        manager = TensorBoardWriterManager(temp_experiment_path)
        manager.on_train_step(sample_context)

        # Check pixel_wise reward (mean of [0.5, 0.3, 0.8, 0.2] = 0.45)
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "reward/pixel_wise" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][0] == "train/reward/pixel_wise"
        assert abs(calls[0][0][1] - 0.45) < 1e-6

        # Check binary reward (mean of [1.0, 0.0, 1.0, 0.0] = 0.5)
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "reward/binary" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][0] == "train/reward/binary"
        assert abs(calls[0][0][1] - 0.5) < 1e-6

    def test_log_replay_buffer_stats(
        self, temp_experiment_path, mock_writer, sample_context, mock_replay_buffer
    ):
        """Test that replay buffer stats are logged."""
        manager = TensorBoardWriterManager(
            temp_experiment_path, replay_buffer=mock_replay_buffer
        )
        manager.on_train_step(sample_context)

        # Check size
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "replay/size" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][1] == 250

        # Check capacity
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "replay/capacity" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][1] == 1000

        # Check warmup progress (250/1000 = 0.25)
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "replay/warmup_progress" in str(call)
        ]
        assert len(calls) == 1
        assert abs(calls[0][0][1] - 0.25) < 1e-6

        # Check alpha and beta
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "replay/alpha" in str(call)
        ]
        assert len(calls) == 1
        assert abs(calls[0][0][1] - 0.6) < 1e-6

        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "replay/beta" in str(call)
        ]
        assert len(calls) == 1
        assert abs(calls[0][0][1] - 0.4) < 1e-6

    def test_validation_phase_uses_val_prefix(
        self, temp_experiment_path, mock_writer, sample_context
    ):
        """Test that validation metrics use 'val/' prefix."""
        manager = TensorBoardWriterManager(temp_experiment_path)
        manager.on_validation_step(sample_context)

        # Check that all scalars use 'val/' prefix
        for call in mock_writer.add_scalar.call_args_list:
            metric_name = call[0][0]
            assert metric_name.startswith(
                "val/"
            ), f"Expected 'val/' prefix, got {metric_name}"

    def test_handles_none_losses_gracefully(
        self, temp_experiment_path, mock_writer, mock_core
    ):
        """Test that None losses don't cause errors."""
        ctx = OnStepContext(
            epoch=1,
            step_idx=10,
            iteration=10,
            batch=None,
            loss_actor=None,
            loss_critic=None,
            td_error=None,
            core=mock_core,
            hyperparameters={},
        )

        manager = TensorBoardWriterManager(temp_experiment_path)
        # Should not raise
        manager.on_train_step(ctx)

    def test_handles_none_batch_gracefully(
        self, temp_experiment_path, mock_writer, mock_core
    ):
        """Test that None batch doesn't cause errors."""
        ctx = OnStepContext(
            epoch=1,
            step_idx=10,
            iteration=10,
            batch=None,
            loss_actor=torch.tensor(0.1),
            loss_critic=torch.tensor(0.2),
            td_error=None,
            core=mock_core,
            hyperparameters={},
        )

        manager = TensorBoardWriterManager(temp_experiment_path)
        # Should not raise
        manager.on_train_step(ctx)

    def test_periodic_flush_every_100_steps(
        self, temp_experiment_path, mock_writer, sample_context
    ):
        """Test that writer is flushed every 100 steps."""
        manager = TensorBoardWriterManager(temp_experiment_path)

        # iteration=500 is divisible by 100
        manager.on_train_step(sample_context)
        assert mock_writer.flush.call_count >= 1

        # Reset and test non-divisible iteration
        mock_writer.reset_mock()
        sample_context.iteration = 501
        manager.on_train_step(sample_context)
        # Should not flush on non-100 step
        assert mock_writer.flush.call_count == 0


class TestMediumMetricsLogging:
    """Test medium-frequency metrics (policy entropy, Q estimates, weight norms)."""

    def test_policy_entropy_logged_at_medium_frequency(
        self, temp_experiment_path, mock_writer, sample_context, mock_core
    ):
        """Test that policy entropy is logged at medium frequency."""
        # Mock actor forward to return action probabilities
        action_probs = TensorDict(
            {
                "x_location": torch.softmax(torch.randn(4, 16), dim=-1),
                "y_location": torch.softmax(torch.randn(4, 16), dim=-1),
                "color_values": torch.softmax(torch.randn(4, 11), dim=-1),
            },
            batch_size=[4],
        )
        mock_core.actor.return_value = action_probs

        manager = TensorBoardWriterManager(temp_experiment_path, medium_frequency=100)

        # iteration=500 is divisible by 100
        manager.on_train_step(sample_context)

        # Check entropy was logged
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "policy/entropy" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][0] == "train/policy/entropy"
        # Entropy should be positive
        assert calls[0][0][1] > 0

    def test_policy_entropy_not_logged_off_frequency(
        self, temp_experiment_path, mock_writer, sample_context, mock_core
    ):
        """Test that policy entropy is NOT logged when off frequency."""
        manager = TensorBoardWriterManager(temp_experiment_path, medium_frequency=100)

        # iteration=501 is NOT divisible by 100
        sample_context.iteration = 501
        manager.on_train_step(sample_context)

        # Check entropy was NOT logged
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "policy/entropy" in str(call)
        ]
        assert len(calls) == 0

    def test_critic_q_expected_logged_per_head(
        self, temp_experiment_path, mock_writer, sample_context, mock_core
    ):
        """Test that expected Q is logged per reward head."""
        # Mock actor and critic forwards
        action_probs = TensorDict(
            {
                "x_location": torch.softmax(torch.randn(4, 16), dim=-1),
                "y_location": torch.softmax(torch.randn(4, 16), dim=-1),
                "color_values": torch.softmax(torch.randn(4, 11), dim=-1),
            },
            batch_size=[4],
        )
        critic_probs = TensorDict(
            {
                "pixel_wise": torch.softmax(torch.randn(4, 51), dim=-1),
                "binary": torch.softmax(torch.randn(4, 3), dim=-1),
            },
            batch_size=[4],
        )
        mock_core.actor.return_value = action_probs
        mock_core.critic.return_value = critic_probs

        manager = TensorBoardWriterManager(temp_experiment_path, medium_frequency=100)
        manager.on_train_step(sample_context)

        # Check pixel_wise Q
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "critic/q_expected_mean/pixel_wise" in str(call)
        ]
        assert len(calls) == 1

        # Check binary Q
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "critic/q_expected_mean/binary" in str(call)
        ]
        assert len(calls) == 1

    def test_weight_norms_logged(
        self, temp_experiment_path, mock_writer, sample_context, mock_core
    ):
        """Test that actor and critic weight L2 norms are logged."""
        # Mock actor forward for entropy computation
        action_probs = TensorDict(
            {
                "x_location": torch.softmax(torch.randn(4, 16), dim=-1),
            },
            batch_size=[4],
        )
        mock_core.actor.return_value = action_probs

        manager = TensorBoardWriterManager(temp_experiment_path, medium_frequency=100)
        manager.on_train_step(sample_context)

        # Check actor weight norm
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "actor/weight_l2_norm" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][1] > 0  # Should be positive

        # Check critic weight norm
        calls = [
            call
            for call in mock_writer.add_scalar.call_args_list
            if "critic/weight_l2_norm" in str(call)
        ]
        assert len(calls) == 1
        assert calls[0][0][1] > 0  # Should be positive


class TestHyperparametersLogging:
    """Test hyperparameters logging."""

    def test_log_hparams(self, temp_experiment_path, mock_writer):
        """Test that hyperparameters are logged."""
        manager = TensorBoardWriterManager(temp_experiment_path)

        hparams = {
            "grid_size": 16,
            "batch_size": 32,
            "actor_lr": 0.001,
            "critic_lr": 0.003,
        }
        metrics = {
            "final_loss_actor": 0.15,
            "final_loss_critic": 0.22,
        }

        manager.log_hparams(hparams, metrics)

        mock_writer.add_hparams.assert_called_once_with(hparams, metrics)

    def test_log_hparams_with_no_metrics(self, temp_experiment_path, mock_writer):
        """Test logging hparams without metrics."""
        manager = TensorBoardWriterManager(temp_experiment_path)

        hparams = {"grid_size": 16}
        manager.log_hparams(hparams)

        mock_writer.add_hparams.assert_called_once_with(hparams, {})

    def test_log_hparams_handles_exception(self, temp_experiment_path, mock_writer):
        """Test that hparams logging handles exceptions gracefully."""
        manager = TensorBoardWriterManager(temp_experiment_path)
        mock_writer.add_hparams.side_effect = RuntimeError("HParams plugin unavailable")

        # Should not raise
        manager.log_hparams({"test": 1})


class TestCallbackFactory:
    """Test make_tensorboard_callbacks factory function."""

    def test_factory_returns_manager_and_callbacks(
        self, temp_experiment_path, mock_writer
    ):
        """Test that factory returns manager and two callbacks."""
        manager, train_cb, val_cb = make_tensorboard_callbacks(temp_experiment_path)

        assert isinstance(manager, TensorBoardWriterManager)
        assert callable(train_cb)
        assert callable(val_cb)

    def test_factory_with_replay_buffer(
        self, temp_experiment_path, mock_writer, mock_replay_buffer
    ):
        """Test factory with replay buffer."""
        manager, _, _ = make_tensorboard_callbacks(
            temp_experiment_path, replay_buffer=mock_replay_buffer
        )

        assert manager.replay_buffer is mock_replay_buffer

    def test_factory_with_custom_frequencies(self, temp_experiment_path, mock_writer):
        """Test factory with custom frequencies."""
        manager, _, _ = make_tensorboard_callbacks(
            temp_experiment_path,
            warmup_buffer_ratio=0.3,
            medium_frequency=200,
        )

        assert manager.warmup_buffer_ratio == 0.3
        assert manager.medium_frequency == 200

    def test_callbacks_are_bound_methods(
        self, temp_experiment_path, mock_writer, sample_context
    ):
        """Test that returned callbacks are properly bound to manager."""
        manager, train_cb, val_cb = make_tensorboard_callbacks(temp_experiment_path)

        # Should not raise
        train_cb(sample_context)
        val_cb(sample_context)

        # Verify writer was called
        assert mock_writer.add_scalar.call_count > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_missing_reward_key_gracefully(
        self, temp_experiment_path, mock_writer, sample_context
    ):
        """Test that missing reward keys don't cause errors."""
        # Remove reward from batch
        sample_context.batch = TensorDict(
            {
                "state": sample_context.batch["state"],
                "actions": sample_context.batch["actions"],
            },
            batch_size=[4],
        )

        manager = TensorBoardWriterManager(temp_experiment_path)
        # Should not raise
        manager.on_train_step(sample_context)

    def test_handles_malformed_reward_gracefully(
        self, temp_experiment_path, mock_writer, sample_context
    ):
        """Test that malformed rewards don't cause errors."""
        # Replace reward with non-dict
        sample_context.batch["reward"] = torch.tensor([1.0, 2.0, 3.0, 4.0])

        manager = TensorBoardWriterManager(temp_experiment_path)
        # Should not raise
        manager.on_train_step(sample_context)

    def test_handles_missing_optimizer_gracefully(
        self, temp_experiment_path, mock_writer, sample_context
    ):
        """Test that missing optimizers don't cause errors."""
        # Remove optimizers from core
        del sample_context.core.actor_optimizer
        del sample_context.core.critic_optimizer

        manager = TensorBoardWriterManager(temp_experiment_path)
        # Should not raise
        manager.on_train_step(sample_context)

    def test_handles_missing_entropy_coef_gracefully(
        self, temp_experiment_path, mock_writer, sample_context
    ):
        """Test that missing entropy_coef doesn't cause errors."""
        del sample_context.core.entropy_coef

        manager = TensorBoardWriterManager(temp_experiment_path)
        # Should not raise
        manager.on_train_step(sample_context)

    def test_step_zero_uses_step_one(
        self, temp_experiment_path, mock_writer, sample_context
    ):
        """Test that step 0 is mapped to step 1 for nicer charts."""
        sample_context.iteration = 0

        manager = TensorBoardWriterManager(temp_experiment_path)
        manager.on_train_step(sample_context)

        # All scalar calls should use step=1
        for call in mock_writer.add_scalar.call_args_list:
            step = call[0][2]
            assert step == 1, f"Expected step=1 for iteration=0, got {step}"
