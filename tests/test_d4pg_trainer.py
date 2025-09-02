"""Unit tests for D4PG trainer functionality."""

import unittest
from unittest.mock import MagicMock, patch
import torch
from tensordict import TensorDict

from rlarcworld.algorithms.d4pg_trainer import D4PGTrainer


class TestD4PGTrainer(unittest.TestCase):
    def setUp(self):
        # Create a mock D4PGCore with parameters property
        self.mock_core = MagicMock()
        # Mock parameters to return a tensor with device
        mock_params = MagicMock()
        mock_params.return_value = [torch.tensor([0.0], device="cpu")]
        type(self.mock_core).parameters = mock_params

        # Default trainer config
        self.trainer_kwargs = {
            "core": self.mock_core,
            "step_stream_factory": lambda x: iter([]),
            "replay_buffer": None,
            "batch_size": 32,
            "device": "cpu",
        }

    def create_step_state(self, batch_size=1):
        """Create a sample step state TensorDict."""
        return TensorDict(
            {
                "state": {
                    "observation": torch.randn(batch_size, 4),
                    "terminated": torch.zeros(batch_size, dtype=torch.bool),
                },
                "actions": torch.randn(batch_size, 2),
                "reward": torch.randn(batch_size, 1),
                "next_state": {
                    "observation": torch.randn(batch_size, 4),
                    "terminated": torch.zeros(batch_size, dtype=torch.bool),
                },
                "terminated": torch.zeros(batch_size, dtype=torch.bool),
                "truncated": torch.zeros(batch_size, dtype=torch.bool),
            },
            batch_size=[batch_size],
        )

    def test_no_replay_buffer_returns_batch_directly(self):
        """Test that without replay buffer, it returns the batch directly."""
        # Create a copy of trainer kwargs without replay_buffer
        kwargs = self.trainer_kwargs.copy()
        kwargs["replay_buffer"] = None
        trainer = D4PGTrainer(**kwargs)
        step_state = self.create_step_state()

        result = trainer._maybe_store_and_sample(step_state)

        self.assertIsNotNone(result)
        self.assertEqual(result.batch_size, torch.Size([1, *step_state.batch_size]))

    @patch("torch.full")
    def test_with_replay_buffer_not_warmed_up(self, mock_full):
        """Test when replay buffer exists but not enough samples yet."""
        mock_buffer = MagicMock()
        mock_buffer.storage = MagicMock()
        mock_buffer.storage.max_size = 1000
        mock_buffer.storage.__len__.return_value = 100  # 10% of capacity

        # Create a copy of trainer kwargs and update with test-specific values
        kwargs = self.trainer_kwargs.copy()
        kwargs.update(
            {
                "replay_buffer": mock_buffer,
                "warmup_buffer_ratio": 0.2,  # Need 200 samples to warm up
            }
        )
        trainer = D4PGTrainer(**kwargs)

        step_state = self.create_step_state()
        result = trainer._maybe_store_and_sample(step_state)

        self.assertIsNone(result)
        mock_buffer.extend.assert_called_once()
        mock_buffer.sample.assert_not_called()

    @patch("torch.full")
    def test_with_replay_buffer_warmed_up(self, mock_full):
        """Test when replay buffer is warmed up and ready to sample."""
        mock_buffer = MagicMock()
        mock_buffer.storage = MagicMock()
        mock_buffer.storage.max_size = 1000
        mock_buffer.storage.__len__.return_value = 300  # 30% of capacity

        # Mock the sample to return a specific value
        mock_sample = self.create_step_state(batch_size=32)
        mock_buffer.sample.return_value = mock_sample

        # Create a copy of trainer kwargs and update with test-specific values
        kwargs = self.trainer_kwargs.copy()
        kwargs.update(
            {
                "replay_buffer": mock_buffer,
                "warmup_buffer_ratio": 0.2,  # 200 samples needed
            }
        )
        trainer = D4PGTrainer(**kwargs)

        step_state = self.create_step_state()
        result = trainer._maybe_store_and_sample(step_state)

        self.assertIsNotNone(result)
        mock_buffer.extend.assert_called_once()
        mock_buffer.sample.assert_called_once_with(32)

    @patch("torch.full")
    def test_per_priority_update(self, mock_full):
        """Test priority updates for prioritized experience replay."""
        mock_sampler = MagicMock()
        mock_sampler._max_priority = 2.5

        mock_buffer = MagicMock()
        mock_buffer.sampler = mock_sampler
        mock_buffer.storage = MagicMock()
        mock_buffer.storage.max_size = 1000
        mock_buffer.storage.__len__.return_value = 500

        # Configure mock to return indices when extend is called
        mock_buffer.extend.return_value = [0]  # Simulate returning an index

        # Create a copy of trainer kwargs and update with test-specific values
        kwargs = self.trainer_kwargs.copy()
        kwargs["replay_buffer"] = mock_buffer
        trainer = D4PGTrainer(**kwargs)

        step_state = self.create_step_state()
        mock_full.return_value = torch.tensor([2.5])
        trainer._maybe_store_and_sample(step_state)

        # Check if priority was updated correctly
        mock_buffer.update_priority.assert_called_once()
        args, _ = mock_buffer.update_priority.call_args
        self.assertTrue(torch.equal(args[1], torch.tensor([2.5])))

    def test_tensor_shapes_in_maybe_store_and_sample(self):
        """Test that tensor shapes are preserved in _maybe_store_and_sample."""
        # Test with batch size > 1
        batch_size = 3
        step_state = self.create_step_state(batch_size=batch_size)

        # Test with replay buffer (warmed up)
        mock_buffer = MagicMock()
        mock_buffer.storage = MagicMock()
        mock_buffer.storage.max_size = 1000
        mock_buffer.storage.__len__.return_value = 500  # Enough samples to be warmed up
        mock_buffer.sample.return_value = (
            step_state  # Return the same state for testing
        )

        # Mock the priority tensor creation to avoid recursion
        with patch("torch.full") as mock_full:
            # Return a simple tensor with the correct shape
            def create_mock_tensor(size, fill_value, **kwargs):
                return torch.ones(size, **kwargs) * fill_value

            mock_full.side_effect = create_mock_tensor

            kwargs = self.trainer_kwargs.copy()
            kwargs.update({"replay_buffer": mock_buffer, "batch_size": batch_size})
            trainer = D4PGTrainer(**kwargs)

            # Test with replay buffer (warmed up)
            result = trainer._maybe_store_and_sample(step_state)

            # Check shapes of the returned batch
            self.assertEqual(result["state"]["observation"].shape, (batch_size, 4))
            self.assertEqual(result["actions"].shape, (batch_size, 2))
            self.assertEqual(result["reward"].shape, (batch_size, 1))
            self.assertEqual(result["next_state"]["observation"].shape, (batch_size, 4))
            self.assertEqual(result["terminated"].shape, (batch_size,))
            self.assertEqual(result["truncated"].shape, (batch_size,))

        # Test without replay buffer
        with patch("torch.full") as mock_full:
            # Return a simple tensor with the correct shape
            def create_mock_tensor(size, fill_value, **kwargs):
                return torch.ones(size, **kwargs) * fill_value

            mock_full.side_effect = create_mock_tensor

            kwargs = self.trainer_kwargs.copy()
            kwargs.update({"replay_buffer": None, "batch_size": batch_size})
            trainer = D4PGTrainer(**kwargs)
            result = trainer._maybe_store_and_sample(step_state)

            # Should add a batch dimension when no replay buffer
            self.assertEqual(result["state"]["observation"].shape, (1, batch_size, 4))
            self.assertEqual(result["actions"].shape, (1, batch_size, 2))
            self.assertEqual(result["reward"].shape, (1, batch_size, 1))
            self.assertEqual(
                result["next_state"]["observation"].shape, (1, batch_size, 4)
            )
            self.assertEqual(result["terminated"].shape, (1, batch_size))
            self.assertEqual(result["truncated"].shape, (1, batch_size))


if __name__ == "__main__":
    unittest.main()
