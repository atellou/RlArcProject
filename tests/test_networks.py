import torch
from tensordict import TensorDict

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork

import unittest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArcNetworksTest(unittest.TestCase):

    def test_train_arc_critic_network(self):
        """
        Test forward and backward pass of the ArcCriticNetwork class.
        """
        # Create an instance of the ArcCriticNetwork
        batch_size = torch.randint(1, 20, size=(1,))
        size = 30
        color_values = 11
        logger.info(
            "Testing ArcCriticNetwork with batch size: {}, size: {} and color values: {}".format(
                batch_size, size, color_values
            )
        )
        n_atoms = {"pixel_wise": torch.randint(50, 100, size=(1,)), "binary": 1}
        network = ArcCriticNetwork(size, color_values, n_atoms)

        # Create dummy input tensors
        input_sample = TensorDict(
            {
                "last_grid": torch.randn(batch_size, 1, size, size),
                "grid": torch.randn(batch_size, 1, size, size),
                "examples": torch.randn(batch_size, 10, 2, size, size),
                "initial": torch.randn(batch_size, 1, size, size),
                "index": torch.randn(batch_size, 1, size, size),
                "actions": torch.randn(batch_size, 1, 4),
                "terminated": torch.randn(batch_size, 1),
            }
        )

        # Validate the input
        network.input_val(input_sample)
        # Bad input
        for key in input_sample.keys():
            dc = input_sample.clone()
            with self.assertRaises(AssertionError):
                dc.pop(key)
                network.input_val(dc)

        # Forward pass
        org_sample = input_sample.clone()
        output = network(input_sample)

        # Validate not inplace changes to input
        torch.testing.assert_close(input_sample, org_sample)

        # Validate the output
        network.output_val(output)

        # Define a loss function and an optimizer
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Create dummy target tensors
        target = TensorDict(
            {
                "pixel_wise": torch.softmax(
                    torch.randn(batch_size, n_atoms["pixel_wise"]), dim=1
                ),
                "binary": torch.softmax(
                    torch.randn(batch_size, n_atoms["binary"]), dim=1
                ),
            }
        )

        # Assert probability mass function
        for key, dist in output.items():
            torch.testing.assert_close(
                torch.sum(dist, dim=1), torch.ones(batch_size)
            ), f"Probability mass function not normalized for key: {key}"
            assert torch.all(dist >= 0), f"Negative probability values for key: {key}"
            assert torch.all(
                dist <= 1
            ), f"Probability values greater than 1 for key: {key}"

        # Calculate the loss
        loss = criterion(output["pixel_wise"], target["pixel_wise"]) + criterion(
            output["binary"], target["binary"]
        )

        assert not torch.isnan(loss), "{} is NaN for Critic network".format(
            criterion._get_name()
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def test_train_arc_actor_network(self):
        """
        Test forward and backward pass of the ArcActorNetwork class.
        """
        # Create an instance of the ArcActorNetwork
        batch_size = torch.randint(1, 20, size=(1,))
        size = 30
        color_values = 11
        network = ArcActorNetwork(size, color_values)
        logger.info(
            "Testing ArcActorNetwork with batch size: {}, size: {} and color values: {}".format(
                batch_size, size, color_values
            )
        )

        # Create dummy input tensors
        input_sample = TensorDict(
            {
                "last_grid": torch.randn(batch_size, 1, size, size),
                "grid": torch.randn(batch_size, 1, size, size),
                "examples": torch.randn(batch_size, 10, 2, size, size),
                "initial": torch.randn(batch_size, 1, size, size),
                "index": torch.randn(batch_size, 1, size, size),
                "terminated": torch.randn(batch_size, 1),
            }
        )

        network.input_val(input_sample)
        # Bad input
        for key in input_sample.keys():
            dc = input_sample.clone()
            with self.assertRaises(AssertionError):
                dc.pop(key)
                network.input_val(dc)

        # Forward pass
        org_sample = input_sample.clone()
        output = network(input_sample)

        # Validate not inplace changes to input
        torch.testing.assert_close(input_sample, org_sample)

        # Define a loss function and an optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Create dummy target tensors
        target = TensorDict(
            {
                "x_location": torch.rand(size=(batch_size, size)),
                "y_location": torch.rand(size=(batch_size, size)),
                "color_values": torch.rand(size=(batch_size, color_values)),
                "submit": torch.rand(size=(batch_size, 2)),
            }
        )

        # Calculate the loss
        loss = sum([criterion(o, t) for o, t in zip(output.values(), target.values())])

        # Assert los is not NaN
        assert not torch.isnan(loss), "{} is NaN for Actor network".format(
            criterion._get_name()
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
