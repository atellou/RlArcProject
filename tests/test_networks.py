import torch
from tensordict import TensorDict

from rlarcworld.agent.critic import ArcCriticNetwork

import unittest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArcCriticNetworkTest(unittest.TestCase):

    def test_train_arc_critic_network(self):
        """
        Test forward and backward pass of the ArcCriticNetwork class.
        """
        # Create an instance of the ArcCriticNetwork
        size = 10
        color_values = 10
        n_atoms = {"pixel_wise": 100, "binary": 3}
        network = ArcCriticNetwork(size, color_values, n_atoms)

        # Create dummy input and output tensors
        input_sample = TensorDict(
            {
                "last_grid": torch.randn(1, 1, size, size),
                "grid": torch.randn(1, 1, size, size),
                "examples": torch.randn(1, 10, size, size, size),
                "initial": torch.randn(1, 1, size, size),
                "index": torch.randn(1, 1, size, size),
                "actions": torch.randn(1, 4),
                "terminated": torch.randn(1, 1),
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
        output = network(input_sample)

        # Validate the output
        network.output_val(output)

        # Define a loss function and an optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Create dummy target tensors
        target = TensorDict(
            {"pixel_wise": torch.randn(1, 100), "binary": torch.randn(1, 3)}
        )

        # Calculate the loss
        loss = criterion(output["pixel_wise"], target["pixel_wise"]) + criterion(
            output["binary"], target["binary"]
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
