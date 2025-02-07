import os
import torch
from tensordict import TensorDict

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork

import unittest
import logging


logger = logging.getLogger(__name__)


class ArcNetworksTest(unittest.TestCase):

    def setUp(self):
        self.batch_size = torch.randint(1, 20, size=(1,))

    def test_train_arc_actor_network(self):
        losses = []
        for i in range(10):
            losses.append(self.train_arc_actor_network())
        # Validate that at least 50% of the values in the list are not close or equal
        self.assertGreaterEqual(
            sum(
                [
                    not torch.isclose(losses[i], losses[i + 1])
                    for i in range(len(losses) - 1)
                ]
            )
            / (len(losses) - 1),
            0.5,
        )

    def test_train_arc_critic_network(self):
        losses = []
        for i in range(10):
            losses.append(self.train_arc_critic_network())
        # Validate that at least 50% of the values in the list are not close or equal
        self.assertGreaterEqual(
            sum(
                [
                    not torch.isclose(losses[i], losses[i + 1])
                    for i in range(len(losses) - 1)
                ]
            )
            / (len(losses) - 1),
            0.5,
        )

    def train_arc_critic_network(self):
        """
        Test forward and backward pass of the ArcCriticNetwork class.
        """
        # Create an instance of the ArcCriticNetwork
        size = 30
        color_values = 11
        logger.info(
            "Testing ArcCriticNetwork with batch size: {}, size: {} and color values: {}".format(
                self.batch_size, size, color_values
            )
        )
        num_atoms = {"pixel_wise": int(torch.randint(50, 100, size=(1,))), "binary": 1}
        v_min = {"pixel_wise": -40, "binary": 0}
        v_max = {"pixel_wise": 2, "binary": 1}
        network = ArcCriticNetwork(
            size, color_values, num_atoms, v_min, v_max, test=True
        )

        # Create dummy input tensors
        input_sample = TensorDict(
            {
                "last_grid": torch.randint(
                    0, color_values, size=(self.batch_size, 1, size, size)
                ),
                "grid": torch.randint(
                    0, color_values, size=(self.batch_size, 1, size, size)
                ),
                "examples": torch.randint(
                    0, color_values, size=(self.batch_size, 10, 2, size, size)
                ),
                "initial": torch.randint(
                    0, color_values, size=(self.batch_size, 1, size, size)
                ),
                "index": torch.randint(0, size, size=(self.batch_size, 1, size, size)),
                "terminated": torch.randint(0, 2, size=(self.batch_size, 1)).float(),
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

        action_probs = {
            "x_location": torch.softmax(torch.randn(self.batch_size, 30), dim=-1),
            "y_location": torch.softmax(torch.randn(self.batch_size, 30), dim=-1),
            "color_values": torch.softmax(torch.randn(self.batch_size, 11), dim=-1),
            "submit": torch.softmax(torch.randn(self.batch_size, 2), dim=-1),
        }
        best_next_action = torch.cat(
            [torch.argmax(x, dim=-1).unsqueeze(-1) for x in action_probs.values()],
            dim=-1,
        ).float()

        # numeric stability test
        for key in input_sample.keys():
            if torch.rand(1).item() < 0.2:
                input_sample[key] = input_sample[key] * 0.0
        if torch.rand(1).item() < 0.1:
            best_next_action = best_next_action * 0.0
        # Forward pass
        org_sample = input_sample.clone()
        output = network(input_sample, action=best_next_action)

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
                    torch.randint(
                        -40, 2, size=(self.batch_size, num_atoms["pixel_wise"])
                    ).float(),
                    dim=-1,
                ),
                "binary": torch.randint(
                    0, 2, size=(self.batch_size, num_atoms["binary"])
                ).float(),
            }
        )

        # Assert probability mass function
        for key, dist in output.items():
            torch.testing.assert_close(
                torch.sum(dist, dim=1), torch.ones(self.batch_size)
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
        for name, param in network.named_parameters():
            if param.grad is None:
                raise ValueError(
                    f"Gradient not flowing in ArcCriticNetwork for: {name}"
                )
            # NOTE: For debugging on update with relevan testing data
            # else:
            #     print(name, param.grad.abs().sum())
            #     assert torch.all(
            #         param.grad.abs().sum() > 0
            #     ), f"Gradient of zero for ArcCriticNetwork for: {name}"
        optimizer.step()

        return loss

    def train_arc_actor_network(self):
        """
        Test forward and backward pass of the ArcActorNetwork class.
        """
        # Create an instance of the ArcActorNetwork
        size = 30
        color_values = 11
        network = ArcActorNetwork(size, color_values, test=True)
        logger.info(
            "Testing ArcActorNetwork with batch size: {}, size: {} and color values: {}".format(
                self.batch_size, size, color_values
            )
        )

        # Create dummy input tensors
        input_sample = TensorDict(
            {
                "last_grid": torch.randint(
                    0, color_values, size=(self.batch_size, 1, size, size)
                ),
                "grid": torch.randint(
                    0, color_values, size=(self.batch_size, 1, size, size)
                ),
                "examples": torch.randint(
                    0, color_values, size=(self.batch_size, 10, 2, size, size)
                ),
                "initial": torch.randint(
                    0, color_values, size=(self.batch_size, 1, size, size)
                ),
                "index": torch.randint(0, size, size=(self.batch_size, 1, size, size)),
                "terminated": torch.randint(0, 2, size=(self.batch_size, 1)).float(),
            }
        )

        network.input_val(input_sample)
        # Bad input
        for key in input_sample.keys():
            dc = input_sample.clone()
            with self.assertRaises(AssertionError):
                dc.pop(key)
                network.input_val(dc)

        # numeric stability test
        for key in input_sample.keys():
            if torch.rand(1).item() < 0.2:
                input_sample[key] = input_sample[key] * 0.0
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
                "x_location": torch.softmax(
                    torch.rand(size=(self.batch_size, size)), dim=1
                ),
                "y_location": torch.softmax(
                    torch.rand(size=(self.batch_size, size)), dim=1
                ),
                "color_values": torch.softmax(
                    torch.rand(size=(self.batch_size, color_values)), dim=1
                ),
                "submit": torch.softmax(torch.rand(size=(self.batch_size, 2)), dim=1),
            }
        )

        # Assert probability mass function
        for key, dist in output.items():
            torch.testing.assert_close(
                torch.sum(dist, dim=1), torch.ones(self.batch_size)
            ), f"Probability mass function not normalized for key: {key}"
            assert torch.all(dist >= 0), f"Negative probability values for key: {key}"
            assert torch.all(
                dist <= 1
            ), f"Probability values greater than 1 for key: {key}"

        # Calculate the loss
        loss = sum([criterion(o, t) for o, t in zip(output.values(), target.values())])

        # Assert los is not NaN
        assert not torch.isnan(loss), "{} is NaN for Actor network".format(
            criterion._get_name()
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        for name, param in network.named_parameters():
            if param.grad is None:
                raise ValueError(f"Gradient not flowing in ArcActorNetwork for: {name}")
            # NOTE: For debugging on update with relevan testing data
            # else:
            #     assert torch.all(
            #         param.grad.abs().sum() > 0
            #     ), f"Gradient of zero for ArcActorNetwork for: {name}"
        optimizer.step()
        return loss


if __name__ == "__main__":
    unittest.main()
