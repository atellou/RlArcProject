import os
import torch
from tensordict import TensorDict

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.agent.models.resnet_module import ResNetModule

import unittest
import logging


logger = logging.getLogger(__name__)


class ArcNetworksTest(unittest.TestCase):

    def setUp(self):
        """
        Set up the test.

        This method sets up the test by generating a random batch size.
        """
        self.batch_size = torch.randint(1, 20, size=(1,))

    def test_resnet(self):
        logger.info("Testing resnet50")
        model = ResNetModule(do_not_freeze="layer4")
        input_tensor = torch.randn(1, 1, 30, 30)
        optimizer = torch.optim.RMSprop(model.parameters())
        criterion = torch.nn.MSELoss()
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 256))
        loss = criterion(output, torch.randn(1, 256))
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(model.embedding_layer.weight.grad is not None)
        self.assertTrue(model.base_model.conv1.weight.grad is not None)
        # Test that the batch normalization layers have gradients
        for module in model.base_model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                for param in module.parameters():
                    self.assertTrue(param.grad is not None)

        for name, param in model.named_parameters():
            if name.startswith("base_model.layer4"):
                self.assertTrue(param.grad is not None)

        optimizer.step()

        logger.info("Testing resnet18")
        model = ResNetModule(
            resnet_version="resnet18",
            resnet_weights="ResNet18_Weights.DEFAULT",
            freeze="ALL",
        )
        input_tensor = torch.randn(1, 1, 30, 30)
        optimizer = torch.optim.RMSprop(model.parameters())
        criterion = torch.nn.MSELoss()
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 256))
        loss = criterion(output, torch.randn(1, 256))
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(model.embedding_layer.weight.grad is not None)
        self.assertTrue(model.base_model.conv1.weight.grad is not None)
        # Test that the batch normalization layers have gradients
        for module in model.base_model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                for param in module.parameters():
                    self.assertTrue(param.grad is not None)

        optimizer.step()

    def test_train_arc_actor_network(self):
        """
        Validate that the actor network is trainable.

        This test validates that the actor network is trainable by
        training it 10 times and checking that at least 50% of the
        losses are not close or equal.

        The test is successful if at least 50% of the losses are not
        close or equal.
        """
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
        """
        Validate that the critic network is trainable.

        This test validates that the critic network is trainable by
        training it 10 times and checking that at least 50% of the
        losses are not close or equal.

        The test is successful if at least 50% of the losses are not
        close or equal.
        """
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

        This method tests the forward and backward pass of the
        ArcCriticNetwork class by creating an instance of the
        network, validating the input, performing a forward pass,
        calculating the loss, and performing backpropagation.
        """
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
        network = ArcCriticNetwork(size, color_values, num_atoms, v_min, v_max)

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
        for key in input_sample.keys():
            dc = input_sample.clone()
            with self.assertRaises(AssertionError):
                dc.pop(key)
                network.input_val(dc)

        action_probs = TensorDict(
            {
                "x_location": torch.softmax(torch.randn(self.batch_size, 30), dim=-1),
                "y_location": torch.softmax(torch.randn(self.batch_size, 30), dim=-1),
                "color_values": torch.softmax(torch.randn(self.batch_size, 11), dim=-1),
                "submit": torch.softmax(torch.randn(self.batch_size, 2), dim=-1),
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

        for key, dist in output.items():
            torch.testing.assert_close(
                torch.sum(dist, dim=1), torch.ones(self.batch_size)
            ), f"Probability mass function not normalized for key: {key}"
            assert torch.all(dist >= 0), f"Negative probability values for key: {key}"
            assert torch.all(
                dist <= 1
            ), f"Probability values greater than 1 for key: {key}"

        loss = criterion(output["pixel_wise"], target["pixel_wise"]) + criterion(
            output["binary"], target["binary"]
        )

        assert not torch.isnan(loss), "{} is NaN for Critic network".format(
            criterion._get_name()
        )

        optimizer.zero_grad()
        loss.backward()
        for name, param in network.named_parameters():
            input_key = name.split(".")[1]
            if param.grad is None:
                raise ValueError(
                    f"Gradient not flowing in ArcCriticNetwork for: {name}"
                )
            elif input_key not in to_zero:
                assert (
                    not torch.all(param.grad.abs().sum() == 0)
                    or torch.all(input_sample.get(key, torch.tensor(0)) == 0)
                    or torch.all(action_probs.get(key, torch.tensor(0)) == 0)
                ), f"Gradient of zero for ArcCriticNetwork for: {name}"
        optimizer.step()

        return loss

    def train_arc_actor_network(self):
        """
        Test forward and backward pass of the ArcActorNetwork class.

        This method tests the forward and backward pass of the
        ArcActorNetwork class by creating an instance of the
        network, validating the input, performing a forward pass,
        calculating the loss, and performing backpropagation.
        """
        size = 30
        color_values = 11
        network = ArcActorNetwork(size, color_values)
        logger.info(
            "Testing ArcActorNetwork with batch size: {}, size: {} and color values: {}".format(
                self.batch_size, size, color_values
            )
        )

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
            }
        )
        network.input_val(input_sample)
        for key in input_sample.keys():
            dc = input_sample.clone()
            with self.assertRaises(AssertionError):
                dc.pop(key)
                network.input_val(dc)

        to_zero = []
        for key in input_sample.keys():
            if torch.rand(1).item() < 0.2:
                input_sample[key] = input_sample[key] * 0.0
                to_zero.append(key)
        org_sample = input_sample.clone()
        output = network(input_sample)

        torch.testing.assert_close(input_sample, org_sample)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(network.parameters())

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

        for key, dist in output.items():
            torch.testing.assert_close(
                torch.sum(dist, dim=1), torch.ones(self.batch_size)
            ), f"Probability mass function not normalized for key: {key}"
            assert torch.all(dist >= 0), f"Negative probability values for key: {key}"
            assert torch.all(
                dist <= 1
            ), f"Probability values greater than 1 for key: {key}"

        loss = sum([criterion(o, t) for o, t in zip(output.values(), target.values())])

        assert not torch.isnan(loss), "{} is NaN for Actor network".format(
            criterion._get_name()
        )

        optimizer.zero_grad()
        loss.backward()
        for name, param in network.named_parameters():
            input_key = name.split(".")[1]
            if param.grad is None:
                raise ValueError(f"Gradient not flowing in ArcActorNetwork for: {name}")
            elif input_key not in to_zero:
                assert not torch.all(param.grad.abs().sum() == 0) or torch.all(
                    input_sample.get(key, torch.tensor(0)) == 0
                ), f"Gradient of zero for ArcActorNetwork for: {name}"
        optimizer.step()
        return loss


if __name__ == "__main__":
    unittest.main()
