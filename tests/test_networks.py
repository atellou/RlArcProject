import logging

import pytest
import torch
from tensordict import TensorDict

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.agent.nn_modules import (
    CnnAttention,
    CnnPreTrainedModule,
    CrossAttentionClassifier,
    MultiHeadAttention,
)
from rlarcworld.common.enums import ActionKeys, RewardKeys, StateKeys

logger = logging.getLogger(__name__)


# ---------------- Fixtures ----------------


@pytest.fixture()
def batch_size() -> int:
    # Randomize to keep test variability but cast to int
    return int(torch.randint(1, 20, size=(1,)).item())


# ---------------- Tests ----------------


def test_cnn_pretrained():
    logger.info("Testing CNN Pretrained Module")
    model = CnnPreTrainedModule()
    input_tensor = torch.randn(1, 1, 30, 30)
    optimizer = torch.optim.RMSprop(model.parameters())
    criterion = torch.nn.MSELoss()
    output = model(input_tensor)
    assert output.shape == (1, 128)
    loss = criterion(output, torch.randn(1, 128))
    optimizer.zero_grad()
    loss.backward()
    assert model.embedding_layer.weight.grad is not None
    # Test that the batch normalization layers have gradients
    for i, module in enumerate(model.modules()):
        if i == 0 or isinstance(module, torch.nn.BatchNorm2d):
            for j, (name, param) in enumerate(module.named_parameters()):
                if j != 0:
                    continue
                assert param.grad is not None, f"Gradient not flowing in {name}"

    optimizer.step()


def test_mha():
    logger.info("Testing MultiHeadAttention")
    model = MultiHeadAttention(E_q=128, E_k=128, E_v=128, E_total=128, nheads=4)
    q = torch.randn(10, 10, 128)
    k = torch.randn(10, 10, 128)
    v = k
    output = model(q, k, v)
    optimizer = torch.optim.RMSprop(model.parameters())
    criterion = torch.nn.MSELoss()
    loss = criterion(output, torch.randn(10, 10, 128))
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None

    optimizer.step()


def test_cnn_mha():
    logger.info("Testing CnnModule + MultiHeadAttention")
    model = CnnAttention(embedding_size=128, nheads=4, dropout=0.1, bias=True)
    input_tensor = torch.randn(20, 10, 2, 30, 30)
    output = model(input_tensor)
    optimizer = torch.optim.RMSprop(model.parameters())
    criterion = torch.nn.MSELoss()
    loss = criterion(output, torch.randn(20, 10, 128))
    optimizer.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        if "base_model" in name:
            continue
        assert param.grad is not None, f"Gradient not flowing in {name}"

    optimizer.step()


def test_cross_attention():
    logger.info("Testing CnnModule + MultiHeadAttention + CrossAttention")
    output_classes = {
        ActionKeys.X_LOCATION.value: 10,
        ActionKeys.Y_LOCATION.value: 5,
        ActionKeys.COLOR_VALUES.value: 2,
    }
    model = torch.nn.Sequential(
        CnnAttention(embedding_size=128, nheads=4, dropout=0.1, bias=True),
        CrossAttentionClassifier(
            output_classes=output_classes,
            embedding_size=128,
            nheads=4,
            dropout=0.1,
            bias=True,
        ),
    )
    input_tensor = torch.randn(20, 10, 2, 30, 30)
    output = model(input_tensor)
    optimizer = torch.optim.RMSprop(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    loss = {
        k: criterion(output[k], torch.softmax(torch.randn(20, v), dim=-1))
        for k, v in output_classes.items()
    }
    loss = sum(loss.values()) / len(loss)
    optimizer.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        if "base_model" in name:
            continue
        assert param.grad is not None, f"Gradient not flowing in {name}"

    optimizer.step()


def test_train_arc_actor_network(batch_size: int):
    """Validate the actor network is trainable by observing loss variability."""
    losses = [train_arc_actor_network(batch_size) for _ in range(10)]
    variability = sum(
        [not torch.isclose(losses[i], losses[i + 1]) for i in range(len(losses) - 1)]
    ) / max(1, (len(losses) - 1))
    assert variability >= 0.5


def test_train_arc_critic_network(batch_size: int):
    """Validate the critic network is trainable by observing loss variability."""
    losses = [train_arc_critic_network(batch_size) for _ in range(10)]
    variability = sum(
        [not torch.isclose(losses[i], losses[i + 1]) for i in range(len(losses) - 1)]
    ) / max(1, (len(losses) - 1))
    assert variability >= 0.5


# ---------------- Helpers (former methods) ----------------


def train_arc_critic_network(batch_size: int):
    size = 30
    color_values = 11
    logger.info(
        "Testing ArcCriticNetwork with batch size: %s, size: %s and color values: %s",
        batch_size,
        size,
        color_values,
    )
    num_atoms = {
        RewardKeys.PIXEL_WISE.value: int(torch.randint(50, 100, size=(1,)).item()),
        RewardKeys.BINARY.value: 1,
    }
    v_min = {RewardKeys.PIXEL_WISE.value: -40, RewardKeys.BINARY.value: 0}
    v_max = {RewardKeys.PIXEL_WISE.value: 2, RewardKeys.BINARY.value: 1}
    network = ArcCriticNetwork(size, color_values, num_atoms, v_min, v_max)

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
                torch.randn(batch_size, size), dim=-1
            ),
            ActionKeys.Y_LOCATION.value: torch.softmax(
                torch.randn(batch_size, size), dim=-1
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
                    -40, 2, size=(batch_size, num_atoms[RewardKeys.PIXEL_WISE.value])
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
    for name, param in network.named_parameters():
        input_key = name.split(".")[1]
        if "base_model" not in name and param.grad is None:
            raise ValueError(f"Gradient not flowing in ArcCriticNetwork for: {name}")
        elif "base_model" not in name and input_key not in to_zero:
            assert (
                not torch.all(param.grad.abs().sum() == 0)
                or torch.all(input_sample.get(key, torch.tensor(0)) == 0)
                or torch.all(action_probs.get(key, torch.tensor(0)) == 0)
            ), f"Gradient of zero for ArcCriticNetwork for: {name}"
    optimizer.step()

    return loss


def train_arc_actor_network(batch_size: int):
    size = 30
    color_values = 11
    network = ArcActorNetwork(size, color_values)
    logger.info(
        "Testing ArcActorNetwork with batch size: %s, size: %s and color values: %s",
        batch_size,
        size,
        color_values,
    )

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
    network.input_val(input_sample)
    for key in input_sample.keys():
        dc = input_sample.clone()
        import pytest

        with pytest.raises(AssertionError):
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
            ActionKeys.X_LOCATION.value: torch.softmax(
                torch.rand(size=(batch_size, size)), dim=1
            ),
            ActionKeys.Y_LOCATION.value: torch.softmax(
                torch.rand(size=(batch_size, size)), dim=1
            ),
            ActionKeys.COLOR_VALUES.value: torch.softmax(
                torch.rand(size=(batch_size, color_values)), dim=1
            ),
            ActionKeys.SUBMIT.value: torch.softmax(
                torch.rand(size=(batch_size, 2)), dim=1
            ),
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

    loss = sum([criterion(o, t) for o, t in zip(output.values(), target.values())])

    assert not torch.isnan(loss), f"{criterion._get_name()} is NaN for Actor network"

    optimizer.zero_grad()
    loss.backward()
    for name, param in network.named_parameters():
        input_key = name.split(".")[1]
        if "base_model" not in name and param.grad is None:
            raise ValueError(f"Gradient not flowing in ArcActorNetwork for: {name}")
        elif "base_model" not in name and input_key not in to_zero:
            assert not torch.all(param.grad.abs().sum() == 0) or torch.all(
                input_sample.get(key, torch.tensor(0)) == 0
            ), f"Gradient of zero for ArcActorNetwork for: {name}"
    optimizer.step()
    return loss
