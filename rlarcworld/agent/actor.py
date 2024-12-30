import torch
from tensordict import TensorDict

import logging

logger = logging.getLogger(__name__)


class ArcActorNetwork:
    def __init__(self, size: int, color_values):
        self.size = size
        self.color_values = color_values

    def input_val(self, state: TensorDict):
        assert isinstance(state, TensorDict), TypeError(
            "Input State must be a TensorDict"
        )
        in_keys = {
            "last_grid",
            "grid",
            "examples",
            "initial",
            "index",
            "terminated",
        }
        assert set(state.keys()) == in_keys, ValueError(
            "Action keys must be {}".format(in_keys)
        )

    def output_val(self, action: TensorDict):
        assert isinstance(action, TensorDict), TypeError("Action must be a TensorDict")
        in_keys = {"color_values", "submit", "x_location", "y_location"}
        assert set(action.keys()) == in_keys, ValueError(
            "Action keys must be {}".format(in_keys)
        )

    def predict(self, state: TensorDict):
        self.input_val(state)
        batch_size = state["state"].shape[0]
        output = TensorDict(
            {
                "x_location": torch.nn.functional.softmax(
                    torch.rand(size=(batch_size, self.size)), dim=1
                ),
                "y_location": torch.nn.functional.softmax(
                    torch.rand(size=(batch_size, self.size)), dim=1
                ),
                "color_values": torch.nn.functional.softmax(
                    torch.rand(size=(batch_size, self.color_values)), dim=1
                ),
                "submit": torch.nn.functional.softmax(
                    torch.rand(size=(batch_size, 2)), dim=1
                ),
            }
        )
        self.output_val(output)
        return output

    def get_discrete_actions(self, action: TensorDict):
        for key, value in action.items():
            action[key] = torch.argmax(value, dim=1)
        return action
