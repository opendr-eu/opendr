"""
Copyright (c) Lukas Hedegaard. All Rights Reserved.
Included in the OpenDR Toolit with permission from the author.
"""

import torch
from torch import Tensor
from typing import Tuple
from logging import getLogger
from .utils import FillMode

State = Tuple[Tensor, int]

logger = getLogger(__name__)


class Delay(torch.nn.Module):
    def __init__(
        self, window_size: int, temporal_fill: FillMode = "replicate",
    ):
        assert window_size > 0
        assert temporal_fill in {"zeros", "replicate"}
        self.window_size = window_size
        self.make_padding = {"zeros": torch.zeros_like, "replicate": torch.clone}[
            temporal_fill
        ]

        super(Delay, self).__init__()
        # state is initialised in self.forward

    def init_state(self, first_output: Tensor,) -> State:
        padding = self.make_padding(first_output)
        state_buffer = torch.stack([padding for _ in range(self.window_size)], dim=0)
        state_index = 0
        if not hasattr(self, "state_buffer"):
            self.register_buffer("state_buffer", state_buffer, persistent=False)
        return state_buffer, state_index

    def clean_state(self):
        self.state_buffer = None
        self.state_index = None

    def get_state(self):
        if (
            hasattr(self, "state_buffer") and
            self.state_buffer is not None and
            hasattr(self, "state_index") and
            self.state_buffer is not None
        ):
            return (self.state_buffer, self.state_index)
        else:
            return None

    def forward3d(self, input: Tensor) -> Tensor:
        # Pass into delay line, but discard output
        self.forward(input)

        # No delay during forward3d
        return input

    def forward(self, input: Tensor) -> Tensor:
        output, (self.state_buffer, self.state_index) = self._forward(
            input, self.get_state()
        )
        return output

    def _forward(self, input: Tensor, prev_state: State) -> Tuple[Tensor, State]:
        assert len(input.shape) == 4, "Only a single frame should be passed at a time."

        if prev_state is None:
            buffer, index = self.init_state(input)
        else:
            buffer, index = prev_state

        # Get output
        output = buffer[index]

        # Update state
        new_buffer = buffer.clone() if self.training else buffer.detach()
        new_index = (index + 1) % self.window_size
        new_buffer[(index - 1) % self.window_size] = input

        return output, (new_buffer, new_index)
