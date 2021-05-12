"""
Copyright (c) Lukas Hedegaard. All Rights Reserved.
Included in the OpenDR Toolit with permission from the author.
"""

from enum import Enum
from functools import wraps
from typing import Callable

from torch import Tensor
from torch.nn import Module
from logging import getLogger

logger = getLogger(__name__)


class FillMode(Enum):
    REPLICATE = "replicate"
    ZEROS = "zeros"


def unsqueezed(instance: Module, dim: int = 2):
    def decorator(func: Callable[[Tensor], Tensor]):
        @wraps(func)
        def call(x: Tensor) -> Tensor:
            x = x.unsqueeze(dim)
            x = func(x)
            x = x.squeeze(dim)
            return x

        return call

    instance.forward3d = instance.forward
    instance.forward = decorator(instance.forward)

    return instance


def once(fn: Callable):
    called = 0

    @wraps(fn)
    def wrapped(*args, **kwargs):
        nonlocal called
        if not called:
            called = 1
            return fn(*args, **kwargs)

    return wrapped


@once
def warn_once_if(cond: bool, msg: str):
    if cond:
        logger.warning(msg)
