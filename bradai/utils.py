from __future__ import annotations

from typing import Any, Mapping
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


def to_device(thing: Any, device: str | torch.device) -> Any:
    if hasattr(thing, "to"):
        return thing.to(device)
    elif isinstance(thing, list):
        return [to_device(thing, device) for thing in thing]
    elif isinstance(thing, tuple):
        return tuple(to_device(thing, device) for thing in thing)
    elif isinstance(thing, Mapping):
        return {k: to_device(v, device) for k, v in thing.items()}
    else:
        raise TypeError(f"Cannot move {type(thing)} to device")


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader | None = None
    test: DataLoader | None = None
