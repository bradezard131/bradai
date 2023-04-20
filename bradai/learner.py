from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from operator import attrgetter
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .callbacks import Callback, LRFinderCallback


@dataclass
class DataLoaders:
    train: DataLoader | None = None
    val: DataLoader | None = None
    test: DataLoader | None = None


class CancelException(Exception):
    def __init__(self, *args, name: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name


class _CallbackWrapper:
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> Callable:
        def _wrapper(owner, *args, **kwargs):
            try:
                owner.callback(f"before_{self.name}")
                result = func(*args, **kwargs)
                owner.callback(f"after_{self.name}")
                return result
            except CancelException as e:
                if e.name != self.name:
                    raise e
            finally:
                owner.callback(f"cleanup_{self.name}")

        return _wrapper


class Learner:
    def __init__(
        self,
        model: nn.Module,
        dataloaders,
        criterion: Callable,
        default_learning_rate: float,
        opt_fn: Callable,
        callbacks: list[Callback] = [],
    ) -> None:
        super().__init__()
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.default_learning_rate = default_learning_rate
        self.opt_fn = opt_fn
        self.callbacks = callbacks

    @property
    def training(self) -> bool:
        return self.model.training

    @_CallbackWrapper("batch")
    def one_batch(self) -> None:
        self.predict()
        self.callback("after_predict")
        self.get_loss()
        self.callback("after_loss")
        if self.training:
            self.backward()
            self.callback("after_backward")
            self.step()
            self.callback("after_step")
            self.zero_grad()

    @_CallbackWrapper("epoch")
    def _one_epoch(self) -> None:
        for self.iteration, self.batch in enumerate(self.dataloader):
            self.one_batch()

    def one_epoch(self, train: bool) -> None:
        self.model.train(train)
        self.dataloader = self.dataloaders.train if train else self.dataloaders.val
        self._one_epoch()

    @_CallbackWrapper("fit")
    def _fit(self, val_freq: int = 1) -> None:
        for self.epoch in range(self.total_epochs):
            self.one_epoch(True)
            self.callback("after_train")
            if self.epoch % val_freq == 0:
                with torch.inference_mode():
                    self.one_epoch(False)
                self.callback("after_val")

    def fit(
        self,
        epochs: int,
        val_freq: int = 1,
        callbacks: list[Callback] = [],
        learning_rate: float | None = None,
    ) -> None:
        self.callbacks.extend(callbacks)  # add temporary callbacks
        try:
            self.total_epochs = epochs
            learning_rate = learning_rate or self.default_learning_rate
            self.opt = self.opt_fn(self.model.parameters(), lr=learning_rate)
            self._fit(val_freq)
        finally:
            self.callbacks = self.callbacks[: -len(callbacks)]  # remove temp callbacks

    def __getattr__(self, name: str) -> Callable:
        if name in ("predict", "get_loss", "backward", "step", "zero_grad", "log"):
            return partial(self.callback, name)
        raise AttributeError(name)

    def callback(self, name: str) -> None:
        for callback in sorted(self.callbacks, key=attrgetter("order")):
            method = getattr(callback, name, None)
            if method is not None:
                method(self)

    def lr_find(
        self,
        gamma: float = 1.3,
        hysteresis: float = 3.0,
        start_lr: float = 1e-6,
        max_epochs: int = 10,
    ) -> None:
        self.fit(
            max_epochs,
            val_freq=0,
            callbacks=[LRFinderCallback(gamma, hysteresis)],
            learning_rate=start_lr,
        )
