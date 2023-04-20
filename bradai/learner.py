from __future__ import annotations
from dataclasses import dataclass
from functools import partial, wraps
from operator import attrgetter
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .callbacks import Callback, LRFinderCallback, MetricsCallback
from .exceptions import CancelException


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader | None = None
    test: DataLoader | None = None


class _CallbackWrapper:
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def _wrapper(owner: Learner, *args: Any, **kwargs: Any) -> Any:
            try:
                owner.callback(f"before_{self.name}")
                result = func(owner, *args, **kwargs)
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
        dataloaders: DataLoaders,
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

        # type definitions for stuff that is dynamically added
        self.batch: Sequence
        self.batch_inputs: Sequence
        self.batch_targets: Sequence
        self.dataloader: DataLoader | Iterable
        self.epochs: Iterable
        self.log: Callable
        self.loss: torch.Tensor
        self.metrics: MetricsCallback
        self.opt: torch.optim.Optimizer
        self.preds: Any

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
        self.dataloader = self.dataloaders.train if train else (self.dataloaders.val or self.dataloaders.train)
        self._one_epoch()

    @_CallbackWrapper("fit")
    def _fit(self, val_freq: int = 1) -> None:
        for self.epoch in self.epochs:
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
            self.epochs = range(self.total_epochs)
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
