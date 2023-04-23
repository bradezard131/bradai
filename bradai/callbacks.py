from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Any, Callable, TYPE_CHECKING

import accelerate
import matplotlib.pyplot as plt
import torch
import wandb
from fastprogress.fastprogress import master_bar, progress_bar, MasterBar
from torcheval.metrics import Metric, Mean
from wandb.sdk.wandb_run import Run

from .exceptions import CancelException
from .utils import to_device

if TYPE_CHECKING:
    from .learner import Learner


class Callback:
    order: int = 0


class CompletionCallback(Callback):
    count: int = 0

    def before_fit(self, learner: Learner) -> None:
        self.count = 0

    def after_batch(self, learner: Learner) -> None:
        self.count += 1

    def after_fit(self, learner: Learner) -> None:
        print(f"{self.count} batches completed")


class DeviceCallback(Callback):
    def __init__(self, device: str | torch.device) -> None:
        super().__init__()
        self.device = torch.device(device)

    def before_fit(self, learner: Learner) -> None:
        learner.model.to(self.device)

    def before_batch(self, learner: Learner) -> None:
        learner.batch = to_device(learner.batch, self.device)


class OneBatchCallback(Callback):
    order: int = 1

    def after_batch(self, learner: Learner) -> None:
        raise CancelException(name="fit")


class LRFinderCallback(Callback):
    def __init__(
        self,
        gamma: float = 1.3,
        hysteresis: float = 3.0,
        max_lr: float = float("inf"),
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.hysteresis = hysteresis
        self.max_lr = max_lr

    def before_fit(self, learner: Learner) -> None:
        self.lrs: list[float] = []
        self.losses: list[float] = []
        self.sched = torch.optim.lr_scheduler.ExponentialLR(learner.opt, self.gamma)
        self.best = float("inf")
        self.initial_state = deepcopy(learner.model.state_dict())

    def after_batch(self, learner: Learner) -> None:
        if not learner.training:
            raise CancelException(name="epoch")

        self.lrs.append(self.sched.get_last_lr()[0])
        loss = float(learner.loss.item())
        self.losses.append(loss)
        self.best = min(self.best, loss)
        if loss > self.best * self.hysteresis or self.lrs[-1] > self.max_lr:
            raise CancelException(name="fit")
        self.sched.step()

    def cleanup_fit(self, learner: Learner) -> None:
        fig, ax = plt.subplots()
        ax.plot(self.lrs, self.losses)
        ax.set_title("Learning Rate Finder")
        ax.set_ylabel("Loss")
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        fig.show()
        learner.model.load_state_dict(self.initial_state)
        del self.initial_state


class LRSchedulerCallback(Callback):
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        mode: str = "step",
    ) -> None:
        super().__init__()
        self.scheduler = scheduler
        if mode == "step":
            self.after_batch = self.step_scheduler
        elif mode == "epoch":
            self.after_epoch = self.step_scheduler
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def step_scheduler(self, learner: Learner) -> None:
        self.scheduler.step()


class TrainCallback(Callback):
    def __init__(self, num_inputs: int = 1) -> None:
        super().__init__()
        self.num_inputs = num_inputs

    def predict(self, learner: Learner) -> None:
        learner.batch_inputs = learner.batch[: self.num_inputs]
        learner.preds = learner.model(*learner.batch_inputs)

    def get_loss(self, learner: Learner) -> None:
        learner.batch_targets = learner.batch[self.num_inputs :]
        learner.loss = learner.criterion(learner.preds, *learner.batch_targets)

    def backward(self, learner: Learner) -> None:
        learner.loss.backward()

    def step(self, learner: Learner) -> None:
        learner.opt.step()

    def zero_grad(self, learner: Learner) -> None:
        learner.opt.zero_grad(True)


class AccelerateCallback(TrainCallback):
    def __init__(
        self,
        *args: Any,
        num_inputs: int = 1,
        accelerator: accelerate.Accelerator | None = None,
        **kwargs: Any,
    ) -> None:
        if accelerator is None:
            self.accelerator = accelerate.Accelerator(*args, **kwargs)
        else:
            if len(args) > 0 or len(kwargs) > 0:
                raise ValueError(
                    "Cannot specify accelerator and accelerator args/kwargs"
                )
            self.accelerator = accelerator
        super().__init__(num_inputs=num_inputs)

    def before_fit(self, learner: Learner) -> None:
        (
            learner.model,
            learner.criterion,
            learner.opt,
            learner.dataloaders.train,
            learner.dataloaders.val,
            learner.dataloaders.test,
        ) = self.accelerator.prepare(
            learner.model,
            learner.criterion,
            learner.opt,
            learner.dataloaders.train,
            learner.dataloaders.val,
            learner.dataloaders.test,
        )

    def before_batch(self, learner: Learner) -> None:
        self.accumulator = self.accelerator.accumulate(learner.model)
        self.accumulator.__enter__()

    def after_batch(self, learner: Learner) -> None:
        self.accumulator.__exit__(None, None, None)

    def backward(self, learner: Learner) -> None:
        self.accelerator.backward(learner.loss)


class AmpCallback(TrainCallback):
    def __init__(
        self,
        device_type: str,
        dtype: torch.dtype,
        enabled: bool = True,
        cache_enabled: bool | None = None,
    ) -> None:
        super().__init__()
        self.autocast = torch.autocast(device_type, dtype, enabled, cache_enabled)  # type: ignore
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

    def before_batch(self, learner: Learner) -> None:
        self.autocast.__enter__()

    def backward(self, learner: Learner) -> None:
        self.autocast.__exit__(None, None, None)
        self.scaler.scale(learner.loss).backward()  # type: ignore

    def step(self, learner: Learner) -> None:
        self.scaler.step(learner.opt)
        self.scaler.update()


class MetricsCallback(Callback):
    def __init__(self, *metrics: Metric, **named_metrics: Metric) -> None:
        super().__init__()
        for metric in metrics:
            named_metrics[type(metric).__name__] = metric
        self.metrics = named_metrics
        self.loss = Mean()

    def before_fit(self, learner: Learner) -> None:
        learner.metrics = self

    def before_epoch(self, learner: Learner) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self.loss.reset()

    def after_epoch(self, learner: Learner) -> None:
        prefix = "train" if learner.training else "val"
        log = {
            f"{prefix}/{name}": metric.compute()
            for name, metric in self.metrics.items()
        }
        log[f"{prefix}/loss"] = self.loss.compute()
        log[f"{prefix}/epoch"] = learner.epoch
        if learner.training:
            self.train_log = log
        else:
            log.update(self.train_log)
            learner.log(log)

    def after_batch(self, learner: Learner) -> None:
        targets = to_device(learner.batch_targets, "cpu")
        for metric in self.metrics.values():
            metric.update(to_device(learner.preds, "cpu"), targets)
        self.loss.update(
            to_device(learner.loss, "cpu"), weight=len(learner.batch_inputs[0])
        )


class ProgressBarCallback(Callback):
    order: int = 1

    def __init__(self, plot: bool = False, plot_update_freq: int = 1) -> None:
        super().__init__()
        self.plot = plot
        self.first_epoch: bool = True
        self.plot_update_freq = plot_update_freq
        self.mbar: MasterBar

    def _update_graph(self, learner: Learner) -> None:
        if learner.iteration % self.plot_update_freq == 0:
            self.mbar.update_graph(
                [
                    [range(len(self.losses)), self.losses],
                    [
                        torch.arange(1, len(self.val_losses) + 1)
                        * len(learner.dataloaders.train),  # type: ignore
                        self.val_losses,
                    ],
                ]
            )

    def _log(
        self, log: dict[str, float], learner: Learner, wrapped: Callable | None = None
    ) -> None:
        if self.first_epoch:
            self.mbar.write(list(log), table=True)  # type: ignore
            self.first_epoch = False
        self.mbar.write(list(log.values()), table=True)  # type: ignore
        if wrapped is not None:
            wrapped(log)

    def before_fit(self, learner: Learner) -> None:
        learner.epochs = self.mbar = master_bar(learner.epochs)
        self.first_epoch = True
        if hasattr(learner, "metrics"):
            log_fn = partial(
                self._log, learner=learner, wrapped=getattr(learner, "log", None)
            )
            learner.log = log_fn
        self.losses: list[float] = []
        self.val_losses: list[float] = []

    def before_epoch(self, learner: Learner) -> None:
        learner.dataloader = progress_bar(
            learner.dataloader, parent=self.mbar, leave=False
        )

    def after_batch(self, learner: Learner) -> None:
        learner.dataloader.comment = f"Loss: {learner.loss:10.04e}"  # type: ignore
        if self.plot and hasattr(learner, "metrics") and learner.training:
            self.losses.append(learner.loss.item())
            if self.val_losses:
                self._update_graph(learner)

    def after_epoch(self, learner: Learner) -> None:
        if not learner.training:
            if self.plot and hasattr(learner, "metrics"):
                self.val_losses.append(learner.metrics.loss.compute().item())
                self._update_graph(learner)


class PrintLoggerCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def log(self, learner: Learner, log: dict[str, Any]) -> None:
        print(log)


class WandBLoggerCallback(Callback):
    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: list[str] | None = None,
        name: str | None = None,
        notes: str | None = None,
        resume: str | None = None,
        reinit: bool = False,
    ) -> None:
        super().__init__()
        self.init_run = partial(
            wandb.init,
            project=project,
            entity=entity,
            group=group,
            job_type=job_type,
            tags=tags,
            name=name,
            notes=notes,
            resume=resume,
            reinit=reinit,
        )

    def before_fit(self, learner: Learner) -> None:
        self.run: Run = self.init_run()  # type: ignore
        assert isinstance(self.run, Run)
        self.run.watch(learner.model)

    def log(self, learner: Learner, log: dict[str, Any]) -> None:
        self.run.log(log)

    def cleanup_fit(self, learner: Learner) -> None:
        self.run.finish()
