from __future__ import annotations
from functools import partial
from typing import Any, Callable

import accelerate
import matplotlib.pyplot as plt
import torch
import wandb
from fastprogress import master_bar, progress_bar
from torcheval.metrics import Metric, Mean
from wandb.sdk.wandb_run import Run

from .learner import Learner, CancelException
from .utils import to_device


class Callback:
    order: int = 0


class AccelerateCallback(Callback):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.accelerator = accelerate.Accelerator(*args, **kwargs)

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
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.hysteresis = hysteresis

    def before_fit(self, learner: Learner) -> None:
        self.lrs: list[float] = []
        self.losses: list[float] = []
        self.sched = torch.optim.lr_scheduler.ExponentialLR(learner.opt, self.gamma)
        self.best = float("inf")

    def after_batch(self, learner: Learner) -> None:
        if not learner.training:
            raise CancelException(name="epoch")

        self.lrs.append(self.sched.get_last_lr()[0])
        loss = float(learner.loss.item())
        self.losses.append(loss)
        self.best = min(self.best, loss)
        if loss > self.best * self.hysteresis:
            raise CancelException(name="fit")
        self.sched.step()

    def cleanup_fit(self, learner: Learner) -> None:
        fig, ax = plt.subplots()
        ax.plot(self.lrs, self.losses)
        ax.set_xscale("log")
        fig.show()


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
        learner.batch_inputs = learner.batch[: self.num_inputs]  # type: ignore
        learner.preds = learner.model(*learner.batch_inputs)  # type: ignore

    def get_loss(self, learner: Learner) -> None:
        learner.batch_targets = learner.batch[self.num_inputs :]  # type: ignore
        learner.loss = learner.criterion(  # type: ignore
            learner.preds,
            learner.batch_targets,
        )

    def backward(self, learner: Learner) -> None:
        learner.loss.backward()

    def step(self, learner: Learner) -> None:
        learner.opt.step()

    def zero_grad(self, learner: Learner) -> None:
        learner.opt.zero_grad(True)


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
        learner.metrics = self  # type: ignore

    def before_epoch(self, learner: Learner) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def after_epoch(self, learner: Learner) -> None:
        prefix = "train" if learner.training else "val"
        log = {
            f"{prefix}/{name}": metric.compute()
            for name, metric in self.metrics.items()
        }
        log[f"{prefix}/epoch"] = learner.epoch
        learner.log(log)

    def after_batch(self, learner: Learner) -> None:
        targets = to_device(learner.batch_targets, "cpu")
        for metric in self.metrics.values():
            metric.update(to_device(learner.preds, "cpu"), targets)
        self.loss.update(to_device(learner.loss, "cpu"), weight=len(learner.batch_inputs[0]))  # type: ignore


class ProgressBarCallback(Callback):
    order: int = 1

    def __init__(self, plot: bool = False) -> None:
        super().__init__()
        self.plot = plot

    def _update_graph(self, learner: Learner) -> None:
        self.mbar.update_graph(
            [
                [range(len(self.losses)), self.losses],
                [
                    torch.arange(1, learner.total_epochs + 1)
                    * len(learner.dataloaders.train),
                    self.val_losses,
                ],
            ]
        )

    def _log(
        self, learner: Learner, log: dict[str, float], wrapped: Callable | None = None
    ) -> None:
        if self.first_epoch:  # type: ignore
            self.mbar.write(list(log), table=True)
            self.first_epoch = False
        self.mbar.write(list(log.values()), table=True)
        if wrapped is not None:
            wrapped(log)

    def before_fit(self, learner: Learner) -> None:
        learner.epochs = self.mbar = master_bar(learner.total_epochs)  # type: ignore
        self.first_epoch = True
        if hasattr(learner, "metrics"):
            log_fn = partial(self._log, wrapped=getattr(learner, "log", None))
            learner.log = log_fn  # type: ignore
        self.losses: list[float] = []
        self.val_losses: list[float] = []

    def before_epoch(self, learner: Learner) -> None:
        learner.dataloader = progress_bar(
            learner.dataloader, parent=self.mbar, leave=False
        )

    def after_batch(self, learner: Learner) -> None:
        learner.dataloader.comment = f"Loss: {learner.loss:10.04e}"
        if self.plot and hasattr(learner, "metrics") and learner.training:
            self.losses.append(learner.loss.item())
            if self.val_losses:
                self._update_graph(learner)

    def after_epoch(self, learner: Learner) -> None:
        if not learner.training:
            if self.plot and hasattr(learner, "metrics"):
                self.val_losses.append(learner.metrics.loss.compute())
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
