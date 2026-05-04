"""Training loop for the unified LatentDynamicsAutoencoder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.schema import ArchConfig, TrainingConfig
from ..models.autoencoder import LatentDynamicsAutoencoder
from .checkpoints import save_checkpoint
from .losses import LossBreakdown, build_loss


@dataclass
class LossHistory:
    """Per-epoch component-wise loss values for train and test splits."""

    train: dict[str, list[float]] = field(default_factory=lambda: _empty_loss_dict())
    test: dict[str, list[float]] = field(default_factory=lambda: _empty_loss_dict())

    def append_train(self, breakdown: dict[str, float]) -> None:
        for k, v in breakdown.items():
            self.train[k].append(v)

    def append_test(self, breakdown: dict[str, float]) -> None:
        for k, v in breakdown.items():
            self.test[k].append(v)

    def to_json(self) -> str:
        return json.dumps({"train": self.train, "test": self.test}, indent=2)


def _empty_loss_dict() -> dict[str, list[float]]:
    return {"loss_ae1": [], "loss_ae2": [], "loss_dyn": [], "loss_total": []}


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Train a :class:`LatentDynamicsAutoencoder` end-to-end with early stopping."""

    def __init__(
        self,
        model: LatentDynamicsAutoencoder,
        train_loader: DataLoader,
        test_loader: DataLoader,
        training_cfg: TrainingConfig,
        arch_cfg: ArchConfig,
        *,
        device: torch.device | None = None,
        verbose: bool = True,
    ) -> None:
        self.device = device or _select_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = training_cfg
        self.arch = arch_cfg
        self.verbose = bool(verbose)

        self.loss_fn = build_loss(training_cfg.loss_mode, training_cfg.loss_weights).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=training_cfg.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=training_cfg.scheduler_factor,
            patience=training_cfg.patience,
            threshold=training_cfg.scheduler_threshold,
            min_lr=training_cfg.scheduler_min_lr,
        )
        self.history = LossHistory()
        self._best_train_loss = float("inf")
        self._no_improve = 0

    def _run_epoch(self, loader: DataLoader, *, training: bool) -> dict[str, float]:
        self.model.train(training)
        accum = {"loss_ae1": 0.0, "loss_ae2": 0.0, "loss_dyn": 0.0, "loss_total": 0.0}
        n_batches = max(1, len(loader))
        ctx = torch.enable_grad if training else torch.no_grad
        with ctx():
            for x_t, x_tau in loader:
                x_t = x_t.to(self.device, non_blocking=False)
                x_tau = x_tau.to(self.device, non_blocking=False)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                fp = self.model(x_t, x_tau)
                breakdown: LossBreakdown = self.loss_fn(fp)
                if training:
                    if not torch.isfinite(breakdown.total):
                        raise FloatingPointError(
                            f"non-finite loss after epoch {len(self.history.train['loss_total'])} "
                            f"(loss_total={breakdown.total.item():.4e}); training diverged"
                        )
                    breakdown.total.backward()
                    if self.cfg.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.cfg.gradient_clip_norm,
                        )
                    self.optimizer.step()
                for k, v in breakdown.detach_dict().items():
                    accum[k] += v
        return {k: v / n_batches for k, v in accum.items()}

    def fit(self) -> LossHistory:
        if self.verbose:
            print(f"device: {self.device}")
            print(f"loss_mode={self.cfg.loss_mode}, weights={self.cfg.loss_weights}")
        iterator = tqdm(range(self.cfg.epochs), disable=not self.verbose)
        for epoch in iterator:
            train_breakdown = self._run_epoch(self.train_loader, training=True)
            test_breakdown = self._run_epoch(self.test_loader, training=False)
            self.history.append_train(train_breakdown)
            self.history.append_test(test_breakdown)
            self.scheduler.step(test_breakdown["loss_total"])

            if train_breakdown["loss_total"] < self._best_train_loss:
                self._best_train_loss = train_breakdown["loss_total"]
                self._no_improve = 0
            else:
                self._no_improve += 1
                if self._no_improve >= self.cfg.patience:
                    if self.verbose:
                        iterator.write(f"early stopping at epoch {epoch + 1}")
                    break

            if self.verbose:
                iterator.set_postfix(
                    train=f"{train_breakdown['loss_total']:.4e}",
                    test=f"{test_breakdown['loss_total']:.4e}",
                )
        return self.history

    def save(self, output_dir: str | Path, *, basename: str = "autoencoder") -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(self.model, self.arch, out / "models", basename=basename)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "logs" / "history.json").write_text(self.history.to_json())
        return out
