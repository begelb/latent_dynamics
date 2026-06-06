"""Training loop for the unified LatentDynamicsAutoencoder."""

from __future__ import annotations

import json
from collections import defaultdict
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
from .losses import LossBreakdown, ReconstructionLoss


@dataclass
class LossHistory:
    """Per-epoch component-wise loss values for train and val splits.

    Keys are created on demand from each epoch's loss breakdown, so optional
    terms (e.g. the cycle loss) appear only when they are actually used.
    """

    train: dict[str, list[float]] = field(default_factory=dict)
    val: dict[str, list[float]] = field(default_factory=dict)

    def append_train(self, breakdown: dict[str, float]) -> None:
        for k, v in breakdown.items():
            self.train.setdefault(k, []).append(v)

    def append_val(self, breakdown: dict[str, float]) -> None:
        for k, v in breakdown.items():
            self.val.setdefault(k, []).append(v)

    def to_json(self) -> str:
        return json.dumps({"train": self.train, "val": self.val}, indent=2)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Trainer:
    """Train a :class:`LatentDynamicsAutoencoder` end-to-end with early stopping."""

    def __init__(
        self,
        model: LatentDynamicsAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_cfg: TrainingConfig,
        arch_cfg: ArchConfig,
        *,
        device: torch.device | None = None,
        verbose: bool = True,
    ) -> None:
        self.device = device or _select_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = training_cfg
        self.arch = arch_cfg
        self.verbose = bool(verbose)

        self.loss_fn = ReconstructionLoss(training_cfg.loss_weights).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=training_cfg.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=training_cfg.scheduler_factor,
            patience=training_cfg.lr_patience,
            threshold=training_cfg.scheduler_threshold,
            min_lr=training_cfg.scheduler_min_lr,
        )
        self.history = LossHistory()
        self._best_val_loss = float("inf")
        self._best_epoch = -1
        self._best_state_dict: dict[str, torch.Tensor] | None = None
        self._no_improve = 0

    @property
    def best_epoch(self) -> int:
        """0-indexed epoch at which the lowest val_loss_total was observed."""
        return self._best_epoch

    def _run_epoch(self, loader: DataLoader, *, training: bool) -> dict[str, float]:
        self.model.train(training)
        accum: dict[str, float] = defaultdict(float)
        total_samples = 0
        ctx = torch.enable_grad if training else torch.no_grad
        with ctx():
            for x_t, x_tau in loader:
                x_t = x_t.to(self.device, non_blocking=False)
                x_tau = x_tau.to(self.device, non_blocking=False)
                batch_size = x_t.size(0)
                total_samples += batch_size
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
                # MSELoss(reduction="mean") returns the per-sample mean; weight
                # by batch size so the epoch average is sample-weighted, not
                # batch-count-weighted (which is biased when drop_last=False
                # and the final batch is smaller than the rest).
                for k, v in breakdown.detach_dict().items():
                    accum[k] += v * batch_size
        denom = max(1, total_samples)
        return {k: v / denom for k, v in accum.items()}

    def fit(self) -> LossHistory:
        if self.verbose:
            print(f"device: {self.device}")
            print(f"loss_weights={self.cfg.loss_weights}")
        iterator = tqdm(range(self.cfg.epochs), disable=not self.verbose)
        for epoch in iterator:
            train_breakdown = self._run_epoch(self.train_loader, training=True)
            val_breakdown = self._run_epoch(self.val_loader, training=False)
            self.history.append_train(train_breakdown)
            self.history.append_val(val_breakdown)
            self.scheduler.step(val_breakdown["loss_total"])

            if val_breakdown["loss_total"] < self._best_val_loss:
                self._best_val_loss = val_breakdown["loss_total"]
                self._best_epoch = epoch
                self._best_state_dict = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
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
                    val=f"{val_breakdown['loss_total']:.4e}",
                )

        # Restore the best-val weights so save() and any in-memory use see
        # the model selected on held-out loss, not the final-epoch state.
        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)
        return self.history

    def save(self, output_dir: str | Path, *, basename: str = "autoencoder") -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(self.model, self.arch, out / "models", basename=basename)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "logs" / "history.json").write_text(self.history.to_json())
        return out
