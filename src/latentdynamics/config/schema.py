"""Typed configuration schema for paper experiments (pydantic v2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Activation = Literal["relu", "tanh", "gelu"]
TerminalActivation = Literal["tanh", "sigmoid", "none"]
SystemName = Literal[
    "leslie_contraction",
    "leslie3d",
    "leslie4d",
    "coral",
    "chafee_infante",
]
SamplingMethod = Literal["uniform", "sobol", "adaptive"]
ScalingMethod = Literal["minmax", "none"]
BoxMapBackend = Literal[
    "auto",
    "pytorch",
    "numpy",
    "uniform_precomputed",
    "adaptive_precomputed",
]


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: SystemName
    params: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedComponentConfig:
    """Concrete MLP settings for one network component."""

    hidden_shapes: tuple[int, ...]
    activation: Activation
    out_activation: TerminalActivation

    @property
    def num_layers(self) -> int:
        return len(self.hidden_shapes)


class ComponentArchConfig(BaseModel):
    """Optional per-component architecture override.

    ``hidden_shapes`` is the most explicit form and supports asymmetric MLPs
    such as ``[64, 32]``. ``num_layers`` + ``hidden_shape`` remains available
    for repeated-width MLPs.
    """

    model_config = ConfigDict(extra="forbid")

    num_layers: int | None = Field(default=None, ge=1)
    hidden_shape: int | None = Field(default=None, ge=1)
    hidden_shapes: list[int] | None = Field(default=None, min_length=1)
    activation: Activation | None = None
    out_activation: TerminalActivation | None = None

    @field_validator("activation", mode="before")
    @classmethod
    def _lowercase_activation(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v

    @field_validator("out_activation", mode="before")
    @classmethod
    def _lowercase_terminal(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v

    @field_validator("hidden_shapes")
    @classmethod
    def _positive_hidden_shapes(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and any(width < 1 for width in v):
            raise ValueError("hidden_shapes entries must be positive")
        return v

    @model_validator(mode="after")
    def _consistent_layer_count(self) -> ComponentArchConfig:
        if self.hidden_shapes is not None and self.num_layers is not None:
            if len(self.hidden_shapes) != self.num_layers:
                raise ValueError("num_layers must match len(hidden_shapes)")
        return self


class ArchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_layers: int | None = Field(default=None, ge=1)
    hidden_shape: int | None = Field(default=None, ge=1)
    hidden_shapes: list[int] | None = Field(default=None, min_length=1)
    high_dims: int = Field(ge=1)
    low_dims: int = Field(ge=1)
    activation: Activation = "relu"
    encoder_out_activation: TerminalActivation = "tanh"
    latent_out_activation: TerminalActivation = "tanh"
    decoder_out_activation: TerminalActivation = "sigmoid"
    encoder: ComponentArchConfig = Field(default_factory=ComponentArchConfig)
    latent_map: ComponentArchConfig = Field(default_factory=ComponentArchConfig)
    decoder: ComponentArchConfig = Field(default_factory=ComponentArchConfig)

    @field_validator("activation", mode="before")
    @classmethod
    def _lowercase_activation(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v

    @field_validator(
        "encoder_out_activation",
        "latent_out_activation",
        "decoder_out_activation",
        mode="before",
    )
    @classmethod
    def _lowercase_terminal(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v

    @field_validator("hidden_shapes")
    @classmethod
    def _positive_hidden_shapes(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and any(width < 1 for width in v):
            raise ValueError("hidden_shapes entries must be positive")
        return v

    @model_validator(mode="after")
    def _consistent_shared_layer_count(self) -> ArchConfig:
        if (
            self.hidden_shapes is not None
            and self.num_layers is not None
            and len(self.hidden_shapes) != self.num_layers
        ):
            raise ValueError("num_layers must match len(hidden_shapes)")
        return self

    @model_validator(mode="after")
    def _every_component_resolvable(self) -> ArchConfig:
        """Each of encoder/latent_map/decoder must resolve to a concrete
        hidden_shapes tuple via one of four paths:
          (a) component's own hidden_shapes,
          (b) component's own num_layers + hidden_shape,
          (c) shared arch.hidden_shapes (only when component sets neither
              num_layers nor hidden_shape),
          (d) shared arch.num_layers + arch.hidden_shape."""
        for name in ("encoder", "latent_map", "decoder"):
            override: ComponentArchConfig = getattr(self, name)
            if override.hidden_shapes is not None:
                continue
            if (
                self.hidden_shapes is not None
                and override.num_layers is None
                and override.hidden_shape is None
            ):
                continue
            effective_layers = (
                override.num_layers if override.num_layers is not None else self.num_layers
            )
            effective_width = (
                override.hidden_shape if override.hidden_shape is not None else self.hidden_shape
            )
            if effective_layers is None or effective_width is None:
                raise ValueError(
                    f"{name} is unresolvable: supply arch.{name}.hidden_shapes, "
                    f"arch.{name}.num_layers + arch.{name}.hidden_shape, shared "
                    f"arch.hidden_shapes, or shared arch.num_layers + arch.hidden_shape"
                )
        return self

    def component(
        self, name: Literal["encoder", "latent_map", "decoder"]
    ) -> ResolvedComponentConfig:
        """Resolve shared defaults plus per-component overrides."""
        override: ComponentArchConfig = getattr(self, name)
        if override.hidden_shapes is not None:
            hidden_shapes = tuple(int(width) for width in override.hidden_shapes)
        elif (
            self.hidden_shapes is not None
            and override.num_layers is None
            and override.hidden_shape is None
        ):
            hidden_shapes = tuple(int(width) for width in self.hidden_shapes)
        else:
            num_layers = override.num_layers if override.num_layers is not None else self.num_layers
            hidden_shape = (
                override.hidden_shape if override.hidden_shape is not None else self.hidden_shape
            )
            assert num_layers is not None and hidden_shape is not None  # validator guarantees this
            hidden_shapes = tuple(int(hidden_shape) for _ in range(int(num_layers)))

        default_out = {
            "encoder": self.encoder_out_activation,
            "latent_map": self.latent_out_activation,
            "decoder": self.decoder_out_activation,
        }[name]
        return ResolvedComponentConfig(
            hidden_shapes=hidden_shapes,
            activation=override.activation or self.activation,
            out_activation=override.out_activation or default_out,
        )


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(gt=0)
    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    patience: int = Field(ge=1)
    lr_patience: int = Field(default=10, ge=1)
    loss_weights: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    gradient_clip_norm: float | None = 1.0
    scheduler_factor: float = Field(default=0.1, gt=0.0, lt=1.0)
    scheduler_threshold: float = Field(default=1e-3, ge=0.0)
    scheduler_min_lr: float = Field(default=0.0, ge=0.0)

    @field_validator("loss_weights")
    @classmethod
    def _three_weights(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("loss_weights must have length 3 (recon_t, recon_tau, dyn)")
        return v

    @model_validator(mode="after")
    def _lr_patience_below_early_stop(self) -> TrainingConfig:
        # ReduceLROnPlateau.patience and early-stop patience must differ;
        # otherwise the LR drop and the early-stop break fire on the same
        # epoch and the scheduler can never actually act.
        if self.lr_patience >= self.patience:
            raise ValueError(
                f"lr_patience ({self.lr_patience}) must be strictly less than "
                f"patience ({self.patience}); otherwise the scheduler never "
                f"gets to lower the LR before early stopping ends training"
            )
        return self

    @field_validator("gradient_clip_norm")
    @classmethod
    def _positive_clip_norm(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("gradient_clip_norm must be positive or null")
        return v


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sampling_method: SamplingMethod
    scaling: ScalingMethod = "minmax"
    n_samples_train: int | list[int]
    n_samples_val: int = Field(ge=1)
    n_iterations: int = Field(ge=1)
    skip: int = Field(ge=0, default=0)
    sobol_train_seed: int = 42
    sobol_val_seed: int = 9999
    # When set, takes precedence over auto-derivation from ``n_samples_train``.
    # Required for non-numeric labels such as adaptive sweeps where the train
    # file basenames are e.g. ``train_500_300_adaptive``.
    train_files: list[str] | None = None


class CMGDBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subdiv_init: int = Field(ge=1, default=6)
    subdiv_min: int = Field(ge=1, default=8)
    subdiv_max: int = Field(ge=1, default=10)
    subdiv_limit: int = Field(ge=1, default=10000)
    bounds_epsilon_frac: float = Field(ge=0.0, default=0.01)
    lower_bounds: list[float] | None = None
    upper_bounds: list[float] | None = None
    padding: bool = True
    box_map_backend: BoxMapBackend = "auto"
    max_table_points: int = Field(
        ge=1,
        default=10_000_000,
        description=(
            "Hard cap on the number of corner points in precomputed-backend "
            "lattices. Applied to both 'uniform_precomputed' and "
            "'adaptive_precomputed'. Raise if you hit the cap; reduce if you "
            "want to bound memory."
        ),
    )
    # Forward-pass chunk size for precomputed backends. ``max_table_points``
    # bounds the persisted float64 table; this bounds the transient float32
    # activation buffers when evaluating ``latent_map`` across the lattice.
    # ``"auto"`` picks a device- and architecture-aware chunk; a positive int
    # is honored as-is (clamped to the table size). Required for clusters /
    # MPS where single-allocation buffer caps are smaller than total RAM.
    precompute_batch_points: int | Literal["auto"] = "auto"

    @field_validator("precompute_batch_points", mode="before")
    @classmethod
    def _validate_precompute_batch_points(cls, v: Any) -> Any:
        if isinstance(v, str):
            if v != "auto":
                raise ValueError(
                    f"precompute_batch_points must be a positive int or 'auto'; got {v!r}"
                )
            return v
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"precompute_batch_points must be a positive int or 'auto'; got {v!r}"
            )
        if v <= 0:
            raise ValueError(
                f"precompute_batch_points must be positive when an int; got {v}"
            )
        return v

    @model_validator(mode="after")
    def _ordered_subdivs(self) -> CMGDBConfig:
        if not (self.subdiv_init <= self.subdiv_min <= self.subdiv_max):
            raise ValueError("require subdiv_init <= subdiv_min <= subdiv_max")
        if (self.lower_bounds is None) != (self.upper_bounds is None):
            raise ValueError("lower_bounds and upper_bounds must be set together")
        if self.lower_bounds is not None and self.upper_bounds is not None:
            if len(self.lower_bounds) != len(self.upper_bounds):
                raise ValueError("lower_bounds and upper_bounds must have the same length")
            if any(lo >= hi for lo, hi in zip(self.lower_bounds, self.upper_bounds, strict=True)):
                raise ValueError("each lower bound must be strictly less than its upper bound")
        if self.box_map_backend == "uniform_precomputed":
            if not (self.subdiv_init == self.subdiv_min == self.subdiv_max):
                raise ValueError(
                    "box_map_backend='uniform_precomputed' requires uniform mode "
                    "(subdiv_init == subdiv_min == subdiv_max)"
                )
        return self


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path
    output_dir: Path
    scaler_dir_override: Path | None = None
    flat_scaler: bool = False
    read_only: bool = False

    @property
    def model_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def scaler_dir(self) -> Path:
        return (
            self.scaler_dir_override
            if self.scaler_dir_override is not None
            else self.output_dir / "scalers"
        )

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def morse_dir(self) -> Path:
        return self.output_dir / "MG"

    def scaler_path(self, train_file: str) -> Path:
        """Resolve the scaler file location for ``train_file``.

        Standard layout: ``<scaler_dir>/<train_file>/scaler.gz``.
        Legacy flat layout (single shared scaler): ``<scaler_dir>/scaler.gz``,
        triggered by ``flat_scaler: true`` in the config.
        """
        if self.flat_scaler:
            return self.scaler_dir / "scaler.gz"
        return self.scaler_dir / train_file / "scaler.gz"

    def val_csv(self) -> Path:
        """Validation set CSV path. Prefers ``val.csv`` (current name); falls
        back to legacy ``test.csv`` if only the old name is on disk
        (preserved paper artifacts). Returns the canonical ``val.csv`` target
        when neither file exists yet, so writers always emit the new name."""
        val_path = self.data_dir / "val.csv"
        if val_path.exists():
            return val_path
        legacy = self.data_dir / "test.csv"
        if legacy.exists():
            return legacy
        return val_path

    def val_metadata(self) -> Path:
        """Validation set metadata JSON; same legacy fallback as ``val_csv``."""
        val_path = self.data_dir / "val_metadata.json"
        if val_path.exists():
            return val_path
        legacy = self.data_dir / "test_metadata.json"
        if legacy.exists():
            return legacy
        return val_path


class ExperimentConfig(BaseModel):
    """Top-level configuration for one paper experiment (one figure)."""

    model_config = ConfigDict(extra="forbid")

    system: SystemConfig
    arch: ArchConfig
    training: TrainingConfig
    data: DataConfig
    cmgdb: CMGDBConfig = Field(default_factory=CMGDBConfig)
    paths: PathsConfig
    seeds: list[int] = Field(default_factory=lambda: [0])
    # Stable, human-readable id for this experiment. The loader populates this
    # with the YAML file's ``Path.stem`` when it is not explicitly set in YAML;
    # downstream code uses it to compute replay output roots
    # (``output/replay/<experiment_name>/...``) and to label run manifests.
    experiment_name: str | None = None

    @model_validator(mode="after")
    def _arch_dims_match_system(self) -> ExperimentConfig:
        if self.arch.high_dims < self.arch.low_dims:
            raise ValueError("arch.high_dims must be >= arch.low_dims")
        if (
            self.cmgdb.lower_bounds is not None
            and len(self.cmgdb.lower_bounds) != self.arch.low_dims
        ):
            raise ValueError("cmgdb fixed bounds must match arch.low_dims")
        return self
