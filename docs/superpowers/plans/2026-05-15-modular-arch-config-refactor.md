# Modular Arch Config Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape the `arch` configuration surface so that encoder / latent-map / decoder hyperparameters (widths, depths, activations) are specified per-component as the primary form, with a shared shortcut as syntactic sugar. Migrate every checked-in config to use the new structure cleanly and behavior-preservingly.

**Architecture:** The schema in `src/latentdynamics/config/schema.py` already carries `ComponentArchConfig` blocks for `encoder`, `latent_map`, `decoder`. The refactor (1) drops the dual top-level `*_out_activation` knobs in favor of putting `out_activation` inside the per-component block; (2) makes shared `arch.num_layers`/`arch.hidden_shape` truly optional (today they are required even when all three components fully override them — see `chafee_infante.yaml`); (3) adds a validator that every component is fully resolvable; (4) migrates every YAML config to the new shape; (5) refreshes docs and tests. Behavior must be bit-identical: the resolved `ResolvedComponentConfig` for every checked-in experiment must match its pre-refactor value.

**Tech Stack:** Python 3.11+, pydantic v2, PyYAML, pytest. No new dependencies.

---

## Observations (pre-refactor state)

These came out of the audit; included so the implementer doesn't have to re-derive them.

- `ArchConfig` already exposes `ComponentArchConfig` per network (`encoder`, `latent_map`, `decoder`) with `num_layers`, `hidden_shape`, `hidden_shapes`, `activation`, `out_activation` overrides (`src/latentdynamics/config/schema.py:87-159`).
- `arch.num_layers` and `arch.hidden_shape` are currently **required** (`int = Field(ge=1)`), even when every component supplies `hidden_shapes`. `configs/chafee_infante.yaml:17-30` carries a now-dead `num_layers: 2 / hidden_shape: 64` because the schema forces it.
- The three top-level `*_out_activation` fields (`encoder_out_activation`, `latent_out_activation`, `decoder_out_activation`) coexist with the nested `out_activation` field on each `ComponentArchConfig`. Only the nested form can vary per network without polluting the top-level namespace; the top-level form is the only one wired into `_shared/defaults.yaml`.
- Downstream consumers (`models/autoencoder.py`, `pipeline.py`, `reproduce_paper.py`, tests) only read `arch.component(name).hidden_shapes / activation / out_activation`. The top-level `num_layers`, `hidden_shape`, `*_out_activation` fields are not referenced outside `schema.py` itself.
- Existing configs split as follows:
  - **flat shortcut only:** `coral_basic.yaml`, `coral_adaptive.yaml`, `coral_data_scaling.yaml`, `leslie2d_to_2d.yaml`, `leslie2d_to_2d_smoke.yaml`, `leslie3d.yaml`, `leslie3d_spurious.yaml`, `leslie3d_success.yaml`, `leslie_contraction.yaml`, plus their `scratch/` duplicates.
  - **per-component (already on target shape):** `chafee_infante.yaml`.
- `_shared/defaults.yaml` provides `arch.activation: relu`, `arch.encoder_out_activation: tanh`, `arch.decoder_out_activation: sigmoid`. There is no shared `arch.latent_out_activation`; the schema's class-level default (`tanh`) carries it.

## File Structure

**Modify:**
- `src/latentdynamics/config/schema.py` — make shared fields optional, fold out_activation into components, tighten validation.
- `configs/_shared/defaults.yaml` — move `*_out_activation` into per-component blocks.
- `configs/coral_basic.yaml`
- `configs/coral_adaptive.yaml`
- `configs/coral_data_scaling.yaml`
- `configs/leslie2d_to_2d.yaml`
- `configs/leslie2d_to_2d_smoke.yaml`
- `configs/leslie3d.yaml`
- `configs/leslie3d_spurious.yaml`
- `configs/leslie3d_success.yaml`
- `configs/leslie_contraction.yaml`
- `configs/chafee_infante.yaml` — drop the dead shared `num_layers`/`hidden_shape`.
- `configs/scratch/coral_basic.yaml`
- `configs/scratch/coral_adaptive.yaml`
- `configs/scratch/coral_data_scaling.yaml`
- `tests/test_config.py` — add coverage for the new validator and the optional shared fields.
- `docs/PAPER_REPRODUCTION.md` — refresh the "Modular hyperparameters" section.

**Create:**
- `tests/test_config_migration.py` — golden-file test that the resolved `ResolvedComponentConfig` for every checked-in config matches a pinned snapshot.
- `configs/scratch/asymmetric_example.yaml` — small worked example showing a wider encoder than decoder, and per-network activation choices.

---

## Task 1: Pin resolved-arch snapshots for every checked-in config

**Files:**
- Create: `tests/test_config_migration.py`

- [ ] **Step 1: Write the failing test**

```python
"""Golden-file equivalence test for the arch refactor.

Goal: every YAML under ``configs/`` must resolve to the same
``ResolvedComponentConfig`` triple before and after the refactor. The
pinned dictionary below is the pre-refactor snapshot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from latentdynamics.config.loader import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# Pre-refactor expected resolution. Keys are paths relative to ``configs/``.
EXPECTED: dict[str, dict[str, dict[str, object]]] = {
    "coral_basic.yaml": {
        "encoder": {"hidden_shapes": (64, 64, 64), "activation": "relu", "out_activation": "tanh"},
        "latent_map": {"hidden_shapes": (64, 64, 64), "activation": "relu", "out_activation": "tanh"},
        "decoder": {"hidden_shapes": (64, 64, 64), "activation": "relu", "out_activation": "sigmoid"},
    },
    "coral_adaptive.yaml": {
        "encoder": {"hidden_shapes": (64, 64, 64), "activation": "relu", "out_activation": "tanh"},
        "latent_map": {"hidden_shapes": (64, 64, 64), "activation": "relu", "out_activation": "tanh"},
        "decoder": {"hidden_shapes": (64, 64, 64), "activation": "relu", "out_activation": "sigmoid"},
    },
    "coral_data_scaling.yaml": {
        "encoder": {"hidden_shapes": (64, 64, 64), "activation": "relu", "out_activation": "tanh"},
        "latent_map": {"hidden_shapes": (64, 64, 64), "activation": "relu", "out_activation": "tanh"},
        "decoder": {"hidden_shapes": (64, 64, 64), "activation": "relu", "out_activation": "sigmoid"},
    },
    "leslie2d_to_2d.yaml": None,        # fill in Step 2 by reading the YAML
    "leslie2d_to_2d_smoke.yaml": None,
    "leslie3d.yaml": {
        "encoder": {"hidden_shapes": (32, 32, 32), "activation": "relu", "out_activation": "tanh"},
        "latent_map": {"hidden_shapes": (32, 32, 32), "activation": "relu", "out_activation": "tanh"},
        "decoder": {"hidden_shapes": (32, 32, 32), "activation": "relu", "out_activation": "sigmoid"},
    },
    "leslie3d_spurious.yaml": None,
    "leslie3d_success.yaml": None,
    "leslie_contraction.yaml": None,
    "chafee_infante.yaml": {
        "encoder": {"hidden_shapes": (64, 32), "activation": "tanh", "out_activation": "none"},
        "latent_map": {"hidden_shapes": (32, 32), "activation": "tanh", "out_activation": "none"},
        "decoder": {"hidden_shapes": (32, 64), "activation": "tanh", "out_activation": "none"},
    },
    "scratch/coral_basic.yaml": None,
    "scratch/coral_adaptive.yaml": None,
    "scratch/coral_data_scaling.yaml": None,
}


@pytest.mark.parametrize("rel_path", sorted(EXPECTED.keys()))
def test_resolved_arch_matches_snapshot(rel_path: str) -> None:
    expected = EXPECTED[rel_path]
    assert expected is not None, (
        f"snapshot for {rel_path} not yet pinned; fill in Step 2 of Task 1"
    )
    cfg = load_config(CONFIG_DIR / rel_path)
    for name in ("encoder", "latent_map", "decoder"):
        resolved = cfg.arch.component(name)
        assert resolved.hidden_shapes == expected[name]["hidden_shapes"], name
        assert resolved.activation == expected[name]["activation"], name
        assert resolved.out_activation == expected[name]["out_activation"], name
```

- [ ] **Step 2: Fill in every `None` snapshot by running `load_config` against the unmodified YAML**

Run a one-off Python REPL (do not commit this script):
```bash
cd /Users/bdoprad/Work/Projects/latent-dynamics/code
python -c "
from pathlib import Path
from latentdynamics.config.loader import load_config
for p in Path('configs').rglob('*.yaml'):
    if p.parent.name == '_shared':
        continue
    cfg = load_config(p)
    print(p.relative_to('configs'))
    for n in ('encoder','latent_map','decoder'):
        c = cfg.arch.component(n)
        print(f'  {n}: hidden_shapes={c.hidden_shapes}, activation={c.activation!r}, out_activation={c.out_activation!r}')
"
```

Paste each result into `EXPECTED` in `test_config_migration.py`, replacing every `None`.

- [ ] **Step 3: Run test to verify it passes against the unmodified codebase**

Run: `pytest tests/test_config_migration.py -v`
Expected: all parametrize cases PASS (this is the pre-refactor baseline; if any fail, the snapshot is wrong, fix the snapshot — do not change source).

- [ ] **Step 4: Commit**

```bash
git add tests/test_config_migration.py
git commit -m "test(config): pin resolved-arch snapshot for every checked-in config"
```

---

## Task 2: Make shared `arch.num_layers` and `arch.hidden_shape` optional

**Files:**
- Modify: `src/latentdynamics/config/schema.py:87-129`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:
```python
def test_arch_accepts_per_component_only(tmp_path):
    """When every component supplies hidden_shapes, shared num_layers/hidden_shape
    must not be required."""
    from latentdynamics.config.schema import ArchConfig
    arch = ArchConfig.model_validate({
        "high_dims": 4,
        "low_dims": 2,
        "encoder": {"hidden_shapes": [16, 8]},
        "latent_map": {"hidden_shapes": [8, 8]},
        "decoder": {"hidden_shapes": [8, 16]},
    })
    assert arch.component("encoder").hidden_shapes == (16, 8)
    assert arch.component("latent_map").hidden_shapes == (8, 8)
    assert arch.component("decoder").hidden_shapes == (8, 16)


def test_arch_rejects_unresolvable_component():
    """If a component is missing hidden_shapes and there is no shared
    num_layers/hidden_shape, validation must fail with a clear message."""
    from latentdynamics.config.schema import ArchConfig
    import pytest
    with pytest.raises(ValueError, match="encoder.*unresolvable"):
        ArchConfig.model_validate({
            "high_dims": 4,
            "low_dims": 2,
            "latent_map": {"hidden_shapes": [8, 8]},
            "decoder": {"hidden_shapes": [8, 16]},
        })
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py::test_arch_accepts_per_component_only tests/test_config.py::test_arch_rejects_unresolvable_component -v`
Expected: both FAIL — the first with a pydantic `Field required` error on `num_layers`, the second similarly.

- [ ] **Step 3: Edit `src/latentdynamics/config/schema.py`**

Change `ArchConfig` lines 90-91 from:
```python
    num_layers: int = Field(ge=1)
    hidden_shape: int = Field(ge=1)
```
to:
```python
    num_layers: int | None = Field(default=None, ge=1)
    hidden_shape: int | None = Field(default=None, ge=1)
```

Update `_consistent_shared_layer_count` (lines 125-129) to gate the check on both being set:
```python
    @model_validator(mode="after")
    def _consistent_shared_layer_count(self) -> ArchConfig:
        if (
            self.hidden_shapes is not None
            and self.num_layers is not None
            and len(self.hidden_shapes) != self.num_layers
        ):
            raise ValueError("num_layers must match len(hidden_shapes)")
        return self
```

Add a new model validator immediately after it:
```python
    @model_validator(mode="after")
    def _every_component_resolvable(self) -> ArchConfig:
        """Each of encoder/latent_map/decoder must resolve to a concrete
        hidden_shapes tuple via either (a) its own hidden_shapes, (b) its
        own num_layers + hidden_shape, or (c) shared num_layers + hidden_shape."""
        for name in ("encoder", "latent_map", "decoder"):
            override: ComponentArchConfig = getattr(self, name)
            if override.hidden_shapes is not None:
                continue
            n = override.num_layers if override.num_layers is not None else self.num_layers
            w = override.hidden_shape if override.hidden_shape is not None else self.hidden_shape
            if self.hidden_shapes is not None and override.num_layers is None and override.hidden_shape is None:
                continue
            if n is None or w is None:
                raise ValueError(
                    f"{name} is unresolvable: supply arch.{name}.hidden_shapes, "
                    f"arch.{name}.num_layers + arch.{name}.hidden_shape, or shared "
                    f"arch.num_layers + arch.hidden_shape"
                )
        return self
```

Update `ArchConfig.component` (lines 131-159) to tolerate `None` shared values when the component supplies its own override:
```python
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
            hidden_shape = override.hidden_shape if override.hidden_shape is not None else self.hidden_shape
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
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `pytest tests/test_config.py::test_arch_accepts_per_component_only tests/test_config.py::test_arch_rejects_unresolvable_component -v`
Expected: both PASS.

- [ ] **Step 5: Run the snapshot test to confirm behavior preservation**

Run: `pytest tests/test_config_migration.py -v`
Expected: all parametrize cases PASS (no resolved arch changed).

- [ ] **Step 6: Commit**

```bash
git add src/latentdynamics/config/schema.py tests/test_config.py
git commit -m "feat(config): make shared arch.num_layers/hidden_shape optional"
```

---

## Task 3: Drop dead shared-arch fields from `chafee_infante.yaml`

**Files:**
- Modify: `configs/chafee_infante.yaml:17-18`

- [ ] **Step 1: Edit the file**

Delete lines 17-18 (`num_layers: 2` and `hidden_shape: 64`). The file should read:
```yaml
arch:
  # Mirrors archive/marcio/scripts/autoencoder_model.py:
  # encoder 64->64->32->2, latent map 2->32->32->2, decoder 2->32->64->64.
  high_dims: 64
  low_dims: 2
  activation: tanh
  encoder_out_activation: none
  latent_out_activation: none
  decoder_out_activation: none
  encoder:
    hidden_shapes: [64, 32]
  latent_map:
    hidden_shapes: [32, 32]
  decoder:
    hidden_shapes: [32, 64]
```

- [ ] **Step 2: Run the snapshot test**

Run: `pytest tests/test_config_migration.py::test_resolved_arch_matches_snapshot[chafee_infante.yaml] -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add configs/chafee_infante.yaml
git commit -m "refactor(configs): drop dead shared num_layers/hidden_shape from chafee_infante"
```

---

## Task 4: Move `*_out_activation` into per-component blocks (schema)

**Files:**
- Modify: `src/latentdynamics/config/schema.py:87-159`
- Modify: `tests/test_config.py`

Decision: keep the top-level `encoder_out_activation`/`latent_out_activation`/`decoder_out_activation` fields as **deprecated aliases** in the schema for one cycle so paper-replay configs do not break mid-refactor. Migration of YAML files to the nested form happens in Task 6.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:
```python
def test_nested_out_activation_overrides_top_level():
    from latentdynamics.config.schema import ArchConfig
    arch = ArchConfig.model_validate({
        "num_layers": 2,
        "hidden_shape": 8,
        "high_dims": 4,
        "low_dims": 2,
        "encoder_out_activation": "tanh",         # top-level legacy
        "encoder": {"out_activation": "none"},    # nested wins
    })
    assert arch.component("encoder").out_activation == "none"
    assert arch.component("decoder").out_activation == "sigmoid"  # class default
```

- [ ] **Step 2: Verify it already passes against current code**

Run: `pytest tests/test_config.py::test_nested_out_activation_overrides_top_level -v`
Expected: PASS (the existing resolver already prefers `override.out_activation`).

This task is mostly belt-and-suspenders: the behavior already exists, we just lock it down with a test. No schema edit is needed here. Move on.

- [ ] **Step 3: Commit**

```bash
git add tests/test_config.py
git commit -m "test(config): lock down nested-over-top-level out_activation precedence"
```

---

## Task 5: Update `_shared/defaults.yaml` to use the per-component form

**Files:**
- Modify: `configs/_shared/defaults.yaml`

- [ ] **Step 1: Replace the `arch:` block**

Current:
```yaml
arch:
  activation: relu
  encoder_out_activation: tanh
  decoder_out_activation: sigmoid
```

Target:
```yaml
arch:
  activation: relu
  encoder:
    out_activation: tanh
  latent_map:
    out_activation: tanh
  decoder:
    out_activation: sigmoid
```

- [ ] **Step 2: Run the full snapshot test**

Run: `pytest tests/test_config_migration.py -v`
Expected: every parametrize case PASS.

- [ ] **Step 3: Commit**

```bash
git add configs/_shared/defaults.yaml
git commit -m "refactor(configs): move shared out_activation defaults into per-component blocks"
```

---

## Task 6: Migrate every experiment config to the per-component form

**Files:**
- Modify: `configs/coral_basic.yaml`
- Modify: `configs/coral_adaptive.yaml`
- Modify: `configs/coral_data_scaling.yaml`
- Modify: `configs/leslie2d_to_2d.yaml`
- Modify: `configs/leslie2d_to_2d_smoke.yaml`
- Modify: `configs/leslie3d.yaml`
- Modify: `configs/leslie3d_spurious.yaml`
- Modify: `configs/leslie3d_success.yaml`
- Modify: `configs/leslie_contraction.yaml`
- Modify: `configs/scratch/coral_basic.yaml`
- Modify: `configs/scratch/coral_adaptive.yaml`
- Modify: `configs/scratch/coral_data_scaling.yaml`

Migration policy (apply mechanically to every file in this task):
1. If the config uses `arch.num_layers` + `arch.hidden_shape` and intends all three networks to share that shape, **leave the shared fields in place** — they are useful syntactic sugar. The migration only requires that each network be expressible per-component if needed; symmetric configs should stay terse.
2. If the config sets any top-level `*_out_activation`, move it into the matching component block. (None of the listed configs currently override the top-level out_activation, so this is a no-op for them, but apply consistently.)
3. After migration, no top-level `encoder_out_activation` / `latent_out_activation` / `decoder_out_activation` should remain in any experiment YAML.

Concretely, the listed configs are all symmetric and never override out_activation, so this task is mostly a **review + sweep**: confirm there is no `*_out_activation` at top level in any of them, and leave the `arch.num_layers` / `arch.hidden_shape` shortcut intact.

- [ ] **Step 1: Audit each file**

Run:
```bash
grep -nE 'encoder_out_activation|latent_out_activation|decoder_out_activation' configs/*.yaml configs/scratch/*.yaml
```
Expected: no matches (the only legitimate occurrence was in `chafee_infante.yaml`, which is already on the per-component form).

- [ ] **Step 2: Run the full snapshot test**

Run: `pytest tests/test_config_migration.py -v`
Expected: all cases PASS.

- [ ] **Step 3: No commit if no edits were needed**

Skip the commit if Step 1 reported zero matches. Otherwise commit:
```bash
git add configs/*.yaml configs/scratch/*.yaml
git commit -m "refactor(configs): fold top-level out_activation into per-component blocks"
```

---

## Task 7: Add an asymmetric-architecture example

**Files:**
- Create: `configs/scratch/asymmetric_example.yaml`

Purpose: provide a worked, runnable example that demonstrates per-component overrides for every knob the refactor exposes, so future contributors have a template.

- [ ] **Step 1: Write the file**

```yaml
# Asymmetric architecture example. Mirrors the leslie3d setup but with:
#   - wider encoder than decoder (16,16 vs 8,8)
#   - shallower latent map (1 hidden layer)
#   - encoder uses tanh activation, latent map relu, decoder gelu
#   - decoder out_activation overridden to tanh
# This file is documentation-by-example; the resolved networks must train.
system:
  name: leslie3d
  params:
    th1: 28.9
    th2: 29.8
    th3: 22.0
    survival_p1: 0.7
    survival_p2: 0.7

arch:
  high_dims: 3
  low_dims: 2
  encoder:
    hidden_shapes: [16, 16]
    activation: tanh
  latent_map:
    hidden_shapes: [32]
    activation: relu
  decoder:
    hidden_shapes: [8, 8]
    activation: gelu
    out_activation: tanh

training:
  loss_weights: [10, 10, 1]

data:
  sampling_method: uniform
  n_samples_train: 2000
  n_samples_test: 5000
  n_iterations: 30

paths:
  data_dir: data/leslie3d
  output_dir: output/scratch/asymmetric_example
```

- [ ] **Step 2: Confirm it loads**

Run:
```bash
cd /Users/bdoprad/Work/Projects/latent-dynamics/code
python -c "
from latentdynamics.config.loader import load_config
cfg = load_config('configs/scratch/asymmetric_example.yaml')
for n in ('encoder','latent_map','decoder'):
    print(n, cfg.arch.component(n))
"
```
Expected output (exact):
```
encoder ResolvedComponentConfig(hidden_shapes=(16, 16), activation='tanh', out_activation='tanh')
latent_map ResolvedComponentConfig(hidden_shapes=(32,), activation='relu', out_activation='tanh')
decoder ResolvedComponentConfig(hidden_shapes=(8, 8), activation='gelu', out_activation='tanh')
```

- [ ] **Step 3: Confirm the autoencoder builds**

Run:
```bash
python -c "
from latentdynamics.config.loader import load_config
from latentdynamics.models.autoencoder import build_autoencoder
cfg = load_config('configs/scratch/asymmetric_example.yaml')
ae = build_autoencoder(cfg.arch)
print(ae)
"
```
Expected: a printable `LatentDynamicsAutoencoder` with the encoder having two `Linear(*, 16)` layers, the latent map having one `Linear(*, 32)`, and the decoder having two `Linear(*, 8)` layers.

- [ ] **Step 4: Commit**

```bash
git add configs/scratch/asymmetric_example.yaml
git commit -m "docs(configs): add asymmetric-architecture example under scratch/"
```

---

## Task 8: Refresh `docs/PAPER_REPRODUCTION.md`

**Files:**
- Modify: `docs/PAPER_REPRODUCTION.md:135-165`

- [ ] **Step 1: Replace the "Modular hyperparameters" section**

Replace lines 135-165 with:
````markdown
## Modular hyperparameters

The `arch` block specifies hyperparameters per network (`encoder`, `latent_map`, `decoder`). Shared values at the top level act as defaults that any component may override.

Most paper configs are symmetric — all three networks share width, depth, and activation — and use the terse shortcut form:

```yaml
arch:
  num_layers: 3
  hidden_shape: 64
  high_dims: 13
  low_dims: 1
```

Asymmetric architectures, or per-network activation choices, use the per-component form. Each component block accepts `hidden_shapes`, `num_layers` + `hidden_shape`, `activation`, and `out_activation`:

```yaml
arch:
  high_dims: 64
  low_dims: 2
  activation: tanh                  # shared default
  encoder:
    hidden_shapes: [64, 32]         # asymmetric widths
    out_activation: none
  latent_map:
    hidden_shapes: [32, 32]
    activation: relu                # override shared default
    out_activation: none
  decoder:
    hidden_shapes: [32, 64]
    out_activation: none
```

Shared `arch.num_layers` / `arch.hidden_shape` are optional. If every component supplies its own `hidden_shapes`, the shared fields may be omitted entirely (see `configs/chafee_infante.yaml`). If any component lacks an explicit width specification, either provide it per-component or fall back to the shared fields.

A worked example with three different activations and asymmetric widths lives at `configs/scratch/asymmetric_example.yaml`.

Training hyperparameters live under `training`. Data and CMGDB settings are unchanged by this refactor.
````

- [ ] **Step 2: Commit**

```bash
git add docs/PAPER_REPRODUCTION.md
git commit -m "docs: refresh modular-hyperparameter reference for per-component arch"
```

---

## Task 9: Run the full test suite

- [ ] **Step 1: Run pytest**

Run: `pytest -x`
Expected: every test in `tests/` passes. If anything fails, the failure is the refactor's responsibility — fix it before moving on.

- [ ] **Step 2: Spot-check that `pipeline.py` still loads every config**

Run:
```bash
for cfg in configs/*.yaml configs/scratch/*.yaml; do
  python -c "from latentdynamics.config.loader import load_config; load_config('$cfg')" || echo "FAIL: $cfg"
done
```
Expected: no `FAIL:` lines.

- [ ] **Step 3: No commit (verification only)**

---

## Out of scope (deliberately deferred)

These came up during planning and are **not** part of this refactor. If the user wants any of them, treat each as a separate plan.

1. **Per-component dropout / layer-norm / residual connections.** The schema can grow these knobs once a concrete experiment needs them. YAGNI today.
2. **Per-layer activation (different activation in layer 1 vs layer 2 of the same network).** The current `_build_mlp` uses one activation across all hidden layers; relaxing this is a bigger architectural change.
3. **Config-driven hyperparameter sweeps.** A sweep harness on top of `load_config` is a useful next step but is not "refactoring the config files" per the user's ask.
4. **Eliminating the top-level `*_out_activation` aliases entirely.** Kept for one cycle to avoid churn; can be removed in a follow-up once `chafee_infante.yaml` and any external configs migrate.

---

## Self-review notes

- **Spec coverage:** widths ✓ (Task 1, 6, 7), depths ✓ (same), per-network split ✓ (Task 1-7), per-network activation ✓ (Task 4, 7), per-network out_activation ✓ (Task 4, 5).
- **Placeholder scan:** every code/edit step has the actual content; no "TBD" or "similar to above".
- **Type consistency:** `ResolvedComponentConfig`, `ComponentArchConfig`, `ArchConfig.component(name)` used uniformly; the new validator name `_every_component_resolvable` does not collide.
- **Behavior preservation:** Task 1's golden-file snapshot is the safety net; Tasks 2-6 each re-run it.
