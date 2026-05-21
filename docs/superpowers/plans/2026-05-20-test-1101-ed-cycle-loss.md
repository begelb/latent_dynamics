# Test 1101 ED-Cycle Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `test_1101` experiment support where the fourth weighted loss enforces `ED ~= I_Z` on predicted latent states `G(E(x_t))`.

**Architecture:** Keep existing three-loss configs backward compatible by allowing `training.loss_weights` length 3 or 4. Compute the fourth component inside `ReconstructionLoss` from already available forward-pass tensors: `loss_cycle_pred = MSE(E(D(z_tau_pred)), z_tau_pred)`. Include the component in history/final losses for all runs, with zero weight when a three-weight config is used.

**Tech Stack:** Python 3, PyTorch, Pydantic v2, pytest, YAML configs, existing `pipeline.py`/Slurm array workflow.

---

### Task 1: Loss Schema and Unit Tests

**Files:**
- Modify: `tests/test_losses.py`
- Modify: `src/latentdynamics/training/losses.py`
- Modify: `src/latentdynamics/models/autoencoder.py`

- [x] **Step 1: Write the failing test**

Add tests asserting that `ReconstructionLoss([1, 1, 0, 1])` includes `loss_cycle_pred`, that three-weight configs keep a zero fourth weight, and that invalid lengths still fail.

- [x] **Step 2: Run test to verify it fails**

Run: `../.venv/bin/python -m pytest tests/test_losses.py -q`

Expected: FAIL because `ForwardPass` does not expose `z_tau_pred_cycle` and `ReconstructionLoss` rejects length 4.

- [x] **Step 3: Write minimal implementation**

Extend `ForwardPass` with `z_tau_pred_cycle`, computed as `encoder(decoder(z_tau_pred))` inside `LatentDynamicsAutoencoder.forward`. Extend `LossBreakdown` and `ReconstructionLoss` to record `loss_cycle_pred`; normalize a three-weight input to `[w1, w2, w3, 0.0]`.

- [x] **Step 4: Run test to verify it passes**

Run: `../.venv/bin/python -m pytest tests/test_losses.py -q`

Expected: PASS.

### Task 2: Config Schema, Training History, and Smoke Tests

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_training_smoke.py`
- Modify: `src/latentdynamics/config/schema.py`
- Modify: `src/latentdynamics/training/trainer.py`
- Modify: `configs/CONFIG_REFERENCE.yaml`

- [x] **Step 1: Write the failing tests**

Add schema tests for accepted four-weight configs and rejected non-3/4 lengths. Update training smoke assertions so history includes `loss_cycle_pred`.

- [x] **Step 2: Run test to verify it fails**

Run: `../.venv/bin/python -m pytest tests/test_config.py tests/test_training_smoke.py -q`

Expected: FAIL on four-weight schema validation and missing `loss_cycle_pred` in history.

- [x] **Step 3: Write minimal implementation**

Allow 3 or 4 loss weights in `TrainingConfig`. Initialize trainer history using the same keys as `LossBreakdown.detach_dict()`, and document `[recon_t, recon_tau, latent_dyn, ED(GE)-GE]` in `CONFIG_REFERENCE.yaml`.

- [x] **Step 4: Run test to verify it passes**

Run: `../.venv/bin/python -m pytest tests/test_config.py tests/test_training_smoke.py -q`

Expected: PASS.

### Task 3: Test 1101 Experiment Configs

**Files:**
- Create: `configs/leslie2d_to_2d_test_1101.yaml`
- Create: `configs/leslie_contraction_test_1101.yaml`
- Create: `configs/leslie3d_spurious_test_1101.yaml`
- Create: `configs/leslie3d_success_test_1101.yaml`
- Create: `configs/chafee_infante_test_1101.yaml`
- Modify: `tests/test_config.py`

- [x] **Step 1: Write the failing config tests**

Parametrize config loading for the five new files and assert `training.loss_weights == [1.0, 1.0, 0.0, 1.0]`, matching paths, system names, dimensions, and key system parameters.

- [x] **Step 2: Run test to verify it fails**

Run: `../.venv/bin/python -m pytest tests/test_config.py::TestLoader::test_test_1101_yamls_enable_ed_cycle_loss -q`

Expected: FAIL because the five YAML files do not exist.

- [x] **Step 3: Add configs**

Copy the corresponding `test_110` experiment patterns and change only the comments, weight vector, and `paths.data_dir`/`paths.output_dir` stems. Use both Leslie 3D parameter sets under explicit names: `leslie3d_spurious_test_1101` and `leslie3d_success_test_1101`.

- [x] **Step 4: Run test to verify it passes**

Run: `../.venv/bin/python -m pytest tests/test_config.py::TestLoader::test_test_1101_yamls_enable_ed_cycle_loss -q`

Expected: PASS.

### Task 4: AMAREL Setup and Local Smoke

**Files:**
- Modify: `docs/AMAREL.md`
- Modify: `README.md`

- [x] **Step 1: Document Slurm-ready commands**

Add a `test_1101` section with data/scale prep and array commands for all five configs. Keep `STAGES` as an exported shell variable so comma parsing is safe.

- [x] **Step 2: Run verification**

Run:

```bash
../.venv/bin/python -m pytest tests/test_losses.py tests/test_config.py tests/test_training_smoke.py -q
../.venv/bin/python pipeline.py --config configs/leslie2d_to_2d_test_1101.yaml --dry-run
../.venv/bin/python pipeline.py --config configs/leslie_contraction_test_1101.yaml --dry-run
../.venv/bin/python pipeline.py --config configs/leslie3d_spurious_test_1101.yaml --dry-run
../.venv/bin/python pipeline.py --config configs/leslie3d_success_test_1101.yaml --dry-run
../.venv/bin/python pipeline.py --config configs/chafee_infante_test_1101.yaml --dry-run
```

Expected: pytest passes, and each dry run reports one `(train_file, seed)` cell.

- [x] **Step 3: Run local Leslie2D smoke**

Run a short local Leslie2D pipeline smoke with a temporary, reduced-epoch config so the full CMGDB/training workload remains reserved for AMAREL. Record the exact command and result in the final response.
