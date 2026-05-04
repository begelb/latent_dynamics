# fig_leslie_contraction

Paper Fig. 1.83: 2D Leslie map padded with eight contracting tail dimensions, demonstrating that the encoder discovers the 2D essential dynamics from a 10D phase space.

## Paper figures

- `paper/figures/LeslieContraction10D.png`
- `paper/figures/LeslieContraction_10D_10kData_Mgraph.png`
- `paper/figures/LeslieContraction_10D_10kData_Msets.png`

## Source of paper run

Patrick, using Brittany's training infrastructure. **The training script and trained checkpoint are NOT present in any archived tree** (searched `archive/brittany`, `archive/marcio`, `archive/old_code`, `archive/old_paper`). The configuration is reconstructed from the paper text and from the `LeslieContraction` class in `latentdynamics.systems.leslie`.

- training script:    MISSING (originated with Patrick)
- CMGDB script:       MISSING; subdiv values `28/29` are quoted from the paper Sec. 1.83
- legacy config:      none
- system definition:  `code/src/latentdynamics/systems/leslie.py:13-53` (`LeslieContraction` class)

## Status

**scratch-only, with hyperparameter gaps to fill**. Current YAML treats this as a 2D Leslie + 8-coordinate contracting tail (high_dims=10), which matches the paper's description; the exact training-time hyperparameters (epochs, patience, learning rate, batch size) are inherited from `_shared/defaults.yaml` rather than from a recovered Patrick config. Resolution of the source gap is deferred (user will provide Patrick's config later).

Diagnose on the current retrain: encoded extent `[1.08, 1.32]` (healthy), `latent_map` iteration does not converge after 200 steps. Same near-identity-latent regime as `leslie3d_success`.

## Reproduction commands

```bash
python pipeline.py --config configs/leslie_contraction.yaml --stages diagnose --max-seeds 1
```

After Patrick's hyperparameters are filled in, retrain on AMAREL:

```bash
sbatch --array=0-0 --export=ALL,CONFIG=configs/leslie_contraction.yaml,STAGES=train,diagnose,morse,EXPECTED_CELLS=1 \
  slurm/pipeline_array.sbatch
```

## Expected scientific output

Paper figure shows four Morse sets in a chain `3 -> {2, 1, 0}` with Conley indices `(0, x³-1, 0)`, `(0, 0, x-1)`, `(x-1, x-1, 0)`, `(x³-1, 0, 0)` (one repeller, one saddle, two attractors). The 8 contracting tail dimensions should be projected away by the encoder, leaving 2D Leslie-like dynamics in the latent.

## Hyperparameter audit

| param                       | archive value | YAML value              | source line                                | notes |
|-----------------------------|---------------|-------------------------|--------------------------------------------|-------|
| system.params.th1           | MISSING       | 23.5                    | configs/leslie_contraction.yaml:5          | reconstructed from paper Sec. 1.83 |
| system.params.th2           | MISSING       | 23.5                    |                                            | "  |
| system.params.survival_p1   | MISSING       | 0.7                     |                                            | "  |
| system.params.contraction   | MISSING       | 0.25                    |                                            | "  |
| arch.num_layers             | MISSING       | 4                       | configs/leslie_contraction.yaml:11         | inferred; revisit when Patrick config is recovered |
| arch.hidden_shape           | MISSING       | 64                      |                                            | inferred |
| arch.high_dims              | 10            | 10                      |                                            | ✓     |
| arch.low_dims               | 2             | 2                       |                                            | ✓     |
| arch.encoder_out_activation | likely tanh   | tanh (default)          |                                            | inherited from defaults |
| training.loss_weights       | unknown       | [10, 10, 1]             | configs/leslie_contraction.yaml:17         | inherited from sibling Leslie configs |
| data.n_samples_train        | unknown       | 10000                   | configs/leslie_contraction.yaml:23         | inferred from "10kData" in paper figure name |
| data.n_iterations           | unknown       | 20                      |                                            | inherited |
| cmgdb.subdiv_min            | 28            | 28                      | configs/leslie_contraction.yaml:30         | from paper Sec. 1.83 ("smin=28, smax=29") |
| cmgdb.subdiv_max            | 29            | 29                      |                                            | "  |

Open question: with `low_dims=2` and `subdiv_max=29`, the cubical complex has up to `2^58` boxes, which is computationally infeasible at face value. The paper's `subdiv_min=28, subdiv_max=29` likely refers to a different convention; clarify with Patrick.

## Verification

After retrain (and once the architecture is reconciled with Patrick's source):

```bash
python pipeline.py --config configs/leslie_contraction.yaml --stages diagnose --max-seeds 1
# Expect frac_unconverged < 0.5 and n_distinct_limit_points >= 2 (two attractors + a saddle).
```
