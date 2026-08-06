# Leslie3D ground-box wide curriculum 3x5 experiment

Date configured: 2026-08-06  
Status: **configured, not run**  
Scope: five independently sampled training datasets by three independent model
initializations, for 15 baseline models. No margin fine-tuning is part of this
run.

## Question

This experiment asks whether a fixed, full-batch curriculum followed by a
quasi-Newton polish can push the three Leslie3D fitting errors lower and make
the learned two-dimensional recurrent structure more reliable than the earlier
simultaneous-loss recipes. It is a deliberately small data experiment: the
purpose is to stress optimization before designing an intervention around the
actual theorem margin.

This is a **new controlled staged-training experiment**, not a reproduction of
an archived Marcio or Brittany recipe. Marcio's archived loop used full-batch
Adam with `ReduceLROnPlateau` on two simultaneously active terms. Brittany's
Leslie run used Adam, minibatches, a scheduler, early stopping, and one joint
loss call. Neither archive contains AdamW, L-BFGS, or the three stages below.
The five-by-three seed layout and fixed-epoch/full-batch spirit are historical
motivations only; the optimizer sequence here is explicitly new.

## Frozen physical data contract

The analytic map is

\[
f(x_1,x_2,x_3)=\left(
 (28.9x_1+29.8x_2+22x_3)e^{-0.1(x_1+x_2+x_3)},
 0.7x_1,
 0.7x_2
\right).
\]

Initial conditions are independent uniform draws on

\[
B=[0,110]\times[0,77]\times[0,54].
\]

This is the absorbing box used by the accepted direct ground-truth computation,
not the larger manuscript training box. Its saved direct-map configuration is
`output/original_leslie/ground_truth/absorbing_B_i29_m33_M36_L10000/conley/run_config.json`;
the absorption argument is recorded in `../scratch/leslie3d_absorbing_domain.md`.

For every data realization:

- training initial conditions: 1,000;
- validation initial conditions: 200;
- generated map steps: `T=20`;
- discarded steps: zero;
- retained pairs per initial condition: 20, namely
  `(x_0,x_1),...,(x_19,x_20)`;
- training pairs: 20,000;
- validation pairs: 4,000;
- scaling: a separate MinMax scaler fitted to the union of the training current
  and next states, never to validation;
- validation seed: `9999`, shared across all five data realizations.

Here “1,000 initial conditions” means **1,000 training initial conditions**, not
an 800/200 split of a total budget of 1,000. Across the five dataset trees there
are 100,000 generated training pairs. Each physical dataset and its fitted
scaler are reused by its three model initializations.

The two independent seed axes are frozen as follows:

| dataset | training-IC seed | model seeds |
|---:|---:|---|
| 1 | `2158` | `0,1,2` |
| 2 | `4792` | `0,1,2` |
| 3 | `3174` | `0,1,2` |
| 4 | `688` | `0,1,2` |
| 5 | `5727` | `0,1,2` |

These IC seeds preserve the five data realizations used in Marcio's archived
robustness layout. They do not make the present Leslie datasets identical to
Marcio's Chafee--Infante data.

## Architecture

The latent dimension is two. Every hidden activation is `tanh`, and every
component has a linear output:

| component | architecture |
|---|---|
| encoder \(E\) | `3 -> 128 -> 64 -> 2` |
| latent map \(G\) | `2 -> 64 -> 64 -> 2` |
| decoder \(D\) | `2 -> 64 -> 128 -> 3` |

The wider encoder and decoder are intentional. There is no bounded output
activation whose saturation could be confused with the curriculum effect.

## Fixed AdamW curriculum and final L-BFGS polish

For scaled one-step pairs \((x,y)\), define

\[
\begin{aligned}
L_1 &= \operatorname{MSE}(D(E(x)),x),\\
L_2 &= \operatorname{MSE}(D(G(E(x))),y),\\
L_3 &= \operatorname{MSE}(G(E(x)),E(y)).
\end{aligned}
\]

One AdamW optimizer is constructed at learning rate `0.003` before the first
stage. Its moment state is retained across both stage boundaries. Every
first-order epoch is one update on all 20,000 training pairs. AdamW uses betas
`(0.9,0.999)`, epsilon `1e-8`, `amsgrad=false`, `foreach=false`, and
`fused=false`. Weight decay is explicitly zero. Thus this first-order phase is
algorithmically equivalent to Adam; the name records the requested optimizer
implementation, while avoiding an extra regularization objective that the
subsequent L-BFGS phase would not share.

| stage | epochs | weights \((w_1,w_2,w_3)\) | trainable components |
|---:|---:|---|---|
| 1, autoencoder | 4,000 | `(1,0,0)` | encoder and decoder only |
| 2, decoded prediction | 4,000 | `(1,1,0)` | encoder, latent map, decoder |
| 3, semiconjugacy | 4,000 | `(1,1,1)` | encoder, latent map, decoder |

The stage-3 endpoint after 12,000 AdamW updates is saved permanently. A fresh
L-BFGS optimizer then trains **all encoder, latent-map, and decoder parameters**
on the final joint objective `L1+L2+L3`. It runs on the full training batch on
CPU in float64, with strong-Wolfe line search and the following frozen budget:

- 12 outer optimizer calls;
- learning rate `0.25`;
- at most 10 internal iterations and 25 function evaluations per outer call;
- history size 50;
- gradient tolerance `1e-9` and change tolerance `1e-12`.

This final-only placement is deliberate: stages 1 and 2 have provisional
objectives, so polishing them would change the curriculum itself. L-BFGS starts
with fresh state; no optimizer state is transferable from AdamW. Its closure
contains only deterministic full-batch evaluation and differentiation of
`L1+L2+L3`. Holdout loss, topology, theorem tolerances, and margin terms do not
enter the closure.

After the twelfth outer call, the endpoint is cast to float32 and its train and
holdout losses are recomputed. That exact float32 endpoint—not a validation-best
iterate—is the selected checkpoint used by diagnosis and CMGDB. The run fails
instead of silently rolling back if float32 casting leaves the final training
objective materially above the saved AdamW endpoint. Stage-end and AdamW
endpoint checkpoints remain available for audit.

The curriculum trainer has:

- no scheduler;
- no early stopping or patience decision;
- no best-epoch restoration;
- no validation-based optimization or selection;
- no gradient clipping.

The common config schema still requires `patience` and `lr_patience`; they are
stored as the explicitly inert values `12001` and `100`. Validation losses are
evaluated for reporting only after AdamW updates and L-BFGS outer calls. L-BFGS
line-search termination is an optimizer-internal numerical rule, not patience,
scheduling, or validation selection.

## Baseline CMGDB contract

Each final learned map is summarized with the established two-dimensional
latent computation:

- subdivisions `25/28/29` for `init/min/max`;
- subdivision limit `10000`;
- `adaptive_precomputed` box-map backend;
- dense precomputation through `subdiv_init`, with later corners evaluated on
  demand;
- latent rectangle inferred from encoded **training** current and next states;
- one-percent coordinatewise rectangle expansion;
- box-map padding enabled;
- exact regions of attraction disabled.

These are learned two-dimensional graph parameters. They should not be confused
with the direct three-dimensional ground-truth calculation on \(B\), whose
subdivision tuple is `29/33/36` and whose corner box map uses `padding=False`.

The baseline summary must retain, per cell, the stage-end losses, AdamW endpoint
and final float32 training/holdout values of every unweighted term, their
changes, L-BFGS closure/function-evaluation/internal-iteration counts, both
checkpoint identities, diagnosis, latent bounds, Morse-node and minimal-node
counts, complete sink Conley indices, and saved graph/set artifact identities.
A graph with exactly two period-four sink indices is an important comparison
diagnostic, not a checkpoint-selection rule.

## Execution and output isolation

The packaged config is
`src/latentdynamics/configs/leslie3d_groundbox_curriculum_wide.yaml`. The frozen
launcher is
`scripts/run_leslie3d_groundbox_curriculum_wide_3x5.sh`.

Run from `code/` with:

```bash
bash scripts/run_leslie3d_groundbox_curriculum_wide_3x5.sh
```

The AdamW stages default to MPS; L-BFGS is always CPU float64. An explicit
first-order alternative can be selected without editing the experiment
contract, for example:

```bash
TRAIN_DEVICE=cuda bash scripts/run_leslie3d_groundbox_curriculum_wide_3x5.sh
```

The launcher runs these phases in order:

1. preflight and exact config validation;
2. persist the resolved 15-cell plan;
3. generate and scale the five datasets;
4. train and diagnose all 15 models;
5. compute CMGDB separately for every cell;
6. render the Morse products and build the generic sweep summary;
7. strictly verify all 15 generic cells;
8. run the dedicated curriculum summarizer and verify its 15-row report.

Pipeline stages resume at completed cell boundaries through `--skip-completed`.
The AdamW and stage snapshots are audit artifacts, not mid-training restart
states; an interrupted training cell restarts that cell from its declared seed.
The baseline artifacts are isolated under

```text
data/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/
output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/
```

The generic report is `sweep_summary.json`. The dedicated report products are

```text
summary/cells.csv
summary/aggregate_summary.json
summary/SUMMARY.md
```

The launcher refuses to report success unless the generic sweep contains the
exact five-by-three seed grid with all 15 checkpoints and Morse artifacts, the
strict dedicated summarizer exits successfully, and its cell table has 15 rows.

## Deferred margin experiment

There is intentionally no margin objective or margin-fine-tuning phase in this
experiment. First inspect the complete 3x5 distribution of losses, latent Morse
graphs, directly confirmed recurrent objects, and theorem residual/tolerance
diagnostics. Only then define candidate attracting blocks and a separate,
versioned fine-tuning intervention around the actual margin. That later run must
not overwrite or be presented as part of this baseline.
