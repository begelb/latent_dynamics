# Leslie3D Example 2: Patrick-style paper 3x5 replication

Date: 2026-08-04  
Status: running; persistent launcher entered MPS training at 12:19 EDT  
Scope: five independently sampled training datasets by three independent
network initializations, for 15 cells.

## Objective and pass criterion

This run closely repeats the learned Leslie3D Example 2 experiment described
in the paper and used by Patrick. It measures whether the learned latent
dynamics robustly recover bistability across both data and model seeds.

A cell passes when its Morse graph has exactly two sink nodes and the complete
Conley-index tuple of each sink is

```text
(x^p-1, 0, 0), p >= 1.
```

The two periods may differ. Additional nonsink Morse nodes are allowed, as in
Patrick's archived graph. The summary retains every sink's node identifier,
full index tuple, and inferred period so the classification remains auditable.

## Fixed design

- training-data seeds: `1,2,3,4,5`;
- network seeds per dataset: `0,1,2`;
- shared validation seed: `9999`;
- 8,000 training and 2,000 validation initial conditions per dataset;
- uniform initial conditions on `[0,220] x [0,154] x [0,108]`;
- trajectories from `T0=0` through `T=20`, yielding 160,000 training and
  40,000 validation transition pairs per dataset;
- MinMax scaling;
- encoder `3 -> 64 -> 64 -> 2`, latent map `2 -> 64 -> 64 -> 2`, and decoder
  `2 -> 64 -> 64 -> 3`, with the paper's ReLU hidden activations and configured
  output activations;
- Adam at learning rate `0.001`, batch size `1024`, at most 1,000 epochs, and
  early-stopping patience 100;
- loss weights `(100, 10, 20)` for reconstruction, prediction, and
  semiconjugacy;
- CMGDB subdivision ladder `25/28/29`, limit `10000`, and 1% bound padding.

The inferred latent bounds use encoded training pairs only, keeping the shared
validation holdout out of both training and Morse-domain selection. The
adaptive lookup is precomputed through `subdiv_min=28`; any finer map values
needed at subdivision 29 are evaluated in batches on demand.

## Execution plan

The persistent launcher performs these phases sequentially:

1. verify MPS and the native graph-only and batched-map entry points;
2. save the resolved 15-cell plan;
3. generate and scale the five datasets;
4. train and diagnose all 15 models on MPS;
5. compute each Morse graph and its Morse sets in an isolated process;
6. render only the Morse graph and Morse-set figures, then write the aggregate
   sweep summary.

Regions of attraction, overlays, and unrelated extra figures are deliberately
excluded. Each Morse process is separate so its dense level-28 lookup can be
released before the next cell begins. The native safety ceilings are
`40,000,000` map-graph vertices and `1,200,000,000` edges.

The launcher is `scripts/run_leslie3d_patrick_paper_3x5.sh`. Run artifacts are
written under
`output/leslie3d_example2_seedsweep_patrick_paper_3x5_v1/`, and the final
machine-readable classification is `sweep_summary.json` in that directory.
