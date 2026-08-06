# Ives--Myvatn sampled residual and tolerance

Generated 2026-08-06T18:22:38.490800+00:00 for the successful learned run: data seed 2158, model
seed 2. This is a finite numerical diagnostic, not a certified uniform bound.

For each graph-minimal node, the saved cell union is used as the candidate
block `N_q`:

```
R_hat(q)   = max ||g(E(x)) - E(f(x))||_2,  sampled x with E(x) in N_q
tau_hat(q) = min dist_2(g(z), Z \ Int(N_q)), sampled z in N_q
```

Both are unsquared Euclidean distances in the stored two-dimensional latent
coordinates. Images outside or on the block boundary receive zero clearance.

| Set | Role | Boxes | Components | Accepted S_q | Residual candidates | Explicit latent samples | R_hat | tau_hat | Ratio | Diagnostic |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M0 | stable period-12 orbit | 2,717 | 12 | 358,370 | 3,834,518 | 8,392,813 | 1.136612892 | 0.0008083716966 | 1406.052311 | sampled violation |
| M2 | stable fixed point | 52 | 1 | 835,404 | 3,834,518 | 8,388,692 | 1.01960218 | 0.0008111030329 | 1257.056303 | sampled violation |

`R_hat >= tau_hat` is a sampled witness against the strict sufficient lifting
inequality for that candidate block. It does not mean the attractor is
spurious. Conversely, `R_hat < tau_hat` would only mean no sampled violation
was found.

## Residual-source robustness

| Set | Legacy lightweight R_hat | Stored pairs max | Fresh trajectories max | Decoder-guided max | Best non-decoder / tau_hat |
|---|---:|---:|---:|---:|---:|
| M0 | 0.01367610134 | 0.001514949952 | 0.8990408778 | 1.136612892 | 1112.162736 |
| M2 | 0.0041699172 | 1.775014243e-05 | 0.110797435 | 1.01960218 | 136.6009379 |

The final maxima come from decoder-guided states, as in the paper protocol,
but that source is not needed for the conclusion: stored pairs plus fresh
trajectories already give `R_hat > tau_hat` for both sets. The legacy metric
used only 4096 post-transient physical samples and a corner-based tolerance;
the dense search covers the full experiment domain and every retained
trajectory time.

## Cross-check against paper Table 8

| Paper example | q | |S_q| | R_hat | tau_hat | Ratio | Diagnostic |
|---|---:|---:|---:|---:|---:|---|
| 1st Leslie 3D | 0 | 5.68e+05 | 1.07 | 0.000425 | 2.52e+03 | sampled violation |
| 1st Leslie 3D | 1 | 3.78e+06 | 0.697 | 0.000406 | 1.72e+03 | sampled violation |
| 1st Leslie 3D | 4 | 2.84e+04 | 0.231 | 0.000462 | 500 | sampled violation |
| 2nd Leslie 3D | 0 | 446 | 0.0848 | 4.57e-05 | 1.86e+03 | sampled violation |
| 2nd Leslie 3D | 1 | 6.7e+03 | 0.0975 | 4.48e-05 | 2.18e+03 | sampled violation |
| 2D Leslie in 10D | 0 | 2.24e+06 | 0.068 | 5.2e-05 | 1.31e+03 | sampled violation |
| 2D Leslie in 10D | 1 | 1.31e+05 | 0.0531 | 5.41e-05 | 982 | sampled violation |
| Red coral | 0 | 1.94e+06 | 0.054 | 0.00779 | 6.93 | sampled violation |
| Red coral | 1 | 7.16e+05 | 0.248 | 0.00796 | 31.2 | sampled violation |
| Chafee--Infante d=1 | 0 | 1.33e+05 | 6.58 | 0.104 | 63.3 | sampled violation |
| Chafee--Infante d=1 | 1 | 1.31e+05 | 6.11 | 0.0658 | 92.9 | sampled violation |
| Chafee--Infante d=2 | 0 | 1.11e+05 | 0.0352 | 0.0395 | 0.891 | no sampled violation |
| Chafee--Infante d=2 | 1 | 1.25e+05 | 0.016 | 0.0425 | 0.376 | no sampled violation |
| Chafee--Infante d=3 | 0 | 1.19e+05 | 0.00431 | 0.0234 | 0.184 | no sampled violation |
| Chafee--Infante d=3 | 1 | 1.18e+05 | 0.00473 | 0.0236 | 0.2 | no sampled violation |

The useful cross-example comparison is the inequality and dimensionless ratio,
not the raw latent distances: independently trained latent coordinates have
different geometry. The Leslie-family rows have ratios about 500--2516; the
two-dimensional Chafee--Infante rows with no sampled violation have ratios
about 0.376 and 0.891.

The tolerance computation evaluates all corners and centers of every cell,
two independently scrambled boxwise Sobol designs with at least `2^23`
explicit latent points per node, and local differential-evolution searches in
the lowest-clearance cells. The residual pool combines stored train/validation
pairs, 131072 fresh Sobol initial conditions followed for 24 steps, and
decoder-guided states at five noise scales.

See `sampled_residual_tolerance.json` for witnesses, per-source counts, seeds,
timings, software versions, and SHA-256 provenance. The paper definitions and
comparison table are at `paper/main_KM2.tex:2940-2979`.
