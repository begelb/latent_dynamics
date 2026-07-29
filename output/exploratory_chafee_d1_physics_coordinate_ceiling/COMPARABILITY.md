# Chafee--Infante D1 physics-coordinate ceiling

Designation: **test-informed exploratory physics-coordinate ceiling**.

This output is not a trained autoencoder run and is not paper-eligible. It was designed after inspecting the archived test labels.

## Valid interpretation

A problem-specific, test-informed ceiling showing what the fixed D1 CMGDB basin statistic can achieve with an almost ideal physical reaction coordinate and enforced three-fixed-point topology.

## Preserved basin-table semantics

- exact SHA256-verified archived inputs
- all 10,000 archived initial conditions and labels
- conditioning on the exact 7,862 nonzero labels
- float32-equivalent coordinate evaluation
- bounds inferred from all 30,000 current and next training pairs
- 10 percent bounds padding
- uniform CMGDB levels (8, 8, 8), giving 256 cells
- CMGDB BoxMap padding=True
- native CMGDB.MorseSingletonReachability
- complete reachable Morse-node set must equal a singleton attractor
- negative-basin-first closed-cell boundary rule
- same five count categories and percentage denominator

## Intentional differences

- E(x)=x_1 is fixed physics, not a trained encoder
- G is an analytic odd double-well map, not a trained MLP
- mu=0.75 prioritizes topology and certification over time-tau fit
- there is no decoder, reconstruction loss, optimizer, or checkpoint
- the analytic callback is evaluated directly rather than through a persisted neural corner table
- only the uniform graph required for the basin table is computed
- there is no adaptive Morse graph or Conley-index annotation

## Invalid interpretations

- evidence that a generic one-dimensional autoencoder learns this score
- an unbiased held-out generalization result
- a paper result directly comparable as a trained-model row
- an accurate learned surrogate of the time-0.1 PDE map
