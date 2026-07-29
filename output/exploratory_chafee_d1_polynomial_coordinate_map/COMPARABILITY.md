# Chafee--Infante D1 polynomial physics-coordinate map

Designation: **test-informed exploratory polynomial physics-coordinate map**.

This output is not trained, unbiased, or paper-eligible.

## Valid interpretation

An isolated limit test of how the unsaturated cubic changes the same fixed-coordinate, fixed-grid basin calculation.

## Preserved from the rational-map ceiling

- exact SHA256-verified archived inputs
- E(x)=x[:, 0] with float32-equivalent rounding
- a from the same two archived stable roots
- mu=0.75
- bounds from all 30,000 current and next training pairs
- 10 percent bounds padding
- level-8 uniform 256-cell CMGDB graph
- CMGDB BoxMap padding=True
- native CMGDB.MorseSingletonReachability
- negative-first closed-cell classification
- same 7,862 conditioned-trajectory denominator

## Map change

Removed the /(1+q^2) saturation factor: G(z)=a*(q+mu*q*(1-q^2)).

## Known global failure

The cubic folds, reverses sign for sufficiently large |q|, and is unbounded outside the finite CMGDB domain.

## Invalid interpretations

- a trained or learned model result
- an unbiased held-out generalization result
- a paper-eligible comparison
- proof that the polynomial is globally topology preserving
