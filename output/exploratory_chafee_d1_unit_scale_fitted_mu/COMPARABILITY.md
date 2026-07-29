# Raw-coordinate unit-scale fitted-mu cubic

Designation: **exploratory raw-coordinate unit-scale fitted-mu cubic map**.

The only learned object is one scalar mu, fitted on 30,000 training pairs.
No trajectory/test labels or stable-root values enter the fit.

## Valid interpretation

An exploratory one-parameter reduced map showing the effect of forcing a=1 while leaving the physical coordinate unnormalized.

## Not learned

- the fixed first-Fourier-coordinate encoder
- the unit scale a=1
- the cubic functional form

## Evaluation protocol

- raw E(x)=x[:,0] with float32-equivalent rounding
- bounds from all current and next training pairs
- 10 percent bounds padding
- level-8 uniform 256-cell CMGDB graph
- CMGDB BoxMap padding=True
- native CMGDB.MorseSingletonReachability
- negative-first closed-cell basin classification
- 7,862 conditioned-trajectory denominator

## Invalid interpretations

- a trained autoencoder result
- a learned encoder result
- a reconstruction or full-state prediction model
- a paper-eligible model comparison
