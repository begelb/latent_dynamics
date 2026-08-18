# Uniform (22,22,22) sampled residual and tolerance

The two minimal components of the exact fixed-depth Morse graph are nodes 0 and 1.
Node 23 is nonminimal and is not part of this bistability comparison.

| Node | Boxes | Accepted residual samples | Residual candidates | R_hat | Tolerance samples | tau_hat | R_hat/tau_hat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,714 | 904,399 | 6,872,468 | 1.0681129694 | 8,388,974 | 0.000800572277512 | 1334.19 |
| 1 | 85,170 | 3,074,979 | 6,872,468 | 0.696977138519 | 8,431,830 | 0.000792475941125 | 879.493 |

R_hat is a finite sampled maximum and therefore a lower bound on the true supremum residual.
tau_hat is a finite sampled minimum and therefore an upper estimate of the true infimum clearance.
An exact replay of the fixed-22 cell graph verifies that each minimal recurrent component equals its full forward closure.
Thus R_hat >= tau_hat is a numerical witness against the strict sufficient inequality for the sampled recurrent set; the calculation does not classify an attractor as spurious.
