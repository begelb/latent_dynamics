# Normalized Chafee--Infante D1 limit test

## Setup

The coordinate and map are

\[
a=1.2365946,\qquad E(x)=x_1/a,\qquad G_\mu(z)=z+\mu z(1-z^2).
\]

The encoded PDE roots are ±0.9999999658434106, within \(3.42\times10^{-8}\) of the map roots ±1.

## Residual versus basin topology

| variant | μ | training MSE | Morse nodes | correct | outside | wrong |
|---|---:|---:|---:|---:|---:|---:|
| least-squares fit | 0.159077082484416 | 0.00238693829605 | 26 | 671/7862 (8.534724%) | 7191 | 0 |
| μ=0.75 inherited exploratory choice | 0.75 | 0.0193327279548 | 6 | 7092/7862 (90.206054%) | 386 | 384 |
| μ=0.55 post-hoc scan winner | 0.55 | 0.0098031549274 | 7 | 7811/7862 (99.351310%) | 51 | 0 |

The least-squares value is genuinely the best one-step residual fit, yet it produces 26 Morse nodes and only 8.53% strict basin coverage. The post-hoc μ=0.55 map has roughly 4.11 times larger MSE but reaches 99.35%. This is a direct residual-versus-topology disconnect.

The basin rule is deliberately strict: a candidate cell is assigned only when its complete reachable Morse-node set is exactly one singleton attractor. With the fitted μ, the 24 nonminimal recurrent nodes create many intermediate reachable sets, so 7,191 conditioned points are classified outside even though there are two valid minima.

## Theorem-aligned attracting-block audit

For every minimal node, `N_q` is the cell-level forward closure from `attractor_cells`. In all rows below this closure equals the recurrent cells and forward invariance was verified. The margin is

\[
\tau(N_q,G)=\inf_{z\in N_q}\operatorname{dist}(G(z),Z\setminus\operatorname{Int}N_q),
\]

computed analytically from interval endpoints and every derivative-critical point of the cubic.

The normalized uniform grid width is \(h=(Z_{max}-Z_{min})/256=0.0151394755258\). Since `BoxMap` pads one cell on each side, tau-h is also reported as a distinct numerical-robustness diagnostic; τ itself remains the theorem quantity.

| variant | node/sign | cells | width | root clearance | tau | tau-h | stored pairs in N | sample max | max/tau |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fitted | 0/negative | 7 | 0.105976329 | 0.0454225782 | 0.0154509067 | 0.000311431212 | 10012 | 0.445238719 | 28.8163 |
| fitted | 13/positive | 8 | 0.121115804 | 0.0589645882 | 0.0179684314 | 0.00282895591 | 9302 | 0.457323577 | 25.4515 |
| μ=0.75 | 0/positive | 6 | 0.0908368532 | 0.0438251127 | 0.0207146718 | 0.00557519632 | 9263 | 0.432279946 | 20.8683 |
| μ=0.75 | 2/negative | 6 | 0.0908368532 | 0.045414275 | 0.0179905332 | 0.00285105764 | 9987 | 0.464984634 | 25.8461 |
| post-hoc μ=0.55 | 0/positive | 4 | 0.0605579021 | 0.0286856372 | 0.0271547882 | 0.0120153127 | 9210 | 0.440756053 | 16.2312 |
| post-hoc μ=0.55 | 3/negative | 4 | 0.0605579021 | 0.0302747995 | 0.0257180965 | 0.0105786209 | 9938 | 0.458301558 | 17.8202 |

Every sampled maximum exceeds its corresponding τ, so each stored witness directly contradicts the tolerance inequality for that block. More generally, a finite-sample maximum is only a lower bound on the true supremum: a sample maximum below τ would be inconclusive, while one above τ is a valid counterexample.

## Provenance caveat

The least-squares μ uses only the 30,000 training pairs. μ=0.75 was pre-specified for this normalized rerun but inherited from an earlier exploratory/test-informed design. μ=0.55 was selected post-hoc using these same archived basin labels. Neither the scan winner nor this report is an unbiased or paper-eligible result.
