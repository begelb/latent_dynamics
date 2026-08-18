# Dense sampled residual and tolerance evaluation

The computation covers only the minimal nodes of the four learned Morse graphs displayed in the manuscript. Direct Leslie ground-truth computations are excluded because they have no encoder residual. The Chafee--Infante latent d=1 and d=3 appendix estimates are in `chafee_latent_dimensions/dense_sampling.json`.

For each minimal node \(q\), the computed box set is

\[
N_q=|\pi^{-1}(q)|.
\]

The sampled quantities are

\[
\widehat R_q
=
\max_{x\in\mathcal S_q}
\|g(E(x))-E(f(x))\|_2
\]

and

\[
\widehat\tau_q
=
\min_{z\in\mathcal T_q}
\operatorname{dist}_2\!\left(g(z),Z\setminus\operatorname{Int}N_q\right).
\]

Here \(\mathcal S_q\) contains sampled original-system states whose encodings belong to \(N_q\). It combines stored transitions, fresh Sobol trajectories, and decoder-guided states inside the computational domain. The set \(\mathcal T_q\) contains a dense boxwise sample from two independently scrambled Sobol sequences, every box corner and center, and the points evaluated during local minimization.

The dense boxwise pass used at least 8.38 million explicit latent samples for every node. Every sampled image under \(g\) remained in the interior of its block. The residual searches evaluated between 4.11 and 8.28 million candidates for each discrete example. The Chafee--Infante search evaluated 388,920 candidates from 10,216 trajectories in total and four independently scrambled clipped decoder searches.

## Results

| Example | Node | Residual candidates | Accepted \(|\mathcal S_q|\) | Dense latent samples | \(\widehat R_q\) | \(\widehat\tau_q\) | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Leslie 3D (`leslie3d_example1`) | 0 | 8,275,336 | 568,045 | 8,391,990 | 1.0681129694 | 0.000424571335316 | 2515.7444 |
| Leslie 3D (`leslie3d_example1`) | 1 | 8,196,720 | 3,777,740 | 8,525,597 | 0.696977138519 | 0.000406308099627 | 1715.3907 |
| Leslie 3D (`leslie3d_example1`) | 4 | 8,162,783 | 28,363 | 8,388,714 | 0.231353729963 | 0.000462259165943 | 500.4849 |
| Extended 2D Leslie | 0 | 7,795,033 | 2,240,516 | 8,565,163 | 0.0679826512933 | 0.0000519760142197 | 1307.9620 |
| Extended 2D Leslie | 1 | 7,795,033 | 131,462 | 8,391,744 | 0.0531156510115 | 0.0000540655000805 | 982.4315 |
| Red coral | 0 | 4,109,464 | 1,940,312 | 8,388,609 | 0.0539666414261 | 0.00778808388859 | 6.9294 |
| Red coral | 1 | 4,109,464 | 716,059 | 8,388,615 | 0.247852250934 | 0.00795617816038 | 31.1522 |
| Chafee--Infante | 0 | 388,920 | 111,015 | 8,388,621 | 0.0352067910135 | 0.0395290926099 | 0.8907 |
| Chafee--Infante | 1 | 388,920 | 125,123 | 8,388,621 | 0.0160021390766 | 0.0425186231732 | 0.3764 |

For all seven Leslie and red coral nodes, \(\widehat R_q>\widehat\tau_q\). The sampled witnesses therefore numerically contradict the sufficient inequality. This does not imply that the corresponding attractors are spurious.

For the two Chafee--Infante nodes, \(\widehat R_q<\widehat\tau_q\). Thus the dense search found no violation. It does not prove the uniform inequality because finite residual sampling does not give an upper bound on the supremum.

## Historical Leslie value

The older Leslie 3D script reported approximately \(6\times10^{-4}\) for a 20-point node-4 sample. That script computed a squared residual. With the Euclidean norm used by the theorem, the same sample gives \(0.0246492624\), whose square is \(0.0006075861\). The dense node-4 residual witness is \(0.2313537300\).

## Reproducible artifacts

- `latentdynamics.analysis.sampled_metrics.tolerance_protocol` performs the dense latent tolerance search and the numerical interval checks.
- `latentdynamics.analysis.sampled_metrics.residual_protocol` performs the stored, fresh-trajectory, and decoder-guided residual searches.
- `latentdynamics.analysis.sampled_metrics.merge` combines the independent Chafee--Infante batches without duplicating the stored data.
- `scripts/compute_sampled_residual_tolerance.py` is the command-line driver for all of the above.
- `<example>/tolerance_evaluation.json` contains the tolerance samples and witnesses.
- `<example>/dense_sampling.json` contains the final residual samples, witnesses, counts, and comparisons.

The interval calculations use ordinary floating-point arithmetic without outward rounding. None of these computations is presented as a rigorous certificate.
