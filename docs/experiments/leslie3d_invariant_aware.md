# Leslie3D Example 2: invariant-aware training

Date: 2026-08-04  
Status: v1, primary-v2, rollout-v2, and the max-30 resolution audit are complete; the local skeleton is recovered, but the requested numerical Morse/index target is not recovered and no run is a rigorous Conley certificate  
Scope: isolated exploratory computation; no paper artifact is overwritten

## Question

Can the Example-2 Leslie map be learned more faithfully in two latent
dimensions by replacing ordinary uniform trajectories with a data set that
explicitly contains the recurrent skeleton and true transition tubes?

The physical map is

\[
f(x_1,x_2,x_3)=\left(
(28.9x_1+29.8x_2+22x_3)e^{-0.1(x_1+x_2+x_3)},
0.7x_1,
0.7x_2
\right).
\]

The numerical recurrent inventory has 16 phases:

| object | type | phases |
|---|---|---:|
| \(P_0\) | stable period 4 | 4 |
| \(P_1\) | stable period 4 | 4 |
| \(S_2\) | saddle period 2 | 2 |
| \(S_4\) | saddle period 4 | 4 |
| \(p_*\) | positive saddle fixed point | 1 |
| \(0\) | unstable boundary fixed point | 1 |

The orbit/manifold-informed target order is

```text
S2 -> P1
S4 -> P0, P1
p* -> S2
0  -> S4, p*
```

This target is compared with both the saved direct-map CMGDB computation and
the numerical orbit/manifold inventory. The latter resolves a spurious direct
box-graph return corridor but is not itself a computer-assisted proof about
the continuous Leslie map.

## Curated data

### Version 1

Run from `code/`:

```bash
.venv/bin/python scripts/build_leslie3d_invariant_aware_dataset.py
```

The builder writes `data/leslie3d_invariant_aware/`. Every row is an exact
pair \((x,f(x))\) evaluated with the analytic Leslie map; no convex or linear
interpolation is used as a pseudo-transition.

| component | train pairs | model-selection pairs | purpose |
|---|---:|---:|---|
| exact recurrent phases | 8,192 | 0 | force every phase and successor into training; audit all 16 phases separately |
| multiscale recurrent neighborhoods | 4,096 | 1,013 | sample local entry/exit behavior over seven decades of radius |
| balanced direct Morse neighborhoods | 5,120 | 1,280 | cover all five non-origin recurrent box sets equally |
| saddle tangent transition tubes | 6,144 | 3,072 | follow true forward trajectories from linear unstable-direction probes |
| origin positive-cone fan | 1,024 | 512 | resolve the boundary repeller missed by uniform sampling |
| Sobol background trajectories | 12,288 | 6,144 | constrain the neural extension away from the skeleton |
| absorbing-box corners | 8 | 0 | make physical scaling support explicit in training |
| **total** | **36,872** | **12,021** | |

The stochastic validation clouds and box samples use a separate seed. Saddle
probe amplitudes occupy different logarithmic subcells in the two splits. The
builder compares rows at their exact 15-digit CSV representation, removes 11
clipped local-cloud collisions, and records zero remaining overlap. Exact
recurrent anchors are target-defining audits rather than model-selection data.
Component row ranges, hashes, source-box hash, coordinates, and construction
seeds are recorded in `dataset_manifest.json` and the split metadata.

### Version 2: audited finite transition witnesses

The separate v2 builder and configs do not overwrite the v1 data or output
trees:

```bash
.venv/bin/python scripts/build_leslie3d_invariant_aware_v2_dataset.py
```

Version 2 preserves every v1 component and adds an
`audited_origin_p_star_s2_transition_tubes` component. In each split, three
strictly positive base points are expanded to their center and all eight
coordinatewise \(\pm0.1\%\) multiplicative corners. This gives 27 trajectories
per split and 320 analytic Leslie steps per trajectory, or 8,640 additional
pairs per split.

| v2 contribution | train pairs | model-selection pairs |
|---|---:|---:|
| complete v1 construction | 36,872 | 12,021 |
| audited origin--\(p_*\)--\(S_2\) trajectories | 8,640 | 8,640 |
| **v2 total** | **45,512** | **20,661** |

The six base points come from a reproducible four-dimensional scrambled-Sobol
search with seed 90317 and \(2^{19}\) candidates. The first three Sobol
coordinates determine a positive direction, while the fourth selects a
log-uniform scaled radius in \([10^{-12},5\times10^{-3}]\). The selected Sobol
indices are 151783, 234911, and 64934 for training, and 482705, 232929, and
386002 for model selection. The builder recomputes the selected centers from
those indices and records the centers, perturbation factors, source hashes,
and every itinerary diagnostic in the v2 manifest.

Every one of the 54 finite trajectories starts in the saved direct-map origin
cell, enters and leaves the exact union of direct \(p_*\) boxes, subsequently
enters and leaves the exact union of direct \(S_2\) boxes, and has moved onward
by step 320. With distance normalized componentwise by
\((110,77,54)\), the worst closest approaches over each 27-trajectory bank are
\(2.671\times10^{-3}\) and \(2.688\times10^{-3}\) for \(p_*\), and
\(5.009\times10^{-5}\) and \(5.053\times10^{-5}\) for \(S_2\). The 8,640
training witness rows and 8,640 held-out witness rows are internally unique
and mutually disjoint at the 15-digit CSV representation; the final full
splits are also train/model-selection disjoint after the standard overlap
filter.

These are finite forward itinerary witnesses, not certified heteroclinic full
orbits. In particular, box visits do not prove backward convergence to the
origin, forward convergence to a named Morse set, isolation, or the absence of
other connections.

## Version 1 model and optimization

The chart starts from Patrick's archived 3--64--64--2 encoder and
2--64--64--3 decoder. A deterministic ReLU continuation on the curated data
first reduced all four independent validation errors, but a float64 root audit
showed that its nominal \(P_1\) had become a saddle and its nominal \(S_2\) a
sink. A second ReLU refinement passed finite-difference anchor tests yet failed
the same independent audit in the opposite way. Neither candidate was sent
forward as a successful topology result.

The selected local candidate freezes that encoder and decoder and replaces
only the 2--64--64--2 latent transition with a GELU network. A sharp transfer

\[
\operatorname{GELU}(100a)/100\simeq\operatorname{ReLU}(a)
\]

preserves the warm-start function while making exact autograd monodromies
available. Training then combines the full curated replay objective with an
augmented-Lagrangian phase-closure constraint and target trace/determinant
constraints for every return map.

| setting | value |
|---|---|
| optimized component | latent transition only; encoder/decoder frozen |
| latent architecture | 2--64--64--2, GELU hidden layers, tanh output |
| data batch | all 36,872 analytic pairs |
| replay weights | reconstruction 100, prediction 20, semiconjugacy 100, latent cycle 20 |
| smooth training | 12,000 epochs plus a 4,500-epoch low-step continuation |
| final continuation step | \(2\times10^{-7}\) |
| recurrent anchors | all 16 phases, phasewise one-step constraints |
| topology supervision | exact-autograd period-1/2/4 monodromy characteristic polynomials |
| scaler | Patrick's archived full-domain scaler, read-only |
| deterministic seeds | 20260803--20260805 |

Duplicating exact phases implements intentional importance weighting.
Reconstruction supplies the essential anti-collapse pressure; semiconjugacy
and cycle loss alone permit a constant encoder. The independent root solver,
dense recurrent census, and CMGDB computations are deliberately outside the
training-selection loop described above.

### Why the chart was frozen, and what an improved encoder can change

Freezing the archived encoder and decoder was an experimental control, not a
claim that the chart is optimal. It isolates whether invariant-aware data and
local topology supervision can repair the latent transition without moving the
coordinate system at the same time. A better encoder can make the subsequent
box computation easier by separating nearby recurrent objects, reducing folds,
and giving isolating neighborhoods more margin. It can therefore change a
finite-resolution Morse computation even when the conjugacy class on the
encoded invariant set is unchanged.

It cannot, by a coordinate change alone, delete the independently detected
extra latent cycles. A homeomorphic reparameterization (h) replaces (g) by
(hgh^{-1}) and transports those cycles with it. The present fine chart also
already places all six named objects in distinct Morse sets. Its closest named
separations include (0.03445) between (S_2) and (p_*), (0.03826)
between (P_1) and (S_4), and (0.05877) between (P_0) and the origin.
Thus chart quality is relevant, but the persistent extra sink/saddle and wrong
global order require changing the learned transition as well.

The guarded alternating variant performed exactly the first two operations.
Holding (g) fixed, it updated the final encoder and first decoder layers under
reconstruction, reference-chart, cross-object margin, and local anti-fold
constraints. It then refroze (E,D), recomputed every latent sample, anchor,
scale, rollout target, and CMGDB bound, and attempted to repair (g) under the
existing phase and monodromy gates. The accepted chart move preserved physical
accuracy and enlarged the tight named-object separations by roughly 1--2.3%.
The subsequent 4,000-epoch interior-only map repair and a further 4,000-epoch
anchor-nullspace output continuation were both rejected: their best maximum
characteristic errors were `0.128834` and `0.149037`, while the fixed gate was
`0.05`; their exact held-out rollout losses were `0.0219528` and `0.0219452`,
while the absolute gate was `0.00697`. No alternating checkpoint was promoted,
so a new cycle census or Morse computation for that branch would not be
scientifically meaningful. A future repair should recycle the independently
found extra roots and their local rings as moving hard negatives rather than
only fitting the named recurrent skeleton.

### Relation to the paper's Example-2 settings

The invariant-aware experiment uses the same Leslie coefficients and the same
two-dimensional 64-wide model family, but it is **not** a rerun with the
paper's training hyperparameters.

| choice | paper Example 2 | invariant-aware primary v2 |
|---|---|---|
| data | 8,000 train and 2,000 validation initial conditions, uniform on (X), 20 retained steps | 45,512 train and 20,661 held-out analytic pairs, deliberately balanced around recurrent objects and transition tubes |
| networks | (E,D,g) each have two 64-wide ReLU hidden layers; tanh outputs for (E,g), sigmoid for (D) | archived paper-family (E,D); 2--64--64--2 GELU (g) with tanh output |
| trainable components | (E,D,g) jointly | (g) only; (E,D) frozen |
| optimizer | Adam, learning rate (10^{-3}), batch 1024, at most 1,000 epochs, patience 100 | full-batch topology continuation; selected epoch 669 of 3,170, final step (2\times10^{-7}) |
| base loss weights | reconstruction/prediction/semiconjugacy (=100/10/20) | reconstruction/prediction/semiconjugacy/cycle (=100/20/100/20), plus component replay, trust, anchor, and characteristic-polynomial terms |
| latent CMGDB ladder | ((25,28,29)), limit 10,000, 1% data-range padding | audited at ((20,24,26)), ((24,27,28)), ((25,28,29)), and ((25,28,30)) on one fixed rectangle with corner-image padding |

The manuscript explicitly says these architecture and hyperparameter choices
were not optimized. Its preserved tables specify the dataset, networks,
optimizer, and CMGDB ladder, but the original raw Example-2 CSVs, sampling seed,
and exact training script are not present in this checkout, so bitwise
reproduction of that run is unavailable.

## Conditional ideal-data theorem (not an experimental result)

The strongest defensible statement is conditional: ideal data can force an
exact commuting relation, but **semiconjugacy alone neither identifies a
unique latent map nor transfers a Conley index**.

Let \(K\) be a compact subset of a finite-dimensional normed space, let
\(Z\) be a normed latent space, let \(f:K\to K\), and let
\(E:K\to Z\), \(D:E(K)\to K\), and \(g:Z\to Z\) be continuous. Let a
probability measure \(\mu\) have full support on \(K\). Define the
semiconjugacy and reconstruction residuals

\[
r_{\rm sc}(x)=g(E(x))-E(f(x)),\qquad
r_{\rm rec}(x)=D(E(x))-x.
\]

If the hypothesis class contains a zero-loss solution and optimization attains
a population-global minimum with

\[
\int_K \bigl(\|r_{\rm sc}(x)\|^2+\|r_{\rm rec}(x)\|^2\bigr)\,d\mu(x)=0,
\]

then continuity and full support imply \(gE=Ef\) and \(DE=\mathrm{id}_K\)
pointwise on \(K\). Indeed, a positive continuous residual at one point would
be positive on a relatively open set of positive measure. Consequently \(E\)
is injective; because \(K\) is compact and the latent space is Hausdorff,
\(E:K\to E(K)\) is a homeomorphism. Thus \(g|_{E(K)}=EfD\) is conjugate to
\(f|_K\). If only the first residual has zero population loss, the conclusion
is merely the factor relation \(gE=Ef\), and \(E\) may collapse distinct
invariant sets or periodic phases.

**Conditional index-transport theorem.** Let \(S\) be an isolated invariant
set of \(f\), represented by a compact index pair \(P=(P_1,P_0)\) contained in
\(K\), and let \(f_P:P_1/P_0\to P_1/P_0\) be its pointed index map. In addition
to the ideal equalities above, assume all of the following:

1. \(Q_i=E(P_i)\) form a valid index pair \(Q=(Q_1,Q_0)\) for a \(g\)-isolated
   invariant set \(T\), including the required positive-invariance and
   exit-set conditions;
2. \(T=E(S)\), so the latent isolating neighborhood contains no additional
   invariant dynamics and loses none of \(S\); and
3. \(E\) transports the pair and its exit convention so that it descends to a
   pointed homeomorphism
   \(\bar E:P_1/P_0\to Q_1/Q_0\) satisfying
   \(\bar E f_P=g_Q\bar E\).

Then the pointed index maps are conjugate. Their induced relative-homology
maps are conjugate over any common coefficient field, and the discrete
homological Conley indices of \(S\) and \(T\) agree. This conclusion follows
from index-pair transport, not from the semiconjugacy loss by itself.

Exact reconstruction on an open physical neighborhood is unavailable to the
present 3-to-2 architecture: a continuous encoder into \(\mathbb R^2\) cannot
embed an open subset of \(\mathbb R^3\). Embedding only a lower-dimensional
invariant support is also insufficient for its ambient Conley index, because
the exit dynamics live on a neighborhood/index pair. A genuinely
dimension-reducing alternative is therefore to certify shift equivalence
directly. If \(A\) and \(B\) are the pointed index maps and encoder/decoder
constructions induce basepoint-preserving maps \(R\) and \(S\), it is enough to
verify, on the quotient spaces or on induced chain maps up to chain homotopy,

\[
RA=BR,\qquad SB=AS,\qquad SR=A^\ell,\qquad RS=B^\ell
\]

for some \(\ell\geq0\). On relative homology over one common field (the CMGDB
computations here use \(\mathbb Z_5\)), these identities make the index maps
shift equivalent and hence give the same discrete homological Conley index.
The current objective does not impose or verify these identities.

Recovering a full Morse graph needs more than equal nodewise indices. One must
also biject the maximal invariant sets, carry each Morse set to its
counterpart, preserve connecting full orbits and their partial order, and
exclude every extra latent recurrent component. Under a conjugacy of the
relevant maximal invariant dynamics these properties transfer; a finite
collection of encoded anchors or box visits does not establish them.

### Non-uniqueness, even in the ideal limit

The theorem does not identify one coordinate formula as “the optimal map.”
For every latent homeomorphism \(h\), the pair

\[
E'=hE,\qquad g'=hgh^{-1}
\]

is an equivalent exact solution on \(h(E(K))\). Data on \(K\) also leave the
extension of \(g\) outside \(E(K)\) undetermined. Without exact reconstruction
or another injectivity hypothesis, even the factor dynamics on \(E(K)\) may
collapse states: an exact semiconjugacy can map a period-\(p\) orbit to a
proper-divisor period. Therefore ideal learning can at most identify an
appropriate conjugacy/shift-equivalence class, not a unique latent coordinate
map.

A simple counterexample isolates the issue. Let \(f\) be the identity on two
isolated points, let \(g\) be the identity on one point, and let \(E\) collapse
both points. Then \(Ef=gE\) exactly, while the Morse decompositions and
degree-zero index ranks differ.

### What finite data add, and what they do not

Zero residual on a dense set gives the same pointwise commuting relation by
continuity. For a finite \(h\)-net \(\mathcal D\) and an \(L_r\)-Lipschitz
vector residual,

\[
\|gE-Ef\|_{\infty,K}
\leq \max_{x_i\in\mathcal D}\|gE(x_i)-Ef(x_i)\|
    +L_r h(\mathcal D,K).
\]

The v1 and v2 finite mean-square fits supply neither a
certified \(L_r\), a fill-distance bound on the required index pair, an outer
enclosure, nor a no-extra-invariant-set result. Recurrent neighborhoods and
the v2 transition witnesses make the sampled constraints more relevant, but
they do not verify any assumption of the conditional index-transport theorem.

The discrete Conley index is defined through index maps modulo shift
equivalence; see [Weilandt](https://arxiv.org/abs/1801.06403). Rigorous
inference from sampled dynamics requires a valid multivalued/outer
representation, as emphasized by
[Batko--Mischaikow--Mrozek--Przybylski](https://arxiv.org/abs/1904.03757)
and by the correspondence framework of
[Harker--Kokubu--Mischaikow--Pilarczyk](https://arxiv.org/abs/1411.7563).

## Certification boundary: theorem versus evidence

The conditional theorem above is not claimed as a result of this experiment.
The v1 and v2 computations test some consequences numerically, and v2 adds
more relevant finite trajectories plus one-step and multi-step optimization.
Neither closes these proof obligations:

1. finite data do not certify the residual supremum on an index pair or
   exclude an off-support recurrent set in the neural extension;
2. the 3-to-2 model cannot supply the local homeomorphism assumed by the first
   route on an open three-dimensional neighborhood;
3. the current loss does not construct quotient maps or verify the four
   shift-equivalence identities of the second route; and
4. CMGDB evaluates neural box corners with padding rather than an
   outward-rounded interval enclosure of the whole network image.

Accordingly, all reported roots, censuses, Morse sets, indices, graphs, and
overlays are reproducible numerical evidence, not a rigorous recovered Conley
certificate. The v2 itinerary audit is evidence about sampled transitions
only.

## Results

The primary-v2 result and rollout-v2 stress test are reported first. The later
subsections retain the independent v1 result for comparison.

### Version 2 fit, census, and Morse computation

The v2 continuation keeps the accepted v1 encoder/decoder chart fixed and
optimizes only the 2--64--64--2 smooth latent map. It starts from the accepted
v1 GELU checkpoint, uses every one-step analytic pair, and weights the legacy
saddle tubes, legacy origin fan, and new audited origin--\(p_*\)--\(S_2\)
tubes by 4, 3, and 8 respectively. The strict numerical selector stopped
after 3,170 epochs and selected epoch 669. The promoted checkpoint SHA-256 is
`9fbee2cde690d58d2413c0d3521763838abaeb493736120f7173035612da3f3d`.

On the disjoint 20,661-pair model-selection split, the promoted map improves
Patrick's archived baseline mean reconstruction, prediction,
semiconjugacy, and cycle errors by 75.15%, 81.65%, 87.53%, and 69.02%.
Its weighted objective is 1.0332 times the v2 ReLU baseline, below the 1.05
gate. The maximum normalized anchor error is \(5.50\times10^{-5}\), the
maximum characteristic-polynomial relative error is 0.00247, the global
distillation RMSE is 0.00944, and every intended stability/orientation gate
passes.

An independent float64 root solve finds all six named cycles with the correct
least period, unstable dimension, and unstable orientation:

| object | max distance from \(E\)-phase | exact-root multipliers |
|---|---:|---|
| \(P_0\) | \(4.55\times10^{-6}\) | \(0.03822,-0.96889\) |
| \(P_1\) | \(9.43\times10^{-6}\) | \(0.78967\pm0.55148i\), modulus 0.96317 |
| \(S_2\) | \(8.24\times10^{-7}\) | \(0.63512,-1.28348\) |
| \(S_4\) | \(2.41\times10^{-5}\) | \(0.23402,1.94080\) |
| \(p_*\) | \(5.79\times10^{-7}\) | \(0.81855,-1.10766\) |
| \(0\) | \(5.80\times10^{-7}\) | \(-0.12699,2.71496\) |

The independent dense \(p=1,2,4\) census nevertheless finds the same observed
counts as v1: 11 fixed cycles, four least-period-two cycles, and 12
least-period-four cycles. Only six of these 27 cycles are intended; 16 extras
lie inside the fixed CMGDB rectangle and five more lie outside it. Thus the
explicitly witnessed missing transition corridor does not remove the extra
latent recurrence.

The padded corner-sampled CMGDB computation uses the unchanged rectangle
\([-0.340965,0.258329]\times[-0.295279,0.410290]\):

| subdivision \((i,m,M)\) | Morse nodes | boxes | outcome |
|---|---:|---:|---|
| (20,24,26) | 5 | 108,982 | merges \(S_2\) and \(p_*\); exact role-aligned graph gate fails |
| (24,27,28) | 8 | 629,742 | separates all six roles but retains two extra nodes and the wrong \(P_1,S_2\) indices/order |
| (25,28,29) | 11 | 1,133,093 | paper's latent resolution; adds three microscopic trivial nodes near the origin but leaves the seven-node core unchanged |
| (25,28,30) | 8 | 1,133,088 | removes the five-box origin-side splitting and returns exactly to the max-28 graph, while retaining every substantive failure |

The fine graph has edges

```text
2 -> 1, 3 -> 2, 4 -> 3, 5 -> 4, 5 -> 0, 6 -> 5, 7 -> 6
```

with \(P_0,P_1,S_2,S_4,p_*,0\) assigned uniquely to nodes
\(0,4,2,5,3,7\). The observed indices match the requested ones for
\(P_0,S_4,p_*\), and the origin, but \(P_1\) and \(S_2\) are both reported as
trivial. Node 1 is an extra minimal \((x^2-1,0,0)\) component and node 6 an
extra \((0,x-1,0)\) component. The missing requested reachability remains
\(S_2\to P_1\); the observed chain instead contains
\(P_1\to p_*\to S_2\to\text{extra sink}\). The exact six-role graph gate is
false at every tested resolution.

The max-29 graph replaces the final origin edge by a small diamond made from
three additional trivial SCCs of only two, one, and two boxes. At max 30 those
five boxes cease to be recurrent: nodes 0--6 are byte-identical between the
max-29 and max-30 outputs, and max-29 node 10 is geometrically identical to
max-30 node 7 after relabeling. The stable fine result is therefore the
eight-node max-28/max-30 graph shown above. Higher subdivision diagnoses the
max-29 origin-side feature as a grid-level micro-splitting, but confirms the
persistent extra minimal node 1, extra saddle node 6, trivial \(P_1,S_2\)
indices, and missing \(S_2\to P_1\) relation.

### Version 2 projected rollout stress test

To test whether one-step supervision was the limiting factor, a separate
continuation starts from the promoted primary-v2 map, freezes the chart again,
and optimizes the latent map for 6,000 full-batch epochs. The trajectory term
uses horizons

```text
1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64
```

on the saddle tubes, origin fan, and audited origin--\(p_*\)--\(S_2\) tubes.
Components, horizon groups, horizons, and valid starting windows are balanced
hierarchically. After every update, an exact output-layer projection restores
all 16 phasewise anchor equations. Epoch 5,299 is selected after all 6,000
epochs; the run takes 1,438.83 seconds, and the promoted checkpoint SHA-256 is
`707e4b4c59bbf51fc0f9489227043398924aebc4a3c540527f41fa3a103ae6bc`.

On the held-out trajectories, balanced shadowing MSE drops from 0.020402 to
0.006245, a 69.4% reduction. The origin fan improves by 71.0%, the audited
origin--\(p_*\)--\(S_2\) component by 29.3%, and the already-small saddle-tube
error by 0.5%. The weighted one-step validation ratio remains 1.03322 relative
to the v2 ReLU baseline; maximum normalized anchor error is
\(1.70\times10^{-6}\), maximum characteristic-polynomial relative error is
0.00230, global distillation RMSE is 0.00944, and every intended local role
passes. An independent float64 solve again finds all six named cycles with the
correct least periods, unstable dimensions, and orientations.

The global result does not improve. The independent dense census again finds
the same observed lower bound of 27 cycles: 11 fixed, four least-period-two,
and 12 least-period-four. Six are intended, 16 extras lie inside the CMGDB
rectangle, and five lie outside. The coarse \((20,24,26)\) CMGDB computation
also reproduces the primary-v2 five-node graph and the same edges

```text
2 -> 1, 2 -> 0, 3 -> 2, 4 -> 3.
```

It still merges \(S_2\) and \(p_*\), assigns \(P_1\) to a nonminimal trivial
node, and fails the exact role-aligned graph and index gates. Because the
unchanged dense census already violates no-extra-recurrence and the coarse box
graph is identical, a duplicate fine subdivision is not needed to reject this
stress candidate; the primary-v2 fine calculation above remains the available
resolution audit. Multi-step trajectory fidelity therefore sharpens the
sampled itineraries but, without an explicit no-extra-recurrence or certified
outer-map constraint, does not recover the global Morse/Conley structure.

### Version 1 fit quality and intended local dynamics

On the disjoint 12,021-pair model-selection split, the selected smooth map
improves Patrick's archived baseline mean errors by 62.54% in reconstruction,
72.18% in prediction, 83.78% in semiconjugacy, and 74.65% in latent cycle
consistency. Its weighted objective is 1.026 times the curated-data ReLU base,
within the preregistered 1.05 smooth-transfer gate. The maximum normalized
anchor residual is \(8.88\times10^{-5}\), the maximum characteristic-polynomial
relative error is 0.00251, and the global distillation RMSE is 0.00954.

An independent float64 solve of \(G^p(z)-z=0\), initialized from every encoded
phase but not reusing the training-side fixed-anchor Jacobians, finds all six
intended objects with the correct least periods and local roles:

| object | max distance from \(E\)-phase | exact-root multipliers |
|---|---:|---|
| \(P_0\) | \(1.39\times10^{-5}\) | \(0.03826,-0.96884\) |
| \(P_1\) | \(3.98\times10^{-5}\) | \(0.79083\pm0.51994i\), modulus 0.94644 |
| \(S_2\) | \(8.84\times10^{-6}\) | \(0.63943,-1.28476\) |
| \(S_4\) | \(1.32\times10^{-5}\) | \(0.23405,1.94010\) |
| \(p_*\) | \(1.33\times10^{-5}\) | \(0.81940,-1.10803\) |
| \(0\) | \(6.34\times10^{-6}\) | \(-0.14293,2.72071\) |

This is the successful part of the experiment: importance-weighted recurrent
data plus exact local spectral supervision recovers the named recurrent
skeleton locally.

### Version 1 dense recurrent census

Local correctness is not global correctness. A deterministic search of
\(G^p(z)-z=0\), \(p\in\{1,2,4\}\), used 20,961 starts per period equation: an
\(81^2\) grid on the full terminal-tanh square, 8,000 random starts, and 6,400
starts around the encoded phases. Polished roots were reduced to least period,
deduplicated up to cyclic shift, and classified by exact-autograd monodromy.

| least period | cycles found | intended | extra inside CMGDB bounds |
|---:|---:|---:|---:|
| 1 | 11 | 2 | 4 |
| 2 | 4 | 1 | 3 |
| 4 | 12 | 3 | 9 |
| **total** | **27** | **6** | **16** |

The extra cycles occur as close as \(8.05\times10^{-4}\) to the encoded origin,
\(1.68\times10^{-3}\) to \(P_1\), and \(1.71\times10^{-3}\) to \(S_2\). They
include additional sinks, saddles, and repellers. This independently rejects
the candidate under a no-extra-recurrence requirement.

### Version 1 Morse computations

The corner-sampled padded CMGDB calculation was run twice on identical latent
bounds \([-0.340965,0.258329]\times[-0.295279,0.410290]\).

| subdivision \((i,m,M)\) | Morse nodes | boxes | outcome |
|---|---:|---:|---|
| (20,24,26) | 5 | 108,984 | merges \(S_2\) and \(p_*\); only \(P_0,S_4\) have the requested indices |
| (24,27,28) | 8 | 630,529 | separates all six roles but leaves two extra nodes and wrong \(P_1,S_2\) indices/order |

The fine graph has edges

```text
2 -> 1, 3 -> 2, 4 -> 3, 5 -> 4, 5 -> 0, 6 -> 5, 7 -> 6
```

and the role alignment

| role | node | observed index | requested index | match |
|---|---:|---|---|:---:|
| \(P_0\) | 0 | \((x^4-1,0,0)\) | \((x^4-1,0,0)\) | yes |
| \(P_1\) | 4 | \((0,0,0)\) | \((x^4-1,0,0)\) | no |
| \(S_2\) | 2 | \((0,0,0)\) | \((0,x^2+1,0)\) | no |
| \(S_4\) | 5 | \((0,x^4-1,0)\) | \((0,x^4-1,0)\) | yes |
| \(p_*\) | 3 | \((0,x+1,0)\) | \((0,x+1,0)\) | yes |
| \(0\) | 7 | \((0,0,0)\) | \((0,0,0)\) | yes |

Unassigned node 1 is a minimal \((x^2-1,0,0)\) component and node 6 has
\((0,x-1,0)\). All phases of every \(E\)-encoded named orbit lie uniquely in
the stated node, and the independently polished root cycles give the same
membership. The mismatch is therefore not caused by plotting slightly
off-invariant encoded samples.

Five of the six requested positive reachability relations occur. The missing
one is \(S_2\to P_1\); the fine computation instead orders

```text
origin -> extra 6 -> S4 -> P1 -> p* -> S2 -> extra attractor 1
                              \-> P0
```

The exact six-role graph gate is false at both resolutions, and the node count
changes from five to eight under refinement. The experiment therefore does
**not** recover the requested Morse graph or Conley-index inventory. It shows
why an invariant-aware finite data set and excellent local periodic-orbit fit
still do not replace the no-extra-recurrence and index-pair hypotheses in the
ideal argument.

Reproducible artifacts for v1, primary v2, and rollout v2 include their data
manifests, training summaries, independent invariant audits, dense-cycle
censuses, CMGDB outputs, graph renderings, and encoded-invariant overlays under
the corresponding `data/` and `output/` trees. The primary-v2 fine overlay and
graph are under
`output/notebooks/leslie3d_invariant_aware_v2_smooth_fine/seed_20260809/MG/`;
the rollout-v2 stress-test artifacts are under
`output/leslie3d_invariant_aware_v2_smooth_rollout/seed_20260810/`.
