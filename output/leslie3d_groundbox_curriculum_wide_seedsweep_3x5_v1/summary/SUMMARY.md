# Leslie3D ground-box curriculum 3x5 summary

**Status:** COMPLETE — all 15 cells satisfy the reporting contract

Source sweep: `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/sweep_summary.json` (read-only). Derived files are confined to `summary/`.

The fixed design is five data seeds `2158, 4792, 3174, 688, 5727` by model seeds `0, 1, 2`. Each run must use one continuous full-batch AdamW optimizer for three 4,000-update stages with weights `[1,0,0]`, `[1,1,0]`, and `[1,1,1]`, followed by a fresh 12-step CPU float64 L-BFGS polish of the final joint objective. The saved checkpoint is the float32-cast L-BFGS endpoint; it is not selected by validation and is not described as an epoch. There is no scheduler, patience, or early stopping. Here L1 is reconstruction, L2 one-step prediction, and L3 semiconjugacy. Stage endpoint losses are post-update raw terms; `total` is the stage-weighted sum.

## Inventory and topology

Complete: 15/15; invalid: 0; missing: 0; validation issues: 0.

| data seed | model seed | status | diagnosis | nodes | edges | minimal | sink indices | periodic pass | period-4 pass | output directory |
|---:|---:|:---|:---|---:|---:|---:|:---|:---:|:---:|:---|
| 2158 | 0 | complete | ok | 4 | 3 | 2 | `["(x^4-1, 0, 0)","(x-1, 0, 0)"]` | True | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_2158/seed_0` |
| 2158 | 1 | complete | ok | 4 | 3 | 2 | `["(x^4-1, 0, 0)","(x^2-1, 0, 0)"]` | True | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_2158/seed_1` |
| 2158 | 2 | complete | ok | 4 | 3 | 2 | `["(x-1, 0, 0)","(x^4-1, 0, 0)"]` | True | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_2158/seed_2` |
| 4792 | 0 | complete | ok | 5 | 4 | 2 | `["(x^4-1, 0, 0)","(x^4-1, 0, 0)"]` | True | True | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_4792/seed_0` |
| 4792 | 1 | complete | ok | 3 | 2 | 2 | `["(x-1, 0, 0)","(x^4-1, 0, 0)"]` | True | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_4792/seed_1` |
| 4792 | 2 | complete | ok | 4 | 3 | 2 | `["(x^4-1, 0, 0)","(x-1, x-1, 0)"]` | False | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_4792/seed_2` |
| 3174 | 0 | complete | ok | 6 | 5 | 3 | `["(x^4-1, 0, 0)","(x^4-1, 0, 0)","(x-1, 0, 0)"]` | False | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_3174/seed_0` |
| 3174 | 1 | complete | ok | 3 | 2 | 2 | `["(x-1, 0, 0)","(x^4-1, 0, 0)"]` | True | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_3174/seed_1` |
| 3174 | 2 | complete | ok | 3 | 2 | 2 | `["(x^4-1, 0, 0)","(x-1, 0, 0)"]` | True | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_3174/seed_2` |
| 688 | 0 | complete | ok | 2 | 1 | 1 | `["(x-1, 0, 0)"]` | False | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_688/seed_0` |
| 688 | 1 | complete | ok | 4 | 4 | 1 | `["(x^4-1, 0, 0)"]` | False | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_688/seed_1` |
| 688 | 2 | complete | ok | 4 | 3 | 1 | `["(x^4-1, 0, 0)"]` | False | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_688/seed_2` |
| 5727 | 0 | complete | ok | 5 | 4 | 2 | `["(x^4-1, 0, 0)","(x^4-1, 0, 0)"]` | True | True | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_5727/seed_0` |
| 5727 | 1 | complete | ok | 3 | 2 | 2 | `["(x^4-1, 0, 0)","(x-1, 0, 0)"]` | True | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_5727/seed_1` |
| 5727 | 2 | complete | ok | 3 | 2 | 2 | `["(x^4-1, 0, 0)","(x-1, 0, 0)"]` | True | False | `output/leslie3d_groundbox_curriculum_wide_seedsweep_3x5_v1/dataset_5727/seed_2` |

## Stage 1 endpoint losses

| cell | train L1 | train L2 | train L3 | train total | holdout L1 | holdout L2 | holdout L3 | holdout total |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2158/0 | 9.22362e-04 | 8.94383e-02 | 1.98761e+00 | 9.22362e-04 | 1.26466e-03 | 8.08107e-02 | 1.74830e+00 | 1.26466e-03 |
| 2158/1 | 1.01934e-03 | 8.16110e-02 | 2.48678e+00 | 1.01934e-03 | 1.29804e-03 | 7.21117e-02 | 2.22005e+00 | 1.29804e-03 |
| 2158/2 | 9.54608e-04 | 1.28700e-01 | 2.27368e+00 | 9.54608e-04 | 1.01148e-03 | 1.19487e-01 | 2.05630e+00 | 1.01148e-03 |
| 4792/0 | 1.31616e-03 | 1.00199e-01 | 2.33650e+00 | 1.31616e-03 | 1.49059e-03 | 9.17880e-02 | 2.11434e+00 | 1.49059e-03 |
| 4792/1 | 1.21393e-03 | 8.36767e-02 | 2.07224e+00 | 1.21393e-03 | 1.41506e-03 | 7.58242e-02 | 1.86088e+00 | 1.41506e-03 |
| 4792/2 | 8.01892e-04 | 1.31245e-01 | 2.22735e+00 | 8.01892e-04 | 8.91938e-04 | 1.22691e-01 | 2.02507e+00 | 8.91938e-04 |
| 3174/0 | 9.81329e-04 | 9.06032e-02 | 1.77190e+00 | 9.81329e-04 | 1.01853e-03 | 8.43405e-02 | 1.64723e+00 | 1.01853e-03 |
| 3174/1 | 1.13723e-03 | 7.64096e-02 | 1.80588e+00 | 1.13723e-03 | 1.38894e-03 | 7.13321e-02 | 1.67247e+00 | 1.38894e-03 |
| 3174/2 | 7.69619e-04 | 1.16375e-01 | 2.19712e+00 | 7.69619e-04 | 8.68518e-04 | 1.10121e-01 | 2.06076e+00 | 8.68518e-04 |
| 688/0 | 9.99632e-04 | 9.05005e-02 | 1.90945e+00 | 9.99632e-04 | 1.24694e-03 | 8.29680e-02 | 1.72808e+00 | 1.24694e-03 |
| 688/1 | 1.25245e-03 | 1.01364e-01 | 2.09469e+00 | 1.25245e-03 | 1.34038e-03 | 9.50450e-02 | 1.89272e+00 | 1.34038e-03 |
| 688/2 | 9.14244e-04 | 1.14814e-01 | 2.24572e+00 | 9.14244e-04 | 9.46032e-04 | 1.06916e-01 | 2.06659e+00 | 9.46032e-04 |
| 5727/0 | 1.12122e-03 | 1.06636e-01 | 1.09880e+00 | 1.12122e-03 | 1.21461e-03 | 9.71952e-02 | 9.64601e-01 | 1.21461e-03 |
| 5727/1 | 1.12326e-03 | 8.66420e-02 | 2.16181e+00 | 1.12326e-03 | 1.25000e-03 | 7.94348e-02 | 1.92744e+00 | 1.25000e-03 |
| 5727/2 | 8.11879e-04 | 1.52686e-01 | 2.16059e+00 | 8.11879e-04 | 1.06310e-03 | 1.45253e-01 | 1.91432e+00 | 1.06310e-03 |

## Stage 2 endpoint losses

| cell | train L1 | train L2 | train L3 | train total | holdout L1 | holdout L2 | holdout L3 | holdout total |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2158/0 | 8.32131e-04 | 7.69534e-04 | 9.30896e+00 | 1.60167e-03 | 1.01382e-03 | 9.03173e-04 | 8.52852e+00 | 1.91699e-03 |
| 2158/1 | 7.96997e-04 | 7.44551e-04 | 2.99442e+00 | 1.54155e-03 | 9.32734e-04 | 9.21499e-04 | 2.76322e+00 | 1.85423e-03 |
| 2158/2 | 9.83456e-04 | 7.80478e-04 | 5.25504e+00 | 1.76393e-03 | 1.11005e-03 | 9.47097e-04 | 4.71492e+00 | 2.05714e-03 |
| 4792/0 | 1.18949e-03 | 8.44424e-04 | 7.07488e+00 | 2.03392e-03 | 1.47782e-03 | 1.03415e-03 | 6.50977e+00 | 2.51197e-03 |
| 4792/1 | 1.06246e-03 | 6.72993e-04 | 8.42356e-01 | 1.73546e-03 | 1.34578e-03 | 8.59164e-04 | 7.64834e-01 | 2.20494e-03 |
| 4792/2 | 8.32875e-04 | 6.45023e-04 | 6.13707e+00 | 1.47790e-03 | 1.05050e-03 | 8.15254e-04 | 5.55466e+00 | 1.86575e-03 |
| 3174/0 | 9.71672e-04 | 7.41410e-04 | 4.22533e+00 | 1.71308e-03 | 1.24606e-03 | 8.40483e-04 | 4.03315e+00 | 2.08654e-03 |
| 3174/1 | 9.94021e-04 | 8.59517e-04 | 6.69080e-02 | 1.85354e-03 | 1.04912e-03 | 9.00288e-04 | 6.86568e-02 | 1.94941e-03 |
| 3174/2 | 7.84892e-04 | 7.58668e-04 | 3.95942e+00 | 1.54356e-03 | 9.37082e-04 | 9.38549e-04 | 3.72473e+00 | 1.87563e-03 |
| 688/0 | 9.86640e-04 | 8.82059e-04 | 3.81055e+00 | 1.86870e-03 | 1.02699e-03 | 9.53895e-04 | 3.51320e+00 | 1.98088e-03 |
| 688/1 | 1.14649e-03 | 6.71918e-04 | 3.52254e+00 | 1.81841e-03 | 1.39087e-03 | 8.82787e-04 | 3.20244e+00 | 2.27365e-03 |
| 688/2 | 8.78464e-04 | 7.54573e-04 | 6.27246e+00 | 1.63304e-03 | 1.18459e-03 | 9.52824e-04 | 5.77709e+00 | 2.13741e-03 |
| 5727/0 | 1.55930e-03 | 1.44534e-03 | 2.20139e+00 | 3.00464e-03 | 1.51099e-03 | 1.56214e-03 | 2.01059e+00 | 3.07313e-03 |
| 5727/1 | 1.01690e-03 | 9.27841e-04 | 5.11372e-01 | 1.94474e-03 | 1.18258e-03 | 1.08191e-03 | 4.72313e-01 | 2.26449e-03 |
| 5727/2 | 9.80512e-04 | 8.72066e-04 | 1.03480e+01 | 1.85258e-03 | 1.32331e-03 | 1.08686e-03 | 9.33683e+00 | 2.41018e-03 |

## Stage 3 endpoint losses

| cell | train L1 | train L2 | train L3 | train total | holdout L1 | holdout L2 | holdout L3 | holdout total |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2158/0 | 1.58939e-03 | 8.54228e-04 | 3.08800e-04 | 2.75242e-03 | 1.87907e-03 | 1.12712e-03 | 3.47659e-04 | 3.35385e-03 |
| 2158/1 | 1.36458e-03 | 1.52385e-03 | 3.76135e-04 | 3.26456e-03 | 1.51456e-03 | 1.77205e-03 | 4.31418e-04 | 3.71802e-03 |
| 2158/2 | 1.70165e-03 | 8.07311e-04 | 3.31404e-04 | 2.84036e-03 | 2.10705e-03 | 1.15834e-03 | 3.65349e-04 | 3.63074e-03 |
| 4792/0 | 1.30658e-03 | 8.46638e-04 | 3.25040e-04 | 2.47826e-03 | 1.71790e-03 | 1.06726e-03 | 3.58990e-04 | 3.14415e-03 |
| 4792/1 | 1.25788e-03 | 7.58485e-04 | 3.18164e-04 | 2.33453e-03 | 1.64229e-03 | 9.71995e-04 | 3.67299e-04 | 2.98158e-03 |
| 4792/2 | 1.57957e-03 | 1.47128e-03 | 6.74616e-04 | 3.72547e-03 | 2.01661e-03 | 1.84381e-03 | 7.53508e-04 | 4.61393e-03 |
| 3174/0 | 1.61437e-03 | 1.07499e-03 | 2.69824e-04 | 2.95919e-03 | 1.94180e-03 | 1.30121e-03 | 3.63937e-04 | 3.60695e-03 |
| 3174/1 | 1.15108e-03 | 6.66891e-04 | 3.28350e-04 | 2.14632e-03 | 1.40891e-03 | 8.14577e-04 | 3.81504e-04 | 2.60499e-03 |
| 3174/2 | 1.61523e-03 | 1.21529e-03 | 4.17272e-04 | 3.24780e-03 | 2.03459e-03 | 1.37334e-03 | 4.51085e-04 | 3.85902e-03 |
| 688/0 | 1.84924e-03 | 8.72928e-04 | 3.10610e-04 | 3.03278e-03 | 2.13205e-03 | 1.16686e-03 | 3.41365e-04 | 3.64028e-03 |
| 688/1 | 1.73026e-03 | 1.16794e-03 | 4.75764e-04 | 3.37397e-03 | 2.01152e-03 | 1.41400e-03 | 5.40828e-04 | 3.96634e-03 |
| 688/2 | 1.45839e-03 | 1.36561e-03 | 3.91644e-04 | 3.21564e-03 | 1.65286e-03 | 1.61387e-03 | 4.71909e-04 | 3.73864e-03 |
| 5727/0 | 1.74710e-03 | 1.43505e-03 | 4.59123e-04 | 3.64127e-03 | 2.47461e-03 | 1.82198e-03 | 5.36938e-04 | 4.83353e-03 |
| 5727/1 | 1.32302e-03 | 9.19436e-04 | 4.37122e-04 | 2.67957e-03 | 1.46979e-03 | 1.09756e-03 | 4.53286e-04 | 3.02064e-03 |
| 5727/2 | 1.78328e-03 | 9.19068e-04 | 3.14887e-04 | 3.01723e-03 | 2.07820e-03 | 1.11021e-03 | 3.47158e-04 | 3.53557e-03 |

## AdamW-to-L-BFGS polish

Deltas are `final float32 checkpoint - AdamW endpoint`; a negative training delta is an improvement. Holdout deltas are reported but were not used for optimization or checkpoint selection.

| cell | AdamW train total | final train total | train delta | AdamW holdout total | final holdout total | holdout delta | L-BFGS iterations | closure evaluations |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2158/0 | 2.75242e-03 | 2.56124e-03 | -1.91183e-04 | 3.35385e-03 | 3.19315e-03 | -1.60699e-04 | 120 | 134 |
| 2158/1 | 3.26456e-03 | 3.14990e-03 | -1.14655e-04 | 3.71802e-03 | 3.59385e-03 | -1.24177e-04 | 120 | 134 |
| 2158/2 | 2.84036e-03 | 2.54853e-03 | -2.91829e-04 | 3.63074e-03 | 3.34665e-03 | -2.84089e-04 | 120 | 133 |
| 4792/0 | 2.47826e-03 | 2.27694e-03 | -2.01320e-04 | 3.14415e-03 | 2.94878e-03 | -1.95371e-04 | 120 | 134 |
| 4792/1 | 2.33453e-03 | 2.20100e-03 | -1.33526e-04 | 2.98158e-03 | 2.82995e-03 | -1.51630e-04 | 120 | 135 |
| 4792/2 | 3.72547e-03 | 2.54973e-03 | -1.17574e-03 | 4.61393e-03 | 3.42180e-03 | -1.19213e-03 | 120 | 135 |
| 3174/0 | 2.95919e-03 | 2.89325e-03 | -6.59446e-05 | 3.60695e-03 | 3.54594e-03 | -6.10140e-05 | 120 | 135 |
| 3174/1 | 2.14632e-03 | 1.96867e-03 | -1.77655e-04 | 2.60499e-03 | 2.44939e-03 | -1.55607e-04 | 120 | 134 |
| 3174/2 | 3.24780e-03 | 2.92807e-03 | -3.19729e-04 | 3.85902e-03 | 3.58199e-03 | -2.77032e-04 | 120 | 134 |
| 688/0 | 3.03278e-03 | 2.91349e-03 | -1.19288e-04 | 3.64028e-03 | 3.52247e-03 | -1.17810e-04 | 120 | 134 |
| 688/1 | 3.37397e-03 | 3.15010e-03 | -2.23869e-04 | 3.96634e-03 | 3.69865e-03 | -2.67699e-04 | 120 | 134 |
| 688/2 | 3.21564e-03 | 3.06830e-03 | -1.47338e-04 | 3.73864e-03 | 3.57534e-03 | -1.63299e-04 | 120 | 134 |
| 5727/0 | 3.64127e-03 | 3.05511e-03 | -5.86162e-04 | 4.83353e-03 | 4.19081e-03 | -6.42713e-04 | 120 | 134 |
| 5727/1 | 2.67957e-03 | 2.10497e-03 | -5.74600e-04 | 3.02064e-03 | 2.45935e-03 | -5.61296e-04 | 120 | 133 |
| 5727/2 | 3.01723e-03 | 2.88799e-03 | -1.29238e-04 | 3.53557e-03 | 3.41011e-03 | -1.25461e-04 | 120 | 134 |

## Final float32-checkpoint losses

| cell | train L1 | train L2 | train L3 | train total | holdout L1 | holdout L2 | holdout L3 | holdout total |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2158/0 | 1.53119e-03 | 7.74886e-04 | 2.55161e-04 | 2.56124e-03 | 1.83414e-03 | 1.06076e-03 | 2.98248e-04 | 3.19315e-03 |
| 2158/1 | 1.33644e-03 | 1.46938e-03 | 3.44083e-04 | 3.14990e-03 | 1.49144e-03 | 1.70901e-03 | 3.93399e-04 | 3.59385e-03 |
| 2158/2 | 1.58337e-03 | 7.15171e-04 | 2.49996e-04 | 2.54853e-03 | 2.02794e-03 | 1.05028e-03 | 2.68427e-04 | 3.34665e-03 |
| 4792/0 | 1.23483e-03 | 7.64679e-04 | 2.77434e-04 | 2.27694e-03 | 1.64988e-03 | 9.87028e-04 | 3.11869e-04 | 2.94878e-03 |
| 4792/1 | 1.21264e-03 | 7.32981e-04 | 2.55385e-04 | 2.20100e-03 | 1.59016e-03 | 9.41449e-04 | 2.98340e-04 | 2.82995e-03 |
| 4792/2 | 1.23734e-03 | 9.88666e-04 | 3.23726e-04 | 2.54973e-03 | 1.70892e-03 | 1.33607e-03 | 3.76808e-04 | 3.42180e-03 |
| 3174/0 | 1.57545e-03 | 1.07596e-03 | 2.41840e-04 | 2.89325e-03 | 1.93168e-03 | 1.29413e-03 | 3.20128e-04 | 3.54594e-03 |
| 3174/1 | 1.12158e-03 | 6.01977e-04 | 2.45113e-04 | 1.96867e-03 | 1.40186e-03 | 7.52627e-04 | 2.94897e-04 | 2.44939e-03 |
| 3174/2 | 1.57139e-03 | 1.03862e-03 | 3.18062e-04 | 2.92807e-03 | 2.03954e-03 | 1.19896e-03 | 3.43485e-04 | 3.58199e-03 |
| 688/0 | 1.78665e-03 | 8.47869e-04 | 2.78972e-04 | 2.91349e-03 | 2.06657e-03 | 1.14287e-03 | 3.13030e-04 | 3.52247e-03 |
| 688/1 | 1.67631e-03 | 1.07802e-03 | 3.95776e-04 | 3.15010e-03 | 1.95368e-03 | 1.29924e-03 | 4.45721e-04 | 3.69865e-03 |
| 688/2 | 1.41939e-03 | 1.31929e-03 | 3.29630e-04 | 3.06830e-03 | 1.60560e-03 | 1.57059e-03 | 3.99151e-04 | 3.57534e-03 |
| 5727/0 | 1.55553e-03 | 1.16545e-03 | 3.34132e-04 | 3.05511e-03 | 2.25503e-03 | 1.52916e-03 | 4.06630e-04 | 4.19081e-03 |
| 5727/1 | 1.14162e-03 | 6.94852e-04 | 2.68503e-04 | 2.10497e-03 | 1.29816e-03 | 8.71492e-04 | 2.89693e-04 | 2.45935e-03 |
| 5727/2 | 1.74823e-03 | 8.63987e-04 | 2.75774e-04 | 2.88799e-03 | 2.04876e-03 | 1.05540e-03 | 3.05961e-04 | 3.41011e-03 |

## Artifact provenance

Exact per-cell final and AdamW-endpoint checkpoint paths, training-summary, history, Morse-graph, Morse-set, metric, diagnosis, and manifest paths plus SHA-256 hashes are in `cells.csv`. The CSV also records the frozen optimizer settings, actual L-BFGS iterations and closure evaluations, latent CMGDB bounds, and the raw per-component sampled residual/tolerance values. The sampled inequality is a diagnostic, not a classifier of spuriousness or invariant-set correspondence. The saved Morse graph, rather than a manifest default, is the authoritative source for node, edge, sink, index, and pass fields.
