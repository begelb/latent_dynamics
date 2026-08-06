# Leslie3D 3x5 RMSE-bounded CMGDB replay

No models were retrained. The complete original output tree was cloned, and 
each new analysis was written under `rmse_bounded_cmgdb/` in the cloned cell.

Policy: choose the coarsest total depth at which both nominal box side 
lengths are no larger than the smaller of final train and holdout latent RMSE.

| cell | target RMSE | ladder | actual max widths | baseline minima | RMSE-grid minima | RMSE-grid indices |
|---|---:|---|---|---:|---:|---|
| 688/0 | 0.0167025 | 12/15/16 | (0.00844917, 0.00791058) | 1 | 1 | `["(0, 0, 0)"]` |
| 688/1 | 0.0198941 | 11/14/15 | (0.0116072, 0.0147079) | 1 | 1 | `["(x-1, 0, 0)"]` |
| 688/2 | 0.0181557 | 11/14/15 | (0.010987, 0.0173516) | 1 | 1 | `["(x-1, 0, 0)"]` |
| 2158/0 | 0.0159738 | 13/16/17 | (0.00769894, 0.0114198) | 2 | 1 | `["(x-1, 0, 0)"]` |
| 2158/1 | 0.0185495 | 11/14/15 | (0.00930339, 0.010162) | 2 | 1 | `["(0, 0, 0)"]` |
| 2158/2 | 0.0158113 | 13/16/17 | (0.00714218, 0.0090577) | 2 | 1 | `["(x-1, 0, 0)"]` |
| 3174/0 | 0.0155512 | 12/15/16 | (0.0120956, 0.00800823) | 3 | 1 | `["(x-1, 0, 0)"]` |
| 3174/1 | 0.0156561 | 11/14/15 | (0.0146775, 0.0146821) | 2 | 1 | `["(x-1, 0, 0)"]` |
| 3174/2 | 0.0178343 | 12/15/16 | (0.0124627, 0.0103481) | 2 | 1 | `["(x-1, 0, 0)"]` |
| 4792/0 | 0.0166564 | 11/14/15 | (0.016402, 0.0126178) | 2 | 1 | `["(x-1, 0, 0)"]` |
| 4792/1 | 0.0159808 | 11/14/15 | (0.0113537, 0.0159407) | 2 | 1 | `["(x-1, 0, 0)"]` |
| 4792/2 | 0.0179924 | 12/15/16 | (0.0112229, 0.0145105) | 2 | 1 | `["(x-1, 0, 0)"]` |
| 5727/0 | 0.0182793 | 11/14/15 | (0.0129933, 0.0112367) | 2 | 1 | `["(x-1, 0, 0)"]` |
| 5727/1 | 0.0163861 | 13/16/17 | (0.00852449, 0.0103444) | 2 | 1 | `["(x-1, 0, 0)"]` |
| 5727/2 | 0.0166064 | 12/15/16 | (0.00832214, 0.0101055) | 2 | 1 | `["(0, 0, 0)"]` |

All width bounds pass: `True`.
Total CMGDB wall time: `2.656` seconds.

These coarse outer approximations may merge recurrent pieces or add order 
relations. Their topology is a resolution-sensitivity diagnostic, not a 
nonexistence certificate for finer-grid learned invariant sets.
