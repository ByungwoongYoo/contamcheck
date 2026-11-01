# contamcheck report — `examples/clean_manufacturing.csv`

- tool version: 0.1.0
- master seed: **20261101** (cv_seed=20261101, bootstrap_seed=20261101, nn_seed=20261101)
- label column `defect` mapped {'0': 0, '1': 1}; group column `batch`
- rows 1200, positives 560, negatives 640
- feature columns (auto_numeric): melt_temp_C, hold_pressure_bar, cooling_time_s, humidity_pct, screw_speed_rpm

## Verdict: **CLEAN**

No threshold was crossed. The grouping column does not visibly encode the label on this table. Grouped CV is still good practice, but these diagnostics give no evidence of a provenance shortcut.

No flags triggered.

_These cut-offs are defaults chosen by us, not established statistics. Override any of them on the command line and report what you used._

## (a) Can the group column alone predict the label?

| quantity | value |
|---|---|
| leave-one-out target-encoded AUC of group identity | **0.4558** [0.4235, 0.4877] |
| bootstrap draws | 2000 (seed 20261101) |
| rows falling back to the global prior (singleton groups) | 0 |

## (b) Group purity and size distribution

| quantity | value |
|---|---|
| groups | 60 |
| fraction of groups that are single-label | **0.0000** |
| fraction of rows in a label-pure group | 0.0000 |
| fraction of rows in a mixed-label group | 1.0000 |
| group size min / median / mean / max | 20 / 20.0 / 20.0000 / 20 |
| group size p25 / p75 / p95 | 20.0 / 20.0 / 20.0 |
| singleton groups | 0 |
| fraction of rows in the largest group | 0.0167 |

## (c) Random split vs group split, same baseline classifier

Classifier: StandardScaler + LogisticRegression(C=1.0, max_iter=2000); 5-fold, 10 repeats on the random split, cv seed 20261101.

| split | AUC | 95% CI |
|---|---|---|
| random (stratified k-fold) | 0.7932 (mean of repeats); 0.7922 (last repeat) | [0.7671, 0.8167] (last repeat) |
| grouped (GroupKFold) | 0.7898 | [0.7649, 0.8157] |

**delta (grouped - random), paired bootstrap: -0.0023 [-0.0056, 0.0008]**; 93.50% of draws <= 0. Drop from the mean random-split AUC: +0.0034.

## (d) Structure diagnostics

Not computed: None

---

Every number above is also in `report.json` in this directory. Diagnostics are descriptive: they show what the group column encodes on this table, not whether your model is right.
