# contamcheck report — `examples/watch_materials.csv`

- tool version: 0.1.0
- master seed: **20261101** (cv_seed=20261101, bootstrap_seed=20261101, nn_seed=20261101)
- label column `conductive` mapped {'no': 0, 'yes': 1}; group column `source_paper`
- rows 1186, positives 480, negatives 706
- feature columns (auto_numeric): dopant_frac, anneal_temp_K, film_thickness_nm, grain_size_nm, lattice_a_A, o2_partial_pressure_mTorr

## Verdict: **WATCH**

One diagnostic crossed its threshold. Treat random-split numbers with caution and report grouped-CV numbers alongside them.

Flags triggered:
- group identity alone predicts the label at AUC 0.8077 >= threshold 0.70

_These cut-offs are defaults chosen by us, not established statistics. Override any of them on the command line and report what you used._

## (a) Can the group column alone predict the label?

| quantity | value |
|---|---|
| leave-one-out target-encoded AUC of group identity | **0.8077** [0.7799, 0.8334] |
| bootstrap draws | 2000 (seed 20261101) |
| rows falling back to the global prior (singleton groups) | 0 |

## (b) Group purity and size distribution

| quantity | value |
|---|---|
| groups | 70 |
| fraction of groups that are single-label | **0.0714** |
| fraction of rows in a label-pure group | 0.0481 |
| fraction of rows in a mixed-label group | 0.9519 |
| group size min / median / mean / max | 8 / 17.0 / 16.9429 / 24 |
| group size p25 / p75 / p95 | 13.0 / 22.0 / 24.0 |
| singleton groups | 0 |
| fraction of rows in the largest group | 0.0202 |

## (c) Random split vs group split, same baseline classifier

Classifier: StandardScaler + LogisticRegression(C=1.0, max_iter=2000); 5-fold, 10 repeats on the random split, cv seed 20261101.

| split | AUC | 95% CI |
|---|---|---|
| random (stratified k-fold) | 0.9847 (mean of repeats); 0.9847 (last repeat) | [0.9784, 0.9900] (last repeat) |
| grouped (GroupKFold) | 0.9852 | [0.9795, 0.9902] |

**delta (grouped - random), paired bootstrap: +0.0005 [-0.0002, 0.0013]**; 8.70% of draws <= 0. Drop from the mean random-split AUC: -0.0005.

## (d) Structure diagnostics

Not computed: None

---

Every number above is also in `report.json` in this directory. Diagnostics are descriptive: they show what the group column encodes on this table, not whether your model is right.
