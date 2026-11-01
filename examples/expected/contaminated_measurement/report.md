# contamcheck report — `examples/contaminated_measurement.csv`

- tool version: 0.1.0
- master seed: **20261101** (cv_seed=20261101, bootstrap_seed=20261101, nn_seed=20261101)
- label column `corrosion_pass` mapped {'0': 0, '1': 1}; group column `campaign`
- rows 1490, positives 815, negatives 675
- feature columns (auto_numeric): cr_wt_pct, ni_wt_pct, mo_wt_pct, hardness_HV, surface_roughness_um, chloride_ppm

## Verdict: **CONTAMINATED**

Two or more diagnostics crossed their thresholds. The label is substantially recoverable from group membership rather than from the property you intend to model. Random-split performance on this table is not trustworthy; grouped cross-validation is mandatory, and the headline number should be the grouped one.

Flags triggered:
- group identity alone predicts the label at AUC 0.9415 >= threshold 0.70
- baseline AUC fell by 0.1591 from random to grouped split >= threshold 0.05

_These cut-offs are defaults chosen by us, not established statistics. Override any of them on the command line and report what you used._

## (a) Can the group column alone predict the label?

| quantity | value |
|---|---|
| leave-one-out target-encoded AUC of group identity | **0.9415** [0.9258, 0.9547] |
| bootstrap draws | 2000 (seed 20261101) |
| rows falling back to the global prior (singleton groups) | 0 |

## (b) Group purity and size distribution

| quantity | value |
|---|---|
| groups | 48 |
| fraction of groups that are single-label | **0.1667** |
| fraction of rows in a label-pure group | 0.1242 |
| fraction of rows in a mixed-label group | 0.8758 |
| group size min / median / mean / max | 15 / 32.0 / 31.0417 / 44 |
| group size p25 / p75 / p95 | 23.0 / 38.2 / 43.0 |
| singleton groups | 0 |
| fraction of rows in the largest group | 0.0295 |

## (c) Random split vs group split, same baseline classifier

Classifier: StandardScaler + LogisticRegression(C=1.0, max_iter=2000); 5-fold, 10 repeats on the random split, cv seed 20261101.

| split | AUC | 95% CI |
|---|---|---|
| random (stratified k-fold) | 0.6165 (mean of repeats); 0.6168 (last repeat) | [0.5895, 0.6441] (last repeat) |
| grouped (GroupKFold) | 0.4577 | [0.4279, 0.4865] |

**delta (grouped - random), paired bootstrap: -0.1591 [-0.1771, -0.1414]**; 100.00% of draws <= 0. Drop from the mean random-split AUC: +0.1588.

## (d) Structure diagnostics

Not computed: None

---

Every number above is also in `report.json` in this directory. Diagnostics are descriptive: they show what the group column encodes on this table, not whether your model is right.
