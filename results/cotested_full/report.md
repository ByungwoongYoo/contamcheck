# contamcheck report — `results/tables/cotested_full.tsv`

- tool version: 0.1.0
- master seed: **20261101** (cv_seed=20261101, bootstrap_seed=1, nn_seed=7)
- label column `label` mapped {'0': 0, '1': 1}; group column `document`
- rows 4029, positives 539, negatives 3490
- feature columns (derived_from_smiles): cc_mw, cc_clogp, cc_tpsa, cc_hbd, cc_hba, cc_rotb, cc_rings, cc_heavy, cc_formal_charge, cc_fracsp3

## Verdict: **CONTAMINATED**

Two or more diagnostics crossed their thresholds. The label is substantially recoverable from group membership rather than from the property you intend to model. Random-split performance on this table is not trustworthy; grouped cross-validation is mandatory, and the headline number should be the grouped one.

Flags triggered:
- group identity alone predicts the label at AUC 0.9468 >= threshold 0.70
- 93.4% of groups are single-label >= threshold 80%
- same-class nearest-neighbour similarity exceeds cross-class by 0.4465 >= threshold 0.20 (the two classes sit in near-disjoint chemical space)
- between-class Murcko scaffold Jaccard 0.0243 <= threshold 0.10

_These cut-offs are defaults chosen by us, not established statistics. Override any of them on the command line and report what you used._

## (a) Can the group column alone predict the label?

| quantity | value |
|---|---|
| leave-one-out target-encoded AUC of group identity | **0.9468** [0.9345, 0.9581] |
| bootstrap draws | 2000 (seed 1) |
| rows falling back to the global prior (singleton groups) | 344 |

## (b) Group purity and size distribution

| quantity | value |
|---|---|
| groups | 905 |
| fraction of groups that are single-label | **0.9337** |
| fraction of rows in a label-pure group | 0.9308 |
| fraction of rows in a mixed-label group | 0.0692 |
| group size min / median / mean / max | 1 / 2.0 / 4.4519 / 81 |
| group size p25 / p75 / p95 | 1.0 / 5.0 / 16.0 |
| singleton groups | 344 |
| fraction of rows in the largest group | 0.0201 |

## (c) Random split vs group split, same baseline classifier

Classifier: StandardScaler + LogisticRegression(C=1.0, max_iter=2000); 5-fold, 10 repeats on the random split, cv seed 20261101.

| split | AUC | 95% CI |
|---|---|---|
| random (stratified k-fold) | 0.7402 (mean of repeats); 0.7403 (last repeat) | [0.7158, 0.7652] (last repeat) |
| grouped (GroupKFold) | 0.7288 | [0.7048, 0.7527] |

**delta (grouped - random), paired bootstrap: -0.0115 [-0.0151, -0.0078]**; 100.00% of draws <= 0. Drop from the mean random-split AUC: +0.0114.

## (d) Structure diagnostics

| quantity | value |
|---|---|
| molecules parsed | 4029 of 4029 |
| nearest-neighbour Tanimoto, same class (mean over 800 sampled queries) | 0.7658 |
| nearest-neighbour Tanimoto, cross class | 0.3193 |
| Murcko scaffolds, positive / negative / shared | 294 / 1517 / 43 |
| between-class scaffold Jaccard | 0.0243 |
| mean fraction of test scaffolds also seen in train, random split | 0.5321 |
| mean fraction of test scaffolds also seen in train, group split | 0.1010 |

---

Every number above is also in `report.json` in this directory. Diagnostics are descriptive: they show what the group column encodes on this table, not whether your model is right.
