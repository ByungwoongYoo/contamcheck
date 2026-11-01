# contamcheck report — `results/tables/naive_full.tsv`

- tool version: 0.1.0
- master seed: **20261101** (cv_seed=20261101, bootstrap_seed=1, nn_seed=7)
- label column `label` mapped {'0': 0, '1': 1}; group column `document`
- rows 11996, positives 2358, negatives 9638
- feature columns (derived_from_smiles): cc_mw, cc_clogp, cc_tpsa, cc_hbd, cc_hba, cc_rotb, cc_rings, cc_heavy, cc_formal_charge, cc_fracsp3

## Verdict: **CONTAMINATED**

Two or more diagnostics crossed their thresholds. The label is substantially recoverable from group membership rather than from the property you intend to model. Random-split performance on this table is not trustworthy; grouped cross-validation is mandatory, and the headline number should be the grouped one.

Flags triggered:
- group identity alone predicts the label at AUC 0.9929 >= threshold 0.70
- 97.1% of groups are single-label >= threshold 80%
- same-class nearest-neighbour similarity exceeds cross-class by 0.4906 >= threshold 0.20 (the two classes sit in near-disjoint chemical space)
- between-class Murcko scaffold Jaccard 0.0099 <= threshold 0.10

_These cut-offs are defaults chosen by us, not established statistics. Override any of them on the command line and report what you used._

## (a) Can the group column alone predict the label?

| quantity | value |
|---|---|
| leave-one-out target-encoded AUC of group identity | **0.9929** [0.9909, 0.9949] |
| bootstrap draws | 2000 (seed 1) |
| rows falling back to the global prior (singleton groups) | 371 |

## (b) Group purity and size distribution

| quantity | value |
|---|---|
| groups | 1418 |
| fraction of groups that are single-label | **0.9711** |
| fraction of rows in a label-pure group | 0.9637 |
| fraction of rows in a mixed-label group | 0.0363 |
| group size min / median / mean / max | 1 / 4.0 / 8.4598 / 105 |
| group size p25 / p75 / p95 | 1.0 / 12.0 / 29.1 |
| singleton groups | 371 |
| fraction of rows in the largest group | 0.0088 |

## (c) Random split vs group split, same baseline classifier

Classifier: StandardScaler + LogisticRegression(C=1.0, max_iter=2000); 5-fold, 10 repeats on the random split, cv seed 20261101.

| split | AUC | 95% CI |
|---|---|---|
| random (stratified k-fold) | 0.7717 (mean of repeats); 0.7719 (last repeat) | [0.7617, 0.7827] (last repeat) |
| grouped (GroupKFold) | 0.7424 | [0.7315, 0.7534] |

**delta (grouped - random), paired bootstrap: -0.0295 [-0.0321, -0.0271]**; 100.00% of draws <= 0. Drop from the mean random-split AUC: +0.0294.

## (d) Structure diagnostics

| quantity | value |
|---|---|
| molecules parsed | 11996 of 11996 |
| nearest-neighbour Tanimoto, same class (mean over 800 sampled queries) | 0.8161 |
| nearest-neighbour Tanimoto, cross class | 0.3255 |
| Murcko scaffolds, positive / negative / shared | 1055 / 3949 / 49 |
| between-class scaffold Jaccard | 0.0099 |
| mean fraction of test scaffolds also seen in train, random split | 0.5491 |
| mean fraction of test scaffolds also seen in train, group split | 0.1141 |

---

Every number above is also in `report.json` in this directory. Diagnostics are descriptive: they show what the group column encodes on this table, not whether your model is right.
