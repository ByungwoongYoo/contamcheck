# contamcheck report — `results/tables/gate_cotested.tsv`

- tool version: 0.1.0
- master seed: **20261101** (cv_seed=0, bootstrap_seed=3, nn_seed=7)
- label column `label` mapped {'0': 0, '1': 1}; group column `document`
- rows 586, positives 296, negatives 290
- feature columns (explicit): ompf_percentile, multiporin_percentile, lipid_percentile, efflux_percentile, activity_percentile

## Verdict: **CONTAMINATED**

Two or more diagnostics crossed their thresholds. The label is substantially recoverable from group membership rather than from the property you intend to model. Random-split performance on this table is not trustworthy; grouped cross-validation is mandatory, and the headline number should be the grouped one.

Flags triggered:
- group identity alone predicts the label at AUC 0.8858 >= threshold 0.70
- 96.8% of groups are single-label >= threshold 80%
- same-class nearest-neighbour similarity exceeds cross-class by 0.2554 >= threshold 0.20 (the two classes sit in near-disjoint chemical space)
- between-class Murcko scaffold Jaccard 0.0165 <= threshold 0.10

_These cut-offs are defaults chosen by us, not established statistics. Override any of them on the command line and report what you used._

## (a) Can the group column alone predict the label?

| quantity | value |
|---|---|
| leave-one-out target-encoded AUC of group identity | **0.8858** [0.8627, 0.9086] |
| bootstrap draws | 2000 (seed 3) |
| rows falling back to the global prior (singleton groups) | 226 |

## (b) Group purity and size distribution

| quantity | value |
|---|---|
| groups | 346 |
| fraction of groups that are single-label | **0.9682** |
| fraction of rows in a label-pure group | 0.9471 |
| fraction of rows in a mixed-label group | 0.0529 |
| group size min / median / mean / max | 1 / 1.0 / 1.6936 / 9 |
| group size p25 / p75 / p95 | 1.0 / 2.0 / 4.0 |
| singleton groups | 226 |
| fraction of rows in the largest group | 0.0154 |

## (c) Random split vs group split, same baseline classifier

Classifier: StandardScaler + LogisticRegression(C=1.0, max_iter=2000); 5-fold, 10 repeats on the random split, cv seed 0.

| split | AUC | 95% CI |
|---|---|---|
| random (stratified k-fold) | 0.7213 (mean of repeats); 0.7180 (last repeat) | [0.6743, 0.7588] (last repeat) |
| grouped (GroupKFold) | 0.7107 | [0.6691, 0.7503] |

**delta (grouped - random), paired bootstrap: -0.0073 [-0.0169, 0.0021]**; 93.30% of draws <= 0. Drop from the mean random-split AUC: +0.0106.

## (c2) Model arm vs comparison arm, under both splits

| split | mechgate | baseline | delta (A - B) | 95% CI |
|---|---|---|---|---|
| random | 0.7213 | 0.7430 | -0.0220 | [-0.0460, 0.0026] |
| grouped | 0.7107 | 0.7425 | -0.0318 | [-0.0576, -0.0062] |

Shift in A's advantage, random -> grouped: -0.0098.

## (d) Structure diagnostics

| quantity | value |
|---|---|
| molecules parsed | 586 of 586 |
| nearest-neighbour Tanimoto, same class (mean over 586 sampled queries) | 0.5600 |
| nearest-neighbour Tanimoto, cross class | 0.3046 |
| Murcko scaffolds, positive / negative / shared | 189 / 241 / 7 |
| between-class scaffold Jaccard | 0.0165 |
| mean fraction of test scaffolds also seen in train, random split | 0.3155 |
| mean fraction of test scaffolds also seen in train, group split | 0.0771 |

---

Every number above is also in `report.json` in this directory. Diagnostics are descriptive: they show what the group column encodes on this table, not whether your model is right.
