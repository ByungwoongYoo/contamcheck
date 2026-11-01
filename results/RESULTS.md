# Results

Worked example only: a public bioactivity dataset (binary active/inactive label, grouping column = source publication `document`). All numbers are transcribed from `results/<run>/report.json`.

| run | rows | positives / negatives | groups | features | verdict (flags) |
|---|---|---|---|---|---|
| naive_full | 11996 | 2358 / 9638 | 1418 | 10 descriptors derived from SMILES | CONTAMINATED (4) |
| cotested_full | 4029 | 539 / 3490 | 905 | 10 descriptors derived from SMILES | CONTAMINATED (4) |
| gate_cotested | 586 | 296 / 290 | 346 | 5 explicit `*_percentile` columns | CONTAMINATED (4) |

## Before / after: random split vs group-held-out CV
Same classifier (StandardScaler + LogisticRegression), 5 folds; random = StratifiedKFold, 10 repeats; grouped = GroupKFold.

| run | random AUC (mean of repeats) | random AUC last repeat [95% CI] | grouped AUC [95% CI] | delta grouped − random [95% CI] | draws ≤ 0 |
|---|---|---|---|---|---|
| naive_full | 0.7717 | 0.7719 [0.7617, 0.7827] | 0.7424 [0.7315, 0.7534] | −0.0295 [−0.0321, −0.0271] | 1.0 |
| cotested_full | 0.7402 | 0.7403 [0.7158, 0.7652] | 0.7288 [0.7048, 0.7527] | −0.0115 [−0.0151, −0.0078] | 1.0 |
| gate_cotested | 0.7213 | 0.7180 [0.6743, 0.7588] | 0.7107 [0.6691, 0.7503] | −0.0073 [−0.0169, 0.0021] | 0.933 |

No run crossed the split-drop threshold (−0.05), so the verdicts come from the other four flags.

## Metadata-only diagnostic (group column alone, no features)

| run | group-identity AUC [95% CI] | groups single-label | rows in single-label groups | singleton groups (prior fallback) | median / max group size |
|---|---|---|---|---|---|
| naive_full | 0.9929 [0.9909, 0.9949] | 0.9711 | 0.9637 | 371 | 4.0 / 105 |
| cotested_full | 0.9468 [0.9345, 0.9581] | 0.9337 | 0.9308 | 344 | 2.0 / 81 |
| gate_cotested | 0.8858 [0.8627, 0.9086] | 0.9682 | 0.9471 | 226 | 1.0 / 9 |

The publication label alone predicts activity better than the feature-based classifier does under either split.

## Feature-arm comparison (gate_cotested only)

| split | mechgate AUC | baseline (clogp, formal_charge, mw) AUC | delta mechgate − baseline [95% CI] |
|---|---|---|---|
| random | 0.7213 | 0.743 | −0.022 [−0.046, 0.0026] (last repeat) |
| grouped | 0.7107 | 0.7425 | −0.0318 [−0.0576, −0.0062] |

Shift in advantage, random → grouped: −0.0098.

## Structure diagnostics

| run | NN Tanimoto same / cross class | scaffold Jaccard pos vs neg | test scaffolds seen in train: random / grouped |
|---|---|---|---|
| naive_full | 0.8161 / 0.3255 | 0.0099 | 0.5491 / 0.1141 |
| cotested_full | 0.7658 / 0.3193 | 0.0243 | 0.5321 / 0.101 |
| gate_cotested | 0.56 / 0.3046 | 0.0165 | 0.3155 / 0.0771 |

Thresholds used in all runs: group AUC 0.7, purity 0.8, split drop 0.05, NN gap 0.2, scaffold Jaccard 0.1.

Note: absolute paths in the run metadata (`run.input`, `run.out`, interpreter path) were rewritten to repository-relative paths before publishing; no numbers were changed.
