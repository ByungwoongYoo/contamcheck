# Input specification and interpretation

## Input table
| flag | required | meaning |
|---|---|---|
| `--input` | yes | `.csv`, `.tsv` / `.tab` / `.txt` (tab-separated), `.parquet` / `.pq`, or `.gz` (delimiter sniffed). Parquet needs a pandas Parquet engine installed. |
| `--label-col` | yes | Binary label: exactly two distinct non-null values. `0/1` keeps its orientation; otherwise values are sorted as strings and the second becomes 1. Nulls are an error. |
| `--group-col` | yes | Provenance column: publication, campaign, batch, instrument, lab, site… Read as string. |
| `--feature-cols` | no | Comma-separated numeric features for the baseline classifier. Default: every numeric column except label/group/SMILES. NaNs are median-imputed. |
| `--smiles-col` | no | Molecular structures (SMILES); enables structure diagnostics if RDKit imports. |
| `--derive-descriptors` | no | Use 10 RDKit 2-D descriptors from `--smiles-col` as features instead of table columns. |
| `--compare-feature-cols`, `--arm-names` | no | Second feature set; compares two feature arms under both CV schemes. |
| `--out` | yes | Output directory: `report.json`, `report.md`, `group_sizes.csv`. |

Other knobs: `--folds` (default 5), `--repeats` (10), `--bootstrap` (2000), `--seed` (20261101) with `--cv-seed`/`--bootstrap-seed`/`--nn-seed` overrides, `--nn-sample` (800), `--max-rows`, and every verdict threshold (`--thresh-*`).

Materials and physical-property tables work the same way: put the composition/structure descriptors in feature columns and the source paper, campaign or instrument in `--group-col`. The SMILES-based structure section (d) is molecule-specific and is simply skipped without `--smiles-col`.

## Diagnostics
- **(a) Group-identity AUC** — each row is scored by the mean label of the *other* rows in its group (leave-one-out target encoding; singleton groups get the global prior). No features are used. 0.5 = group carries no label information; near 1 = the group alone gives the label away. Stratified bootstrap 95% CI.
- **(b) Group purity and sizes** — fraction of groups (and of rows) that are single-label, plus the group-size distribution (`group_sizes.csv`).
- **(c) Random vs grouped split** — `StandardScaler + LogisticRegression`, out-of-fold AUC under repeated `StratifiedKFold` (random) and `GroupKFold` (no group in both train and test); paired bootstrap of the difference. Negative delta = score fell when groups were held out.
- **(c2) Arm comparison** (optional) — the same two schemes for two feature sets, so you can see whether one arm's advantage survives grouping.
- **(d) Structure** (SMILES only) — mean nearest-neighbour Morgan-fingerprint Tanimoto within vs across class, Murcko scaffold overlap between classes, and fraction of test scaffolds already seen in training under each split.

## Reading the verdict
Flags: group AUC ≥ 0.70; group purity ≥ 80%; grouped-minus-random AUC ≤ −0.05; same-minus-cross NN similarity ≥ 0.20; between-class scaffold Jaccard ≤ 0.10. **0 flags = CLEAN, 1 = WATCH, ≥2 = CONTAMINATED.** On CONTAMINATED, report the grouped-CV number as the headline. The thresholds are defaults chosen by the authors, not established statistics; override them and report what you used.

## Limits (be honest about these)
- Descriptive, not causal: a predictive group may reflect genuine differences between sources (e.g. each study measured a different material family), not only artifacts. The tool cannot tell these apart.
- It only sees the one grouping column you give it; leakage through an unrecorded or different provenance variable is invisible.
- The baseline is a linear model on the given features; a small random-vs-grouped gap does not prove a stronger model would not exploit the groups.
- `GroupKFold` with many singleton groups behaves close to a random split for those rows; group AUC scores singletons with the prior, which dilutes (a).
- Binary labels only; no regression targets. Structure diagnostics are molecule-only.
- The verdict counts flags; flags are correlated (a and b measure similar things), so CONTAMINATED can be reached by one underlying effect.
