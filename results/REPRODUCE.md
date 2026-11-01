# Reproducing `results/` (worked example)

The worked example uses three tables derived from a public bioactivity dataset (molecules labelled active/inactive, grouped by source publication `document`). It is the case that motivated the tool, not its scope.

**The input tables are not redistributed in this repository** (third-party data with its own licence terms); `results/tables/tables_manifest.json` records their row counts (naive_full 11996, cotested_full 4029, gate_cotested 586). `repro/prepare_tables.py` documents how they were derived from the upstream project's artefacts. The reports in `results/<run>/` are the unmodified outputs.

## Environment (from `results/*/report.json` → `run`)
contamcheck 0.1.0 · Python 3.9.25 · numpy 1.23.5 · pandas 2.3.3 · scikit-learn 1.6.1 · rdkit 2022.09.5 · Linux x86_64. Master seed 20261101 in all runs.

## Commands (contamcheck arguments verbatim from `report.json` → `run.argv`)
```bash
contamcheck --input results/tables/naive_full.tsv --label-col label --group-col document --smiles-col smiles --derive-descriptors --out results/naive_full --bootstrap 2000 --seed 20261101 --bootstrap-seed 1 --nn-seed 7 --nn-sample 800

contamcheck --input results/tables/cotested_full.tsv --label-col label --group-col document --smiles-col smiles --derive-descriptors --out results/cotested_full --bootstrap 2000 --seed 20261101 --bootstrap-seed 1 --nn-seed 7 --nn-sample 800

contamcheck --input results/tables/gate_cotested.tsv --label-col label --group-col document --smiles-col smiles --out results/gate_cotested --feature-cols ompf_percentile,multiporin_percentile,lipid_percentile,efflux_percentile,activity_percentile --compare-feature-cols clogp,formal_charge,mw --arm-names mechgate,baseline --cv-seed 0 --bootstrap-seed 3 --nn-seed 7 --bootstrap 2000 --seed 20261101
```

| run | cv_seed | bootstrap_seed | nn_seed | wall s (report) | max RSS kB (`logs/*.time`) |
|---|---|---|---|---|---|
| naive_full | 20261101 | 1 | 7 | 58.94 | 817160 |
| cotested_full | 20261101 | 1 | 7 | 19.46 | 362884 |
| gate_cotested | 0 | 3 | 7 | 5.5 | 200016 |
