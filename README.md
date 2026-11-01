# contamcheck

**Is your cross-validation score measuring the property, or the provenance?**

`contamcheck` tests whether a labelled tabular dataset carries a *provenance shortcut*: a grouping column — source publication, measurement campaign, production batch, instrument, lab, site — that predicts the label on its own. When that happens, a random train/test split puts rows from the same group on both sides, the model learns "which campaign is this?" instead of the physics or chemistry you care about, and the random-split score is inflated.

This is common in compiled datasets: materials property tables merged from many papers, manufacturing data spread over batches and lines, measurement data from several instruments or labs, and bioactivity data compiled from many publications. `contamcheck` gives you a quick, reproducible, seeded answer before you report a number.

It reports:

| | diagnostic | what it tells you |
|---|---|---|
| (a) | **group-alone AUC** | how well the group column predicts the label with *no features at all* (leave-one-out target encoding, bootstrap 95% CI) |
| (b) | **group label purity** and group-size distribution | what fraction of groups contain only one label |
| (c) | **random-split vs group-held-out CV** | the same baseline classifier under `StratifiedKFold` and `GroupKFold`, with a paired bootstrap of the difference |
| (c2) | optional **feature-arm comparison** | does one feature set's advantage survive grouped CV? |
| (d) | optional **structure diagnostics** (molecules only) | nearest-neighbour Tanimoto within vs across class, Murcko scaffold overlap — needs a SMILES column and RDKit |
| (e) | **verdict** | `CLEAN` / `WATCH` / `CONTAMINATED` from counted, overridable thresholds |

## Install

```bash
git clone https://github.com/ByungwoongYoo/contamcheck.git contamcheck
cd contamcheck
pip install -e .            # numpy, pandas, scikit-learn
pip install -e ".[chem]"    # optional: RDKit, for the molecule structure diagnostics
```

Python ≥ 3.9. No network access, no telemetry; everything runs locally on one core.

## Quickstart

```bash
pip install -e .
contamcheck --input examples/contaminated_measurement.csv --label-col corrosion_pass --group-col campaign --out out/demo
cat out/demo/report.md
```

`python -m contamcheck ...` works too. Each run writes `report.json` (every number, plus seeds and library versions), `report.md` (human-readable), and `group_sizes.csv`.

On your own data: `--label-col` is a binary label, `--group-col` is the provenance column, and by default every other numeric column is used as a feature (or pass `--feature-cols a,b,c`). See [`INPUT_SPEC.md`](INPUT_SPEC.md) for all flags.

## Example output

Console output from the quickstart above (copied from [`examples/expected/contaminated_measurement.log`](examples/expected/contaminated_measurement.log)):

```
[contamcheck 0.1.0] seed=20261101 rows=1490 groups=48 -> examples/expected/contaminated_measurement
  (a) group-identity AUC 0.9415
  (b) 48 groups, 0.1667 label-pure
  (c) random 0.6165 vs grouped 0.4577
  (e) VERDICT CONTAMINATED (2 flags) seed=20261101 [6.12s]
```

The campaign alone predicts corrosion pass/fail at AUC 0.9415 [95% CI 0.9258, 0.9547]. A baseline classifier on the composition and process features scores 0.6165 under random 5-fold CV but 0.4577 when whole campaigns are held out (paired-bootstrap delta −0.1591 [−0.1771, −0.1414]): the random-split score came from recognising each campaign's instrument offset, not from a transferable signal.

All three shipped examples (synthetic, seeded — see [`examples/`](examples/)):

| example | domain | group column | rows / groups | group-alone AUC | groups single-label | random-split AUC | grouped AUC | verdict |
|---|---|---|---|---|---|---|---|---|
| `clean_manufacturing.csv` | injection moulding, `defect` | `batch` | 1200 / 60 | 0.4558 | 0.0 | 0.7932 | 0.7898 | **CLEAN** (0 flags) |
| `watch_materials.csv` | thin films, `conductive` | `source_paper` | 1186 / 70 | 0.8077 | 0.0714 | 0.9847 | 0.9852 | **WATCH** (1 flag) |
| `contaminated_measurement.csv` | alloy corrosion tests, `corrosion_pass` | `campaign` | 1490 / 48 | 0.9415 | 0.1667 | 0.6165 | 0.4577 | **CONTAMINATED** (2 flags) |

Numbers are from `examples/expected/*/report.json` (Python 3.13.5, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, seed 20261101). Random-split AUC is the mean of 10 repeats.

### Worked example on real data: compiled bioactivity data

The case that motivated the tool: a public bioactivity dataset (molecules labelled active/inactive), grouped by source publication. Full details in [`results/RESULTS.md`](results/RESULTS.md); numbers from `results/*/report.json`.

| run | rows | groups | group-alone AUC [95% CI] | groups single-label | random-split AUC | grouped AUC | verdict |
|---|---|---|---|---|---|---|---|
| naive_full | 11996 | 1418 | 0.9929 [0.9909, 0.9949] | 0.9711 | 0.7717 | 0.7424 | CONTAMINATED (4) |
| cotested_full | 4029 | 905 | 0.9468 [0.9345, 0.9581] | 0.9337 | 0.7402 | 0.7288 | CONTAMINATED (4) |
| gate_cotested | 586 | 346 | 0.8858 [0.8627, 0.9086] | 0.9682 | 0.7213 | 0.7107 | CONTAMINATED (4) |

The publication label alone predicts activity better than the feature-based classifier under either split. Note the random-vs-grouped gap is small here (none crossed −0.05) — the linear baseline was not exploiting the groups much — yet the group-alone diagnostic shows the shortcut is available to any model flexible enough to find it.

## Reading the verdict

Five flags, each with an overridable threshold (`--thresh-*`):

| flag | default | |
|---|---|---|
| group-alone AUC ≥ | 0.70 | the group column gives the label away |
| groups single-label ≥ | 80 % | most groups contain only one class |
| grouped − random AUC ≤ | −0.05 | the model loses ≥ 0.05 AUC when groups are held out |
| same − cross-class NN Tanimoto ≥ | 0.20 | (SMILES only) classes sit in near-disjoint chemical space |
| between-class scaffold Jaccard ≤ | 0.10 | (SMILES only) classes share almost no scaffolds |

- **CLEAN** (0 flags): no visible shortcut through this column. Grouped CV is still good practice.
- **WATCH** (1 flag): report grouped-CV numbers alongside random-split ones and look at which flag fired.
- **CONTAMINATED** (≥ 2 flags): random-split performance is not trustworthy on this table. Use grouped CV and make the grouped number your headline.

The thresholds are defaults chosen by the author, not established statistics. If you change them, report what you used. Group-alone AUC slightly *below* 0.5 (as in the clean example) is expected when groups carry no label information: leave-one-out encoding is mildly anti-correlated with each row's own label.

## Limitations (please read)

- **Technical diagnostic, not a causal one.** A predictive group can reflect a genuine difference between sources (each paper studied a different material family; each campaign tested a different alloy grade) just as much as an artefact (calibration offsets, selective reporting, curation choices). `contamcheck` cannot tell these apart. It tells you the shortcut *exists*; deciding whether it is legitimate needs domain knowledge.
- **Binary labels only.** No regression targets or multi-class labels yet.
- **One group column at a time.** Leakage through a provenance variable you did not pass (or did not record) is invisible. Run it once per candidate column.
- **Structure diagnostics need SMILES** (and RDKit). For materials, manufacturing and measurement tables, section (d) is simply skipped; put your composition/process descriptors in feature columns.
- **Linear baseline.** The split comparison uses `StandardScaler + LogisticRegression`. A small random-vs-grouped gap does not prove a stronger model would not exploit the groups — that is what the group-alone diagnostic is for.
- **Many singleton groups** behave like a random split under `GroupKFold`, and singletons are scored with the global prior in (a), which dilutes it.
- **Few, large groups.** With only a handful of groups the leave-one-out encoding in (a) scores each row slightly *against* its own label, so the group-alone AUC can land far below 0.5. Read that as "no evidence either way", not as proof the column is safe; with fewer groups than folds the grouped CV in (c) cannot run at all. The shipped `clean_manufacturing` example shows a mild case (group-alone AUC 0.4558 with 20 rows per batch).
- **Correlated flags.** Group-alone AUC and purity measure related things, so `CONTAMINATED` can be reached through one underlying effect.

## Reproducibility

Every run is seeded (`--seed`, default 20261101; per-diagnostic overrides `--cv-seed`, `--bootstrap-seed`, `--nn-seed`) and records library versions in `report.json`. Exact numbers can differ slightly across numpy / scikit-learn versions.

```bash
pip install -e ".[test]"
pytest -q                          # planted-artefact detection, clean control, AUC vs scikit-learn
bash examples/run_examples.sh out/ # re-run the three examples
```

## Citation

If `contamcheck` helps your work, please cite it:

```bibtex
@software{yoo_contamcheck_2026,
  author  = {Yoo, Byungwoong},
  title   = {contamcheck: diagnostics for provenance shortcuts in labelled tabular datasets},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/ByungwoongYoo/contamcheck},
  license = {MIT}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

MIT © 2026 Byungwoong Yoo. See [`LICENSE`](LICENSE).
