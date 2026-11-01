# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [0.1.0] — 2026-11-01

First public release.

### Added
- Diagnostics: group-alone AUC (leave-one-out target encoding, stratified bootstrap CI); group label purity and size distribution; random-split vs `GroupKFold` CV with paired bootstrap delta; optional feature-arm comparison; optional RDKit structure diagnostics (nearest-neighbour Tanimoto, Murcko scaffold overlap and split leakage).
- CLEAN / WATCH / CONTAMINATED verdict with overridable thresholds.
- Outputs: `report.json`, `report.md`, `group_sizes.csv`; JSON is rewritten after each section so partial progress survives interruption.
- `contamcheck` console command (in addition to `python -m contamcheck`) and `--version`.
- `pyproject.toml` packaging; optional extras `chem` (RDKit), `parquet`, `test`.
- Three seeded synthetic examples (`examples/`) with expected outputs: clean manufacturing, watch-level materials, contaminated measurement campaigns.
- Worked example on compiled public bioactivity data (`results/`).
- Tests: planted-artefact detection, clean control, AUC vs scikit-learn, shipped-example verdicts.
