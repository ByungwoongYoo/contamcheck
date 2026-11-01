#!/usr/bin/env bash
# Run contamcheck on the three synthetic examples and write to examples/expected/.
# Usage (from the repository root):  bash examples/run_examples.sh [out_root]
set -euo pipefail
OUT=${1:-examples/expected}
mkdir -p "$OUT"
contamcheck --input examples/clean_manufacturing.csv      --label-col defect         --group-col batch        --out "$OUT/clean_manufacturing"      | tee "$OUT/clean_manufacturing.log"
contamcheck --input examples/watch_materials.csv          --label-col conductive     --group-col source_paper --out "$OUT/watch_materials"          | tee "$OUT/watch_materials.log"
contamcheck --input examples/contaminated_measurement.csv --label-col corrosion_pass --group-col campaign     --out "$OUT/contaminated_measurement" | tee "$OUT/contaminated_measurement.log"
