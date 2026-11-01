"""The shipped synthetic examples must reproduce their documented verdicts."""
import json
import os
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [
    ("clean_manufacturing", "defect", "batch"),
    ("watch_materials", "conductive", "source_paper"),
    ("contaminated_measurement", "corrosion_pass", "campaign"),
]


@pytest.mark.parametrize("name,label,group", CASES)
def test_example_verdict_matches_expected(name, label, group):
    expected = json.load(open(os.path.join(ROOT, "examples", "expected", name, "report.json")))
    with tempfile.TemporaryDirectory() as out:
        rc = subprocess.run(
            [sys.executable, "-m", "contamcheck",
             "--input", os.path.join(ROOT, "examples", f"{name}.csv"),
             "--label-col", label, "--group-col", group, "--out", out,
             "--bootstrap", "300"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert rc.returncode == 0, rc.stderr
        got = json.load(open(os.path.join(out, "report.json")))
    assert got["verdict"]["level"] == expected["verdict"]["level"]
    assert got["verdict"]["n_flags"] == expected["verdict"]["n_flags"]
    # point estimates do not depend on bootstrap draws
    assert got["group_identity"]["auc_group_identity_LOO_target_encoding"] == \
        expected["group_identity"]["auc_group_identity_LOO_target_encoding"]
