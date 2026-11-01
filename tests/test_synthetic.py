"""Synthetic end-to-end tests: a planted group-label artifact must be detected,
a clean control with the same shape must not be.

Run:  python -m pytest tests/ -q      (or: python tests/test_synthetic.py)
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SEED = 20261101
N_GROUPS = 120
PER_GROUP = 12


D = 150  # feature width: wide enough that a linear model can memorise a
         # per-group signature, which is exactly how real leakage behaves


def _gids():
    return np.repeat([f"G{i:04d}" for i in range(N_GROUPS)], PER_GROUP)


def make_contaminated(seed=SEED):
    """Planted artifact: every group is single-label, and the features carry a
    per-group signature and nothing else.

    A model trained on a random split sees rows from every group and can learn
    "this signature -> this label"; under a group split the test groups'
    signatures were never seen, so performance must collapse. Group identity
    alone is perfectly predictive.
    """
    rng = np.random.default_rng(seed)
    gid = _gids()
    group_label = np.tile([0, 1], N_GROUPS // 2)
    y = np.repeat(group_label, PER_GROUP)
    signature = rng.normal(0, 1.0, (N_GROUPS, D))     # group -> signature
    X = np.repeat(signature, PER_GROUP, axis=0) + rng.normal(0, 0.25, (len(y), D))
    df = pd.DataFrame(X, columns=[f"f{i:03d}" for i in range(D)])
    df.insert(0, "label", y)
    df.insert(0, "gid", gid)
    return df


def make_clean(seed=SEED + 1):
    """Clean control of the same shape: the same number of groups of the same
    size and the same feature width, but group membership is independent of the
    label and the signal lives in a feature direction that generalises across
    groups. Both splits should score the same."""
    rng = np.random.default_rng(seed)
    gid = _gids()
    n = len(gid)
    y = rng.integers(0, 2, n)
    w = rng.normal(0, 1.0, D)
    w /= np.linalg.norm(w)
    X = rng.normal(0, 1.0, (n, D)) + 0.9 * np.outer(y - 0.5, w) * 2.0
    # a group effect that is real but label-independent, so groups matter to the
    # data without encoding the answer
    X += np.repeat(rng.normal(0, 0.5, (N_GROUPS, D)), PER_GROUP, axis=0)
    df = pd.DataFrame(X, columns=[f"f{i:03d}" for i in range(D)])
    df.insert(0, "label", y)
    df.insert(0, "gid", gid)
    return df


def _run(df, out):
    path = os.path.join(out, "table.csv")
    df.to_csv(path, index=False)
    rc = subprocess.run(
        [sys.executable, "-m", "contamcheck", "--input", path,
         "--label-col", "label", "--group-col", "gid",
         "--out", os.path.join(out, "report"), "--bootstrap", "300",
         "--repeats", "3", "--seed", str(SEED)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    return json.load(open(os.path.join(out, "report", "report.json")))


def test_detects_planted_group_artifact():
    with tempfile.TemporaryDirectory() as out:
        r = _run(make_contaminated(), out)
    ga = r["group_identity"]["auc_group_identity_LOO_target_encoding"]
    assert ga > 0.95, f"group identity should be highly predictive, got {ga}"
    assert r["group_structure"]["frac_groups_label_pure"] == 1.0
    assert r["split_comparison"]["delta_group_minus_random"]["point"] < -0.05, \
        "performance must fall when groups are held out"
    assert r["verdict"]["level"] == "CONTAMINATED", r["verdict"]
    assert r["run"]["seed"] == SEED


def test_clean_control_is_not_flagged():
    with tempfile.TemporaryDirectory() as out:
        r = _run(make_clean(), out)
    ga = r["group_identity"]["auc_group_identity_LOO_target_encoding"]
    assert ga < 0.60, f"group identity should be uninformative, got {ga}"
    assert r["group_structure"]["frac_groups_label_pure"] < 0.20
    assert abs(r["split_comparison"]["delta_group_minus_random"]["point"]) < 0.05
    assert r["verdict"]["level"] == "CLEAN", r["verdict"]
    assert r["verdict"]["n_flags"] == 0


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    from contamcheck.stats import auc
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.integers(0, 2, 200)
        s = rng.normal(size=200).round(1)  # forces ties
        assert abs(auc(y, s) - roc_auc_score(y, s)) < 1e-12


if __name__ == "__main__":
    test_auc_matches_sklearn()
    print("auc vs sklearn: ok")
    test_detects_planted_group_artifact()
    print("planted artifact detected: ok")
    test_clean_control_is_not_flagged()
    print("clean control not flagged: ok")
    print("ALL TESTS PASSED")
