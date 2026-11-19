import hashlib

import pytest

from contamcheck import tabular
from contamcheck.web import audit, example, load_csv, profile


def run_sample(kind):
    sample = example(kind)
    frame = load_csv(sample["csv"])
    chosen = [name for name in profile(frame)["numeric"] if name not in {sample["label"], sample["group"]}]
    return audit(
        frame, sample["label"], sample["group"], chosen, 5,
        hashlib.sha256(sample["csv"].encode("utf-8")).hexdigest(),
    )


def test_source_shortcut_does_not_transfer_to_held_out_labs():
    result = run_sample("shortcut")
    scores = result["comparison"]
    assert scores["random_auc"] > 0.96
    assert scores["grouped_auc"] < 0.75
    assert scores["random_max_overlapping_groups"] > 0
    assert scores["grouped_max_overlapping_groups"] == 0
    assert result["dataset"]["groups"] == 20
    assert result["method"]["folds"] == 5


def test_transferable_signal_preserves_promising_grouped_performance():
    result = run_sample("transferable")
    scores = result["comparison"]
    assert scores["random_auc"] > 0.85
    assert scores["grouped_auc"] > 0.85
    assert abs(scores["random_minus_grouped_auc"]) < 0.07


def test_csv_correctness_and_explicit_feature_validation():
    with pytest.raises(tabular.InputError, match="unique"):
        load_csv("group,label,group\nL1,no,0")

    with pytest.raises(tabular.InputError, match="wrong number"):
        load_csv("group,label,measurement\nL1,no,1,34")

    sample = example("transferable")
    frame = load_csv(sample["csv"])
    with pytest.raises(tabular.InputError, match="exclude"):
        audit(frame, "outcome", "source", ["outcome"], 5, "example")

    frame.at[0, "signal"] = ""
    result = audit(frame, "outcome", "source", ["signal"], 5, "example")
    assert result["dataset"]["missing_feature_cells"] == 1
    assert result["comparison"]["grouped_max_overlapping_groups"] == 0


def test_grouped_training_fold_with_one_class_returns_explanation():
    frame = load_csv("group,label,measurement\nL1,no,1\nL1,no,2\nL2,yes,3\nL2,yes,4")
    with pytest.raises(tabular.InputError, match="only one class"):
        audit(frame, "label", "group", ["measurement"], 2, "example")