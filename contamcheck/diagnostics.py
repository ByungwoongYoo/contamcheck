"""The five diagnostics. Each returns a plain dict that is written to JSON."""
from __future__ import annotations

import collections

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .stats import auc, boot_auc, boot_delta


# ---------------------------------------------------------------- (a) group id alone
def loo_target_encoding(groups, y):
    """Leave-one-out target encoding of the group column.

    Row i gets the mean label of the OTHER rows in its group, so the row's own
    label can never leak into its own score. Singleton groups fall back to the
    global prior, which is the conservative choice (it carries no information).
    """
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=float)
    tot = collections.Counter()
    cnt = collections.Counter()
    for g, lab in zip(groups, y):
        tot[g] += lab
        cnt[g] += 1
    prior = float(y.mean())
    out = np.empty(len(y), dtype=float)
    for i, (g, lab) in enumerate(zip(groups, y)):
        n = cnt[g] - 1
        out[i] = (tot[g] - lab) / n if n > 0 else prior
    return out


def group_identity_auc(groups, y, n_boot: int, seed: int):
    enc = loo_target_encoding(groups, y)
    a, lo, hi = boot_auc(y, enc, n=n_boot, seed=seed)
    n_singleton = sum(1 for v in collections.Counter(groups).values() if v == 1)
    return {
        "auc_group_identity_LOO_target_encoding": round(a, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "n_bootstrap_draws": n_boot,
        "bootstrap_seed": seed,
        "n_rows_scored_with_prior_fallback": int(n_singleton),
        "note": (
            "AUC of a score that uses the group label only, never the features. "
            "0.5 means the group column carries no label information."
        ),
    }


# ---------------------------------------------------------------- (b) purity + sizes
def group_structure(groups, y):
    groups = np.asarray(groups)
    y = np.asarray(y)
    by = collections.defaultdict(list)
    for g, lab in zip(groups, y):
        by[g].append(int(lab))
    sizes = np.asarray([len(v) for v in by.values()], dtype=float)
    pure = [g for g, labs in by.items() if len(set(labs)) == 1]
    mixed = set(by) - set(pure)
    rows_in_mixed = int(sum(1 for g in groups if g in mixed))
    # purity weighted by how many rows sit in a pure group
    rows_in_pure = len(groups) - rows_in_mixed
    return {
        "n_rows": int(len(groups)),
        "n_groups": int(len(by)),
        "frac_groups_label_pure": round(len(pure) / len(by), 4),
        "frac_rows_in_a_label_pure_group": round(rows_in_pure / len(groups), 4),
        "frac_rows_in_a_mixed_label_group": round(rows_in_mixed / len(groups), 4),
        "group_size": {
            "min": int(sizes.min()),
            "p25": float(np.percentile(sizes, 25)),
            "median": float(np.median(sizes)),
            "mean": round(float(sizes.mean()), 4),
            "p75": float(np.percentile(sizes, 75)),
            "p95": float(np.percentile(sizes, 95)),
            "max": int(sizes.max()),
            "n_singleton_groups": int((sizes == 1).sum()),
            "frac_rows_in_largest_group": round(float(sizes.max() / sizes.sum()), 4),
        },
    }


# ---------------------------------------------------------------- (c) random vs grouped
def _clf():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
    )


def oof_random(X, y, folds: int, repeats: int, seed: int):
    """Out-of-fold predictions, `repeats` independent stratified k-fold shuffles.

    Returns (mean AUC over repeats, OOF predictions of the LAST repeat, fold ids
    of the last repeat).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    aucs, p, fold_id = [], None, None
    for rep in range(repeats):
        p = np.zeros(len(y))
        fold_id = np.zeros(len(y), dtype=int)
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed + rep)
        for k, (tr, te) in enumerate(cv.split(X, y)):
            m = _clf().fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
            fold_id[te] = k
        aucs.append(auc(y, p))
    return float(np.mean(aucs)), p, fold_id


def oof_grouped(X, y, groups, folds: int):
    """GroupKFold out-of-fold predictions: no group spans train and test."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    p = np.zeros(len(y))
    fold_id = np.zeros(len(y), dtype=int)
    cv = GroupKFold(folds)
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = _clf().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
        fold_id[te] = k
    return float(auc(y, p)), p, fold_id


def split_comparison(X, y, groups, folds, repeats, n_boot, seed, boot_seed=None):
    boot_seed = seed if boot_seed is None else boot_seed
    ar, pr, fr = oof_random(X, y, folds, repeats, seed)
    ag, pg, fg = oof_grouped(X, y, groups, folds)
    cr, crlo, crhi = boot_auc(y, pr, n=n_boot, seed=boot_seed)
    cg, cglo, cghi = boot_auc(y, pg, n=n_boot, seed=boot_seed + 1)
    d, dlo, dhi, frac_le0 = boot_delta(y, pg, pr, n=n_boot, seed=boot_seed + 2)
    return {
        "classifier": "StandardScaler + LogisticRegression(C=1.0, max_iter=2000)",
        "n_folds": folds,
        "n_repeats_random_split": repeats,
        "cv_seed": seed,
        "bootstrap_seed": boot_seed,
        "random_split": {
            "mean_auc_over_repeats": round(ar, 4),
            "auc_last_repeat": round(cr, 4),
            "ci95_last_repeat": [round(crlo, 4), round(crhi, 4)],
        },
        "group_split": {
            "auc": round(ag, 4),
            "ci95": [round(cglo, 4), round(cghi, 4)],
        },
        "delta_group_minus_random": {
            "point": round(d, 4),
            "ci95": [round(dlo, 4), round(dhi, 4)],
            "frac_bootstrap_draws_le_0": round(frac_le0, 4),
            "note": "negative means performance fell when groups were held out",
        },
        "drop_mean_random_to_group": round(ar - ag, 4),
    }, {"random_pred": pr, "random_fold": fr, "group_pred": pg, "group_fold": fg}


def arm_comparison(X_a, X_b, y, groups, folds, repeats, n_boot, seed, name_a, name_b,
                   boot_seed=None):
    boot_seed = seed if boot_seed is None else boot_seed
    """Paired model-vs-model delta under both CV schemes (optional extra arm)."""
    ar_a, pr_a, _ = oof_random(X_a, y, folds, repeats, seed)
    ar_b, pr_b, _ = oof_random(X_b, y, folds, repeats, seed)
    ag_a, pg_a, _ = oof_grouped(X_a, y, groups, folds)
    ag_b, pg_b, _ = oof_grouped(X_b, y, groups, folds)
    dr, drlo, drhi, _ = boot_delta(y, pr_a, pr_b, n=n_boot, seed=boot_seed + 3)
    dg, dglo, dghi, _ = boot_delta(y, pg_a, pg_b, n=n_boot, seed=boot_seed + 4)
    return {
        "cv_seed": seed,
        "bootstrap_seed": boot_seed,
        "arm_a": name_a,
        "arm_b": name_b,
        "random_split": {
            f"mean_auc_{name_a}": round(ar_a, 4),
            f"mean_auc_{name_b}": round(ar_b, 4),
            "delta_a_minus_b_last_repeat": round(dr, 4),
            "ci95": [round(drlo, 4), round(drhi, 4)],
        },
        "group_split": {
            f"auc_{name_a}": round(ag_a, 4),
            f"auc_{name_b}": round(ag_b, 4),
            "delta_a_minus_b": round(dg, 4),
            "ci95": [round(dglo, 4), round(dghi, 4)],
        },
        "shift_in_advantage_random_to_group": round(dg - dr, 4),
    }


# ---------------------------------------------------------------- (d) structure
def split_scaffold_leakage(scaffolds, fold_id):
    """Mean over folds of: fraction of test-set scaffolds also seen in training."""
    scaffolds = np.asarray(scaffolds, dtype=object)
    fold_id = np.asarray(fold_id)
    fracs = []
    for k in np.unique(fold_id):
        te = scaffolds[fold_id == k]
        tr = set(scaffolds[fold_id != k]) - {""}
        te_set = {s for s in te if s}
        if not te_set:
            continue
        fracs.append(len(te_set & tr) / len(te_set))
    return round(float(np.mean(fracs)), 4) if fracs else None


# ---------------------------------------------------------------- (e) verdict
def verdict(report, thresholds):
    flags = []
    ga = report["group_identity"]["auc_group_identity_LOO_target_encoding"]
    if ga >= thresholds["group_auc"]:
        flags.append(
            f"group identity alone predicts the label at AUC {ga:.4f} "
            f">= threshold {thresholds['group_auc']:.2f}"
        )
    pur = report["group_structure"]["frac_groups_label_pure"]
    if pur >= thresholds["purity"]:
        flags.append(
            f"{pur*100:.1f}% of groups are single-label >= threshold "
            f"{thresholds['purity']*100:.0f}%"
        )
    sc = report.get("split_comparison")
    if sc is not None:
        d = sc["delta_group_minus_random"]["point"]
        if d <= -thresholds["split_drop"]:
            flags.append(
                f"baseline AUC fell by {abs(d):.4f} from random to grouped split "
                f">= threshold {thresholds['split_drop']:.2f}"
            )
    st = report.get("structure")
    if st and st.get("nn_tanimoto_same_class_mean") is not None:
        same = st["nn_tanimoto_same_class_mean"]
        cross = st["nn_tanimoto_cross_class_mean"]
        if (same - cross) >= thresholds["nn_gap"]:
            flags.append(
                f"same-class nearest-neighbour similarity exceeds cross-class by "
                f"{same - cross:.4f} >= threshold {thresholds['nn_gap']:.2f}"
            )
        jac = st.get("scaffold_jaccard_pos_vs_neg")
        if jac is not None and jac <= thresholds["scaffold_jaccard"]:
            flags.append(
                f"between-class Murcko scaffold Jaccard {jac:.4f} <= threshold "
                f"{thresholds['scaffold_jaccard']:.2f}"
            )

    if len(flags) == 0:
        level = "CLEAN"
        text = (
            "No diagnostic crossed its threshold. Choose evaluation splits to "
            "match how the model will be used."
        )
    elif len(flags) == 1:
        level = "WATCH"
        text = (
            "One diagnostic crossed its threshold. Treat random-split numbers with "
            "caution and report grouped-CV numbers alongside them."
        )
    else:
        level = "CONTAMINATED"
        text = (
            "At least two diagnostics crossed their thresholds. Inspect which "
            "flags fired. If the model will be used on new groups, lead with "
            "grouped cross-validation when available."
        )
    return {
        "level": level,
        "reading": text,
        "flags_triggered": flags,
        "n_flags": len(flags),
        "thresholds_used": thresholds,
        "thresholds_are_conventions": (
            "These cut-offs are defaults chosen by us, not established statistics. "
            "Override any of them on the command line and report what you used."
        ),
    }
