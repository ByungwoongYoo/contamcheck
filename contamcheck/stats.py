"""AUC and bootstrap helpers. Pure numpy, no third-party statistics code."""
from __future__ import annotations

import numpy as np


def auc(y, s) -> float:
    """Rank-based ROC AUC with proper mid-rank handling of ties."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sv = s[order]
    r = np.empty(len(sv), dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _strat_idx(y, rng):
    """One stratified (class-preserving) bootstrap resample of row indices."""
    i1 = np.flatnonzero(y == 1)
    i0 = np.flatnonzero(y == 0)
    return np.concatenate(
        [rng.choice(i1, len(i1), replace=True), rng.choice(i0, len(i0), replace=True)]
    )


def boot_auc(y, s, n: int = 2000, seed: int = 0):
    """Point AUC plus a stratified bootstrap 95% percentile interval."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        b = _strat_idx(y, rng)
        v = auc(y[b], s[b])
        if v == v:
            vals.append(v)
    vals = np.asarray(vals)
    return (
        float(auc(y, s)),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def boot_delta(y, sa, sb, n: int = 2000, seed: int = 0):
    """Paired stratified bootstrap of AUC(sa) - AUC(sb).

    Returns (delta, lo95, hi95, fraction_of_draws_with_delta_le_0).
    """
    y = np.asarray(y)
    sa = np.asarray(sa, dtype=float)
    sb = np.asarray(sb, dtype=float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        b = _strat_idx(y, rng)
        va, vb = auc(y[b], sa[b]), auc(y[b], sb[b])
        if va == va and vb == vb:
            d.append(va - vb)
    d = np.asarray(d)
    return (
        float(auc(y, sa) - auc(y, sb)),
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
        float((d <= 0).mean()),
    )


def boot_delta_unpaired(y_a, sa, y_b, sb, n: int = 2000, seed: int = 0):
    """Bootstrap of AUC(sa|y_a) - AUC(sb|y_b) where the two score vectors come
    from different CV schemes over the SAME rows (so rows are still paired, but
    the label vector is identical). Kept separate for clarity of intent."""
    return boot_delta(y_a, sa, sb, n=n, seed=seed)
