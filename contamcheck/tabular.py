"""Table loading and feature selection."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


class InputError(Exception):
    pass


def load_table(path: str) -> pd.DataFrame:
    """Load CSV / TSV / Parquet by extension. No network, no inference magic."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".csv.gz"):
        df = pd.read_csv(path)
    elif ext in (".tsv", ".tab", ".txt"):
        df = pd.read_csv(path, sep="\t")
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext == ".gz":
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        raise InputError(
            f"unsupported extension {ext!r}; use .csv, .tsv, .txt or .parquet"
        )
    if len(df) == 0:
        raise InputError("input table has zero rows")
    return df


def binarise_label(series: pd.Series):
    """Map a two-valued label column to {0,1}. Returns (y, mapping)."""
    vals = pd.unique(series.dropna())
    if len(vals) != 2:
        raise InputError(
            f"--label-col must have exactly 2 distinct non-null values, found {len(vals)}: {list(vals)[:6]}"
        )
    sv = sorted(vals, key=lambda v: str(v))
    # keep natural 0/1 orientation when the column already looks numeric binary
    try:
        nums = sorted(float(v) for v in vals)
        if nums == [0.0, 1.0]:
            sv = [v for v in sorted(vals, key=lambda v: float(v))]
    except (TypeError, ValueError):
        pass
    mapping = {sv[0]: 0, sv[1]: 1}
    y = series.map(mapping).to_numpy()
    if np.isnan(np.asarray(y, dtype=float)).any():
        raise InputError("--label-col contains nulls; drop or impute them first")
    return np.asarray(y, dtype=int), {str(k): int(v) for k, v in mapping.items()}


def pick_feature_columns(df, label_col, group_col, smiles_col, explicit):
    """Choose the numeric columns the baseline classifier is allowed to use."""
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise InputError(f"feature columns not in table: {missing}")
        return list(explicit), "explicit"
    reserved = {label_col, group_col}
    if smiles_col:
        reserved.add(smiles_col)
    cols = [
        c
        for c in df.columns
        if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
    ]
    if cols:
        return cols, "auto_numeric"
    return [], "none"


def feature_matrix(df, cols):
    X = df[cols].to_numpy(dtype=float)
    # median-impute so a stray NaN does not silently kill a whole run
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(med, idx[1])
    return X
