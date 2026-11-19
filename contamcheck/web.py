"""Run the local browser audit with python -m contamcheck.web."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from . import __version__, diagnostics, tabular

BIND = "127.0.0.1"
MAX_BODY = 4 * 1024 * 1024
MAX_ROWS = 25_000
MAX_COLUMNS = 120
SEED = 20261101
AUDIT_URL = "/api/audit"
SAMPLE_URL = "/api/example"
PROFILE_URL = "/api/columns"
PAGE = Path(__file__).with_name("web.html")


def load_csv(text):
    if not isinstance(text, str):
        raise tabular.InputError("Choose a UTF-8 CSV file.")
    try:
        reader = csv.reader(io.StringIO(text.lstrip("\ufeff"), newline=""), strict=True)
        header = next(reader, [])
        if not 3 <= len(header) <= MAX_COLUMNS:
            raise tabular.InputError(f"The CSV needs 3 to {MAX_COLUMNS} columns.")
        if any(not name.strip() for name in header) or len(set(header)) != len(header):
            raise tabular.InputError("Use unique, nonempty column headers.")
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise tabular.InputError(f"CSV line {reader.line_num} has the wrong number of cells.")
            rows.append(row)
            if len(rows) > MAX_ROWS:
                raise tabular.InputError(f"The local browser audit supports up to {MAX_ROWS:,} rows.")
    except csv.Error as error:
        raise tabular.InputError(f"The CSV could not be read: {error}") from error
    if not rows:
        raise tabular.InputError("The CSV has no data rows.")
    return pd.DataFrame(rows, columns=header)


def numeric_values(series):
    """Return only finite numeric data. Empty cells are missing values."""
    trimmed = series.str.strip()
    present = trimmed.ne("")
    numeric = pd.to_numeric(trimmed.where(present), errors="coerce")
    values = numeric.to_numpy(dtype=float)
    return values if present.any() and np.isfinite(values[present]).all() else None


def profile(frame):
    return {
        "columns": frame.columns.tolist(),
        "numeric": [name for name in frame if numeric_values(frame[name]) is not None],
        "rows": len(frame),
    }


def example(name):
    """Generate contrasting cases with known provenance. All rows are synthetic."""
    if not isinstance(name, str) or name not in {"shortcut", "transferable"}:
        raise tabular.InputError("Unknown example.")
    rng = np.random.default_rng(SEED)
    count = 20
    group_id = np.repeat(np.arange(count), 20)
    lab = [f"Lab {(j + 1):02d}" for j in group_id]
    if name == "shortcut":
        labels_by_lab = rng.permutation(np.tile([0, 1], count // 2))
        y = labels_by_lab[group_id]
        # Each lab has a different trace in the measured channels.
        # That trace identifies the lab without predicting the label of a new lab.
        traces = rng.normal(0, 0.04, (len(y), count)) + np.eye(count)[group_id]
        columns = [f"signal_{i + 1:02d}" for i in range(count)]
        df = pd.DataFrame(traces.round(4), columns=columns)
    else:
        y = np.tile([0, 1], len(group_id) // 2)
        df = pd.DataFrame({
            "signal": (rng.normal(0, 0.9, len(y)) + 2.2 * y).round(4),
            "noise": rng.normal(0, 1, len(y)).round(4),
        })
    df.insert(0, "outcome", np.where(y, "positive", "negative"))
    df.insert(0, "source", lab)
    return {"csv": df.to_csv(index=False), "label": "outcome", "group": "source"}


def audit(frame, label, group, features, folds, data_sha256):
    if not isinstance(label, str) or not isinstance(group, str) or label == group:
        raise tabular.InputError("Choose separate label and group columns.")
    if label not in frame or group not in frame:
        raise tabular.InputError("Choose label and group columns from this CSV.")
    if frame[label].str.strip().eq("").any():
        raise tabular.InputError("The label column contains blank values.")
    if frame[group].str.strip().eq("").any():
        raise tabular.InputError("The group column contains blank values.")
    y, mapping = tabular.binarise_label(frame[label])
    groups = frame[group].to_numpy()
    if type(folds) is not int or not 2 <= folds <= 10:
        raise tabular.InputError("Enter a whole number of folds between 2 and 10.")
    positives = int(y.sum())
    negatives = len(y) - positives
    if min(positives, negatives) < folds:
        raise tabular.InputError("Each label needs at least as many rows as there are folds.")
    if frame[group].nunique() < folds:
        raise tabular.InputError("Use fewer folds or add more distinct groups.")
    if (
        not isinstance(features, list)
        or not 1 <= len(features) <= MAX_COLUMNS
        or any(not isinstance(name, str) for name in features)
        or len(set(features)) != len(features)
    ):
        raise tabular.InputError("Select at least one numeric feature. Each must be unique.")
    if any(name not in frame or name in {label, group} for name in features):
        raise tabular.InputError("Features must come from the CSV and exclude label and group.")
    vectors = []
    for name in features:
        values = numeric_values(frame[name])
        if values is None:
            raise tabular.InputError(f"Feature {name!r} must have finite numbers or empty cells.")
        vectors.append(values)
    X = np.column_stack(vectors)

    # Check viability before either fit. Together with the intrinsic constraint of
    # GroupKFold, these fold IDs provide a verifiable record of group separation.
    from sklearn.model_selection import GroupKFold

    group_cv = GroupKFold(folds)
    for train, _ in group_cv.split(X, y, groups):
        if np.unique(y[train]).size != 2:
            raise tabular.InputError(
                "A group-held-out training fold has only one class. "
                "Change the folds or choose a grouping with both labels represented across groups."
            )
        if not np.isfinite(X[train]).any():
            raise tabular.InputError("A group-held-out training fold has no observed features.")

    try:
        random_auc, _, random_fold = diagnostics.oof_random(X, y, folds, repeats=1, seed=SEED)
        group_auc, _, grouped_fold = diagnostics.oof_grouped(X, y, groups, folds)
    except ValueError as error:
        raise tabular.InputError(f"The comparison could not be fitted: {error}") from error
    grouped_overlap = max_group_overlap(groups, grouped_fold)
    if grouped_overlap != 0:
        raise RuntimeError("Grouped split integrity check failed.")

    return {
        "version": __version__,
        "dataset": {
            "sha256_utf8": data_sha256,
            "rows": len(frame),
            "groups": int(frame[group].nunique()),
            "label": label,
            "group": group,
            "label_mapping": mapping,
            "features": features,
            "positives": positives,
            "negatives": negatives,
            "missing_feature_cells": int(np.isnan(X).sum()),
        },
        "comparison": {
            "random_auc": round(random_auc, 4),
            "grouped_auc": round(group_auc, 4),
            "random_minus_grouped_auc": round(random_auc - group_auc, 4),
            "random_max_overlapping_groups": max_group_overlap(groups, random_fold),
            "grouped_max_overlapping_groups": grouped_overlap,
        },
        "method": {
            "folds": folds,
            "seed": SEED,
            "baseline": "Fold-fit median imputation, standard scaling and logistic regression",
            "metric": "Pooled out-of-fold binary ROC AUC on the same rows",
            "random_split": "Shuffled StratifiedKFold",
            "grouped_split": "GroupKFold",
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        },
        "interpretation": (
            "A drop suggests performance depends on the chosen grouping or a distribution shift. "
            "The comparison alone cannot identify the cause. A small drop does not establish that "
            "the data are free of leakage. Review features and choose a realistic holdout unit."
        ),
    }


def max_group_overlap(groups, fold_ids):
    return max(
        len(set(groups[fold_ids != fold]) & set(groups[fold_ids == fold]))
        for fold in np.unique(fold_ids)
    )


class LocalHandler(BaseHTTPRequestHandler):
    # The server accepts forms from this origin only. There are no file-path inputs,
    # uploads to third parties, data writes, analytics, or arbitrary file serving.
    def _allowed_host(self):
        return self.headers.get("Host") == f"{BIND}:{self.server.server_port}"

    def log_message(self, format, *args):
        return

    def _send(self, value, status=200, kind="application/json; charset=utf-8"):
        content = (
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if kind.startswith("application/json")
            else value
        )
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            f"default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-{self.server.nonce}'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if not self._allowed_host():
            self._send({"error": "Open the loopback address shown in the terminal."}, 403)
        elif self.path == "/":
            page = PAGE.read_text(encoding="utf-8").replace("__AUDIT_NONCE__", self.server.nonce)
            self._send(page.encode("utf-8"), kind="text/html; charset=utf-8")
        else:
            self._send({"error": "Page not found."}, 404)

    def do_POST(self):
        origin = f"http://{BIND}:{self.server.server_port}"
        if (
            not self._allowed_host()
            or self.headers.get("Origin") != origin
            or self.headers.get("X-Contamcheck-Local") != self.server.nonce
        ):
            self._send({"error": "Refresh the local page and try again."}, 403)
            return
        if self.path not in {SAMPLE_URL, PROFILE_URL, AUDIT_URL}:
            self._send({"error": "Page not found."}, 404)
            return
        if self.headers.get_content_type() != "application/json":
            self._send({"error": "Expected a JSON request."}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            self._send({"error": "Choose a CSV under 4 MB and try again."}, 413)
            return
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise tabular.InputError("Invalid audit request.")
            if self.path == SAMPLE_URL:
                self._send(example(request.get("name")))
                return
            text = request.get("csv")
            frame = load_csv(text)
            if self.path == PROFILE_URL:
                self._send(profile(frame))
                return
            self._send(audit(
                frame, request.get("label"), request.get("group"), request.get("features"),
                request.get("folds"), hashlib.sha256(text.encode("utf-8")).hexdigest(),
            ))
        except (UnicodeError, json.JSONDecodeError):
            self._send({"error": "The request was not valid UTF-8 JSON."}, 400)
        except tabular.InputError as error:
            self._send({"error": str(error)}, 422)


def main():
    parser = argparse.ArgumentParser(description="Start the loopback-only Contamcheck browser audit.")
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765, 0 for automatic)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("choose a port between 0 and 65535")
    try:
        server = ThreadingHTTPServer((BIND, args.port), LocalHandler)
    except OSError as error:
        parser.exit(1, f"Could not start the local audit: {error}\n")
    server.nonce = secrets.token_urlsafe(24)
    print(f"Contamcheck is running at http://{BIND}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()