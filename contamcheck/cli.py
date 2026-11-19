"""contamcheck command line entry point."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

from . import __version__, chem, diagnostics, tabular
from .report import render_markdown

DEFAULT_SEED = 20261101


def build_parser():
    p = argparse.ArgumentParser(
        prog="contamcheck",
        description=(
            "Test a labelled table for group-provenance contamination: is the "
            "label recoverable from which source/batch/document a row came from, "
            "rather than from the property you mean to model?"
        ),
    )
    p.add_argument("--version", action="version",
                   version=f"contamcheck {__version__}")
    p.add_argument("--input", required=True, help="CSV, TSV or Parquet table")
    p.add_argument("--label-col", required=True, help="binary label column")
    p.add_argument("--group-col", required=True,
                   help="provenance column: document, batch, assay, lab, site...")
    p.add_argument("--smiles-col", default=None,
                   help="optional SMILES column; enables the structure diagnostics")
    p.add_argument("--out", required=True, help="output directory (created)")
    p.add_argument("--feature-cols", default=None,
                   help="comma-separated numeric feature columns for the baseline "
                        "classifier; default is every numeric column that is not the "
                        "label, group or SMILES column")
    p.add_argument("--derive-descriptors", action="store_true",
                   help="ignore the table's numeric columns and use 2-D descriptors "
                        "computed from --smiles-col instead (requires rdkit)")
    p.add_argument("--compare-feature-cols", default=None,
                   help="optional second feature set; enables an arm-vs-arm "
                        "comparison under both CV schemes")
    p.add_argument("--arm-names", default="arm_a,arm_b",
                   help="names for the two arms of --compare-feature-cols")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"master random seed (default {DEFAULT_SEED})")
    p.add_argument("--cv-seed", type=int, default=None, help="override CV seed")
    p.add_argument("--bootstrap-seed", type=int, default=None,
                   help="override bootstrap seed")
    p.add_argument("--nn-seed", type=int, default=None,
                   help="override nearest-neighbour sampling seed")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap draws")
    p.add_argument("--folds", type=int, default=5, help="CV folds")
    p.add_argument("--repeats", type=int, default=10,
                   help="repeats of the random-split CV")
    p.add_argument("--nn-sample", type=int, default=800,
                   help="molecules sampled as nearest-neighbour queries")
    p.add_argument("--max-rows", type=int, default=0,
                   help="if >0, subsample this many rows (seeded) before anything")
    # verdict thresholds, all overridable
    p.add_argument("--thresh-group-auc", type=float, default=0.70)
    p.add_argument("--thresh-purity", type=float, default=0.80)
    p.add_argument("--thresh-split-drop", type=float, default=0.05)
    p.add_argument("--thresh-nn-gap", type=float, default=0.20)
    p.add_argument("--thresh-scaffold-jaccard", type=float, default=0.10)
    return p


def _write(out_dir, report):
    """Write JSON after every section so partial progress survives a kill."""
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    seed = args.seed
    cv_seed = args.cv_seed if args.cv_seed is not None else seed
    boot_seed = args.bootstrap_seed if args.bootstrap_seed is not None else seed
    nn_seed = args.nn_seed if args.nn_seed is not None else seed

    os.makedirs(args.out, exist_ok=True)
    df = tabular.load_table(args.input)
    for col, flag in ((args.label_col, "--label-col"), (args.group_col, "--group-col")):
        if col not in df.columns:
            raise tabular.InputError(f"{flag} {col!r} not in table; columns: {list(df.columns)}")
    if args.smiles_col and args.smiles_col not in df.columns:
        raise tabular.InputError(f"--smiles-col {args.smiles_col!r} not in table")

    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=seed).reset_index(drop=True)
    df = df.reset_index(drop=True)

    y, mapping = tabular.binarise_label(df[args.label_col])
    groups = df[args.group_col].astype(str).fillna("__NA__").to_numpy()

    explicit = [c.strip() for c in args.feature_cols.split(",")] if args.feature_cols else None
    if args.derive_descriptors:
        if not args.smiles_col:
            raise tabular.InputError("--derive-descriptors needs --smiles-col")
        if explicit:
            raise tabular.InputError("--derive-descriptors conflicts with --feature-cols")
        feat_cols, feat_source = [], "none"
    else:
        feat_cols, feat_source = tabular.pick_feature_columns(
            df, args.label_col, args.group_col, args.smiles_col, explicit
        )

    report = {
        "run": {
            "contamcheck_version": __version__,
            "input": os.path.abspath(args.input),
            "out": os.path.abspath(args.out),
            "label_col": args.label_col,
            "group_col": args.group_col,
            "smiles_col": args.smiles_col,
            "label_mapping": mapping,
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "seed": seed,
            "cv_seed": cv_seed,
            "bootstrap_seed": boot_seed,
            "nn_seed": nn_seed,
            "n_bootstrap": args.bootstrap,
            "folds": args.folds,
            "repeats": args.repeats,
            "feature_cols": feat_cols,
            "feature_source": feat_source,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit_available": chem.HAVE_RDKIT,
            "argv": sys.argv[1:] if argv is None else list(argv),
        }
    }
    import sklearn
    report["run"]["scikit_learn"] = sklearn.__version__
    if chem.HAVE_RDKIT:
        import rdkit
        report["run"]["rdkit"] = rdkit.__version__
    print(f"[contamcheck {__version__}] seed={seed} rows={len(df)} "
          f"groups={len(set(groups))} -> {args.out}", flush=True)
    _write(args.out, report)

    # (a)
    report["group_identity"] = diagnostics.group_identity_auc(
        groups, y, n_boot=args.bootstrap, seed=boot_seed
    )
    _write(args.out, report)
    print(f"  (a) group-identity AUC "
          f"{report['group_identity']['auc_group_identity_LOO_target_encoding']:.4f}", flush=True)

    # (b)
    report["group_structure"] = diagnostics.group_structure(groups, y)
    _write(args.out, report)
    sizes = pd.Series(groups).value_counts()
    pd.DataFrame({"group": sizes.index, "n_rows": sizes.to_numpy()}).to_csv(
        os.path.join(args.out, "group_sizes.csv"), index=False
    )
    print(f"  (b) {report['group_structure']['n_groups']} groups, "
          f"{report['group_structure']['frac_groups_label_pure']:.4f} label-pure", flush=True)

    # structure parsing (shared by (c) fallback features and (d))
    mols = ok_idx = None
    if args.smiles_col and chem.HAVE_RDKIT:
        mols, ok_idx = chem.parse(df[args.smiles_col].tolist())
    elif args.smiles_col and not chem.HAVE_RDKIT:
        report["run"]["structure_skipped_because"] = (
            f"rdkit not importable: {chem.RDKIT_IMPORT_ERROR}"
        )

    # (c)
    X = None
    if feat_cols:
        X = tabular.feature_matrix(df, feat_cols)
    elif mols is not None:
        D = chem.descriptors(mols, ok_idx)
        if np.isnan(D).all(axis=1).any():
            med = np.nanmedian(D, axis=0)
            bad = np.isnan(D).all(axis=1)
            D[bad] = med
        X = D
        report["run"]["feature_cols"] = chem.DESCRIPTOR_COLS
        report["run"]["feature_source"] = "derived_from_smiles"
        feat_cols = chem.DESCRIPTOR_COLS

    folds_info = None
    if X is None:
        report["run"]["split_comparison_skipped_because"] = (
            "no numeric feature columns and no usable SMILES column"
        )
        report["split_comparison"] = None
    elif len(set(groups)) < args.folds:
        report["run"]["split_comparison_skipped_because"] = (
            f"only {len(set(groups))} groups, fewer than --folds {args.folds}"
        )
        report["split_comparison"] = None
    else:
        report["split_comparison"], folds_info = diagnostics.split_comparison(
            X, y, groups, args.folds, args.repeats, args.bootstrap, cv_seed,
            boot_seed=boot_seed,
        )
        print(f"  (c) random {report['split_comparison']['random_split']['mean_auc_over_repeats']:.4f} "
              f"vs grouped {report['split_comparison']['group_split']['auc']:.4f}", flush=True)
    _write(args.out, report)

    # (c2) optional arm comparison
    if args.compare_feature_cols and X is not None:
        cmp_cols = [c.strip() for c in args.compare_feature_cols.split(",")]
        missing = [c for c in cmp_cols if c not in df.columns]
        if missing:
            raise tabular.InputError(f"--compare-feature-cols not in table: {missing}")
        Xb = tabular.feature_matrix(df, cmp_cols)
        na, nb = (args.arm_names.split(",") + ["arm_a", "arm_b"])[:2]
        report["arm_comparison"] = diagnostics.arm_comparison(
            X, Xb, y, groups, args.folds, args.repeats, args.bootstrap, cv_seed,
            na.strip(), nb.strip(), boot_seed=boot_seed,
        )
        report["arm_comparison"]["arm_a_cols"] = feat_cols
        report["arm_comparison"]["arm_b_cols"] = cmp_cols
        _write(args.out, report)
        print(f"  (c2) grouped delta {report['arm_comparison']['group_split']['delta_a_minus_b']:+.4f}",
              flush=True)

    # (d)
    if mols is not None:
        idx = ok_idx
        fps = chem.fingerprints(mols, idx)
        labs = y[idx]
        same, cross, k = chem.nn_tanimoto_by_class(fps, labs, args.nn_sample, nn_seed)
        scafs_ok = chem.murcko(mols, idx)
        st = {
            "n_rows": int(len(df)),
            "n_parsed": int(len(idx)),
            "fingerprint": "Morgan radius 2, 2048 bits",
            "nn_sample_size": k,
            "nn_seed": nn_seed,
            "nn_tanimoto_same_class_mean": round(same, 4),
            "nn_tanimoto_cross_class_mean": round(cross, 4),
        }
        st.update(chem.scaffold_class_overlap(scafs_ok, labs))
        if folds_info is not None:
            full_scaf = np.array([""] * len(df), dtype=object)
            full_scaf[idx] = scafs_ok
            st["scaffold_leakage_random_split"] = diagnostics.split_scaffold_leakage(
                full_scaf, folds_info["random_fold"]
            )
            st["scaffold_leakage_group_split"] = diagnostics.split_scaffold_leakage(
                full_scaf, folds_info["group_fold"]
            )
        report["structure"] = st
        print(f"  (d) nn same {same:.4f} cross {cross:.4f} "
              f"scaffold Jaccard {st['scaffold_jaccard_pos_vs_neg']}", flush=True)
    else:
        report["structure"] = None
    _write(args.out, report)

    # (e)
    thresholds = {
        "group_auc": args.thresh_group_auc,
        "purity": args.thresh_purity,
        "split_drop": args.thresh_split_drop,
        "nn_gap": args.thresh_nn_gap,
        "scaffold_jaccard": args.thresh_scaffold_jaccard,
    }
    report["verdict"] = diagnostics.verdict(report, thresholds)
    report["run"]["wall_seconds"] = round(time.time() - t0, 2)
    _write(args.out, report)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    print(f"  (e) VERDICT {report['verdict']['level']} "
          f"({report['verdict']['n_flags']} flags) seed={seed} "
          f"[{report['run']['wall_seconds']}s]", flush=True)
    return 0


def run() -> None:
    """Console-script entry point (`contamcheck ...`)."""
    try:
        sys.exit(main())
    except tabular.InputError as exc:
        print(f"contamcheck: input error: {exc}", file=sys.stderr)
        sys.exit(2)
