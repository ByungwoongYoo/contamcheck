"""Turn the read-only nova-mechgate artifacts into generic contamcheck tables.

Reads only; writes into contamination-check/results/tables/.
Usage: python repro/prepare_tables.py <nova-mechgate-dir> <out-dir>
"""
import csv
import json
import os
import sys

MG_FEAT = ["ompf_percentile", "multiporin_percentile", "lipid_percentile",
           "efflux_percentile", "activity_percentile"]
MG_NUM = ["ompf_score", "multiporin_score", "lipid_score", "efflux_score",
          "activity_score"] + MG_FEAT
BASE = ["clogp", "formal_charge", "mw"]


def write_tsv(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {path}: {len(rows)} rows x {len(cols)} cols")


def full_set(src, dst, tag):
    data = json.load(open(src))
    rows = []
    for r in data:
        rows.append({
            "molecule_chembl_id": r["molecule_chembl_id"],
            "smiles": r["smiles"],
            "label": r["label"],
            "document": r["docs"][0] if r.get("docs") else "NA",
            "n_docs": r.get("n_docs", len(r.get("docs", []))),
            "src_id": r["srcs"][0] if r.get("srcs") else "NA",
        })
    write_tsv(dst, rows, ["molecule_chembl_id", "smiles", "label", "document",
                          "n_docs", "src_id"])
    return len(rows)


def gate_set(mech_dir, dst):
    """Reproduces nova-mechgate scripts/gate.py load(): features joined to the
    MechGate score table on SMILES, dropping rows with a non-numeric column."""
    feats = [f for f in json.load(
        open(os.path.join(mech_dir, "artifacts/features_cotested.json"))) if f]
    mg = {}
    with open(os.path.join(mech_dir, "artifacts/mechgate_cotested.tsv")) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            mg[row["SMILES"]] = row
    rows = []
    for f in feats:
        m = mg.get(f["smiles"])
        if m is None:
            continue
        try:
            vals = {k: float(m[k]) for k in MG_NUM}
        except (ValueError, TypeError):
            continue
        if any(v != v for v in vals.values()):
            continue
        out = {
            "molecule_chembl_id": f["molecule_chembl_id"],
            "smiles": f["smiles"],
            "label": f["label"],
            "document": f["docs"][0] if f.get("docs") else "NA",
            "scaffold": f.get("scaffold", ""),
        }
        out.update({k: f[k] for k in BASE})
        out.update(vals)
        rows.append(out)
    cols = (["molecule_chembl_id", "smiles", "label", "document", "scaffold"]
            + BASE + MG_NUM)
    write_tsv(dst, rows, cols)
    return len(rows)


if __name__ == "__main__":
    mech, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    manifest = {
        "source_dir": os.path.abspath(mech),
        "n_cotested_full": full_set(os.path.join(mech, "data/cotested_set.json"),
                                    os.path.join(out, "cotested_full.tsv"), "cotested"),
        "n_naive_full": full_set(os.path.join(mech, "data/naive_set.json"),
                                 os.path.join(out, "naive_full.tsv"), "naive"),
        "n_gate_joined": gate_set(mech, os.path.join(out, "gate_cotested.tsv")),
    }
    json.dump(manifest, open(os.path.join(out, "tables_manifest.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))
