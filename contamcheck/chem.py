"""Optional RDKit-backed structure diagnostics. Import is lazy and guarded."""
from __future__ import annotations

import numpy as np

RDKIT_IMPORT_ERROR = None
try:  # pragma: no cover - environment dependent
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Crippen, DataStructs, Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except Exception as exc:  # pragma: no cover
    HAVE_RDKIT = False
    RDKIT_IMPORT_ERROR = repr(exc)


def parse(smiles_list):
    """Return (mols, ok_index). Unparseable SMILES become None and are reported."""
    mols, ok = [], []
    for i, s in enumerate(smiles_list):
        m = None
        if isinstance(s, str) and s:
            m = Chem.MolFromSmiles(s)
        mols.append(m)
        if m is not None:
            ok.append(i)
    return mols, np.asarray(ok, dtype=int)


def fingerprints(mols, idx, radius=2, nbits=2048):
    return [AllChem.GetMorganFingerprintAsBitVect(mols[i], radius, nbits) for i in idx]


def murcko(mols, idx):
    out = []
    for i in idx:
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mols[i]))
        except Exception:
            out.append("")
    return out


def nn_tanimoto_by_class(fps, labels, sample: int, seed: int):
    """Mean nearest-neighbour Tanimoto within class vs across class.

    Samples at most `sample` query molecules (seeded) for runtime; each query is
    compared against the full set.
    """
    labels = np.asarray(labels)
    n = len(fps)
    rng = np.random.default_rng(seed)
    k = min(sample, n)
    idx = rng.choice(n, k, replace=False)
    same, cross = [], []
    for i in idx:
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[int(i)], fps))
        sims[i] = -1.0
        m_same = labels == labels[i]
        m_same[i] = False
        if m_same.any():
            same.append(float(sims[m_same].max()))
        m_cross = labels != labels[i]
        if m_cross.any():
            cross.append(float(sims[m_cross].max()))
    return (
        float(np.mean(same)) if same else float("nan"),
        float(np.mean(cross)) if cross else float("nan"),
        int(k),
    )


def scaffold_class_overlap(scaffolds, labels):
    labels = np.asarray(labels)
    sp = {s for s, l in zip(scaffolds, labels) if l == 1 and s}
    sn = {s for s, l in zip(scaffolds, labels) if l == 0 and s}
    union = sp | sn
    jac = len(sp & sn) / len(union) if union else float("nan")
    return {
        "n_scaffolds_pos": len(sp),
        "n_scaffolds_neg": len(sn),
        "n_scaffolds_shared": len(sp & sn),
        "scaffold_jaccard_pos_vs_neg": round(jac, 4) if jac == jac else None,
    }


DESCRIPTOR_COLS = [
    "cc_mw",
    "cc_clogp",
    "cc_tpsa",
    "cc_hbd",
    "cc_hba",
    "cc_rotb",
    "cc_rings",
    "cc_heavy",
    "cc_formal_charge",
    "cc_fracsp3",
]


def descriptors(mols, idx):
    """Cheap 2-D descriptors used only when the table has no numeric features."""
    rows = np.full((len(mols), len(DESCRIPTOR_COLS)), np.nan)
    for i in idx:
        m = mols[i]
        rows[i] = [
            Descriptors.MolWt(m),
            Crippen.MolLogP(m),
            Descriptors.TPSA(m),
            Descriptors.NumHDonors(m),
            Descriptors.NumHAcceptors(m),
            Descriptors.NumRotatableBonds(m),
            m.GetRingInfo().NumRings(),
            m.GetNumHeavyAtoms(),
            Chem.GetFormalCharge(m),
            Descriptors.FractionCSP3(m),
        ]
    return rows
