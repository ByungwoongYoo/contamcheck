"""Generate the three synthetic example tables shipped in examples/.

Fully synthetic (no real data), seeded, so the CSVs can be regenerated
byte-for-byte:  python examples/make_examples.py

  clean_manufacturing.csv      batches differ, but batch says nothing about the label  -> CLEAN
  watch_materials.csv          source papers have skewed (not pure) label rates        -> WATCH
  contaminated_measurement.csv each campaign measured mostly one class on its own
                               instrument, which shifts the features                   -> CONTAMINATED
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261101


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def clean_manufacturing(seed=SEED):
    """Injection-moulding parts, label = defect (0/1). 60 batches x 20 parts.
    Batches have real, label-independent process offsets."""
    rng = np.random.default_rng(seed)
    n_b, per = 60, 20
    batch = np.repeat([f"B{i:03d}" for i in range(n_b)], per)
    line = np.repeat(rng.choice(["L1", "L2", "L3", "L4"], n_b), per)
    off = np.repeat(rng.normal(0, 0.4, (n_b, 5)), per, axis=0)
    X = rng.normal(0, 1, (n_b * per, 5)) + off
    z = 1.4 * (X[:, 0] - off[:, 0]) - 1.0 * (X[:, 2] - off[:, 2]) + 0.6 * (X[:, 4] - off[:, 4]) - 0.3
    y = (rng.random(len(z)) < _sigmoid(z)).astype(int)
    df = pd.DataFrame({
        "part_id": [f"P{i:05d}" for i in range(len(y))],
        "batch": batch,
        "line": line,
        "melt_temp_C": np.round(230 + 8 * X[:, 0], 2),
        "hold_pressure_bar": np.round(600 + 40 * X[:, 1], 2),
        "cooling_time_s": np.round(18 + 2.5 * X[:, 2], 3),
        "humidity_pct": np.round(45 + 6 * X[:, 3], 2),
        "screw_speed_rpm": np.round(120 + 15 * X[:, 4], 2),
        "defect": y,
    })
    return df


def watch_materials(seed=SEED + 1):
    """Thin-film samples compiled from 70 source papers, label = 'conductive'.
    Each paper studied a skewed (but mixed) set of samples; the features carry a
    real signal that generalises across papers."""
    rng = np.random.default_rng(seed)
    n_p = 70
    sizes = rng.integers(8, 25, n_p)
    rate = np.where(rng.random(n_p) < 0.5, 0.18, 0.82)
    paper, y = [], []
    for i, (s, r) in enumerate(zip(sizes, rate)):
        paper += [f"doi:10.5555/synth.{i:04d}"] * s
        y += list((rng.random(s) < r).astype(int))
    y = np.asarray(y)
    n = len(y)
    X = rng.normal(0, 1, (n, 6))
    X[:, 0] += 1.3 * (y - 0.5) * 2
    X[:, 3] -= 0.8 * (y - 0.5) * 2
    df = pd.DataFrame({
        "sample_id": [f"S{i:05d}" for i in range(n)],
        "source_paper": paper,
        "dopant_frac": np.round(0.05 + 0.02 * X[:, 0], 4),
        "anneal_temp_K": np.round(700 + 50 * X[:, 1], 1),
        "film_thickness_nm": np.round(120 + 20 * X[:, 2], 2),
        "grain_size_nm": np.round(40 + 8 * X[:, 3], 2),
        "lattice_a_A": np.round(3.9 + 0.02 * X[:, 4], 4),
        "o2_partial_pressure_mTorr": np.round(10 + 2 * X[:, 5], 3),
        "conductive": np.where(y == 1, "yes", "no"),
    })
    return df


def contaminated_measurement(seed=SEED + 2):
    """Alloy coupons tested for corrosion pass/fail across 48 measurement
    campaigns. Each campaign mostly tested one class of sample and used one of
    8 instruments whose calibration shifts the recorded features; the true
    physical signal is weak."""
    rng = np.random.default_rng(seed)
    n_c = 48
    sizes = rng.integers(15, 45, n_c)
    camp_rate = np.where(np.arange(n_c) % 2 == 0, 0.05, 0.95)
    inst_of_c = rng.integers(0, 8, n_c)
    inst_off = rng.normal(0, 1.0, (8, 6))
    camp_off = rng.normal(0, 0.8, (n_c, 6))
    camp, inst, lab, y, X = [], [], [], [], []
    for c in range(n_c):
        s = sizes[c]
        yc = (rng.random(s) < camp_rate[c]).astype(int)
        xc = rng.normal(0, 1, (s, 6)) + inst_off[inst_of_c[c]] + camp_off[c]
        xc[:, 0] += 0.25 * (yc - 0.5) * 2  # weak genuine signal
        camp += [f"campaign_{c:02d}"] * s
        inst += [f"instrument_{inst_of_c[c]}"] * s
        lab += list(rng.choice(["lab_north", "lab_south", "lab_east"], s))
        y.append(yc)
        X.append(xc)
    y = np.concatenate(y)
    X = np.vstack(X)
    df = pd.DataFrame({
        "coupon_id": [f"C{i:05d}" for i in range(len(y))],
        "campaign": camp,
        "instrument": inst,
        "lab": lab,
        "cr_wt_pct": np.round(18 + 1.5 * X[:, 0], 3),
        "ni_wt_pct": np.round(9 + 1.0 * X[:, 1], 3),
        "mo_wt_pct": np.round(2 + 0.4 * X[:, 2], 3),
        "hardness_HV": np.round(200 + 15 * X[:, 3], 2),
        "surface_roughness_um": np.round(0.8 + 0.1 * X[:, 4], 4),
        "chloride_ppm": np.round(500 + 60 * X[:, 5], 1),
        "corrosion_pass": y,
    })
    return df


if __name__ == "__main__":
    for name, fn in (("clean_manufacturing", clean_manufacturing),
                     ("watch_materials", watch_materials),
                     ("contaminated_measurement", contaminated_measurement)):
        df = fn()
        path = os.path.join(HERE, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}: {len(df)} rows x {df.shape[1]} cols")
