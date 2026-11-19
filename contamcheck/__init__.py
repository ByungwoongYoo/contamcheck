"""contamcheck - provenance-shortcut diagnostics for labelled tabular datasets.

Tests whether a grouping column (source publication, measurement campaign,
batch, instrument, lab, site) predicts a binary label and compares random and
group-held-out cross-validation. Uses NumPy, pandas, and scikit-learn, with
optional RDKit for molecular structure diagnostics.
"""
__version__ = "0.2.0"
