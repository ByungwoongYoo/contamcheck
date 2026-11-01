"""contamcheck - provenance-shortcut diagnostics for labelled tabular datasets.

Tests whether a grouping column (source publication, measurement campaign,
batch, instrument, lab, site) predicts a binary label on its own and so inflates
random-split cross-validation. Depends only on numpy / pandas / scikit-learn
(and optionally rdkit for the molecule-specific structure diagnostics).
"""
__version__ = "0.1.0"
