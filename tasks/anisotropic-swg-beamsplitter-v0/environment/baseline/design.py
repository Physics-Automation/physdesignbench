"""Starter submission for the anisotropic SWG beamsplitter task."""

import photonforge as pf


def build_submission():
    """Return one component using silicon (1, 0) and air-void (2, 0) masks."""
    return pf.Component("ANISOTROPIC_SWG_BEAMSPLITTER_FINAL")
