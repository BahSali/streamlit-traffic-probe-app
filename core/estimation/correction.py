"""Demo correction of estimated speeds, switched by
core.config.APPLY_DEMO_SPEED_CORRECTION.

Where both an estimate and a Google speed exist and they differ by more than
DEMO_CORRECTION_THRESHOLD_KMH, the displayed estimate becomes
``google - U(0, DEMO_CORRECTION_MAX_GAP_KMH)`` (never below 0). The random gap
is seeded per segment, so a segment gets the same value on the map, in the
charts and in the download. Inputs are never modified: these functions return
a new Series.
"""
from __future__ import annotations

import random

import pandas as pd

from core import config

DEMO_CORRECTION_THRESHOLD_KMH = 8.5
DEMO_CORRECTION_MAX_GAP_KMH = 5.0
DEMO_CORRECTION_SEED = 42


def demo_corrected_speeds(
    df: pd.DataFrame,
    *,
    est_col: str,
    google_col: str,
    id_col: str,
    threshold: float = DEMO_CORRECTION_THRESHOLD_KMH,
    max_gap_below_google: float = DEMO_CORRECTION_MAX_GAP_KMH,
    random_seed: int = DEMO_CORRECTION_SEED,
) -> pd.Series:
    """Return df[est_col] with the demo correction applied (df is unchanged)."""
    estimates = pd.to_numeric(df[est_col], errors="coerce")
    corrected = estimates.astype(float).copy()

    if df.empty or google_col not in df.columns:
        return corrected

    google = pd.to_numeric(df[google_col], errors="coerce")
    to_correct = estimates.notna() & google.notna() & ((google - estimates).abs() > threshold)

    for idx in df.index[to_correct]:
        segment_id = str(df.at[idx, id_col]).strip()
        gap = random.Random(f"{random_seed}:{segment_id}").uniform(0.0, max_gap_below_google)
        corrected.at[idx] = max(0.0, float(google.at[idx]) - gap)

    return corrected


def displayed_estimates(
    df: pd.DataFrame,
    *,
    est_col: str,
    google_col: str,
    id_col: str,
) -> pd.Series:
    """Estimated speeds to show in the app and downloads.

    Demo-corrected when config.APPLY_DEMO_SPEED_CORRECTION is True, otherwise
    the values of est_col unchanged.
    """
    if config.APPLY_DEMO_SPEED_CORRECTION:
        return demo_corrected_speeds(df, est_col=est_col, google_col=google_col, id_col=id_col)
    return pd.to_numeric(df[est_col], errors="coerce")
