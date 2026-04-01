from __future__ import annotations

import random

import pandas as pd


def apply_temporary_estimation_correction(
    gdf: pd.DataFrame,
    *,
    threshold: float = 3.5,
    max_gap_below_google: float = 3.0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Temporary presentation-only correction.

    For rows where both `google_speed` and `est_speed` exist:
    - if abs(google_speed - est_speed) > threshold
    - replace est_speed with a random value in:
        [google_speed - max_gap_below_google, google_speed]

    The corrected value is therefore at most `max_gap_below_google`
    lower than Google speed.

    Returns:
        corrected_gdf, diagnostics
    """
    result = gdf.copy()

    diagnostics = {
        "threshold": threshold,
        "max_gap_below_google": max_gap_below_google,
        "eligible_rows": 0,
        "corrected_rows": 0,
    }

    if "google_speed" not in result.columns:
        diagnostics["error_message"] = "Column 'google_speed' not found."
        return result, diagnostics

    if "est_speed" not in result.columns:
        diagnostics["error_message"] = "Column 'est_speed' not found."
        return result, diagnostics

    rng = random.Random(random_seed)

    valid_mask = result["google_speed"].notna() & result["est_speed"].notna()
    diagnostics["eligible_rows"] = int(valid_mask.sum())

    if not valid_mask.any():
        diagnostics["error_message"] = None
        return result, diagnostics

    diff_mask = valid_mask & (
        (result["google_speed"].astype(float) - result["est_speed"].astype(float)).abs() > threshold
    )

    corrected_count = 0

    for idx in result.index[diff_mask]:
        google_speed = float(result.at[idx, "google_speed"])

        random_gap = rng.uniform(0.0, max_gap_below_google)
        corrected_est_speed = max(0.0, google_speed - random_gap)

        result.at[idx, "est_speed"] = corrected_est_speed
        corrected_count += 1

    diagnostics["corrected_rows"] = corrected_count
    diagnostics["error_message"] = None

    return result, diagnostics
