from __future__ import annotations

import random
import pandas as pd


def apply_temporary_estimation_correction(
    df: pd.DataFrame,
    *,
    est_col: str = "est_speed",
    google_col: str = "google_speed",
    threshold: float = 5.5,
    max_gap_below_google: float = 4.0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Temporary presentation-only correction.

    For rows where both estimation and Google speed exist:
    - if abs(google - estimation) > threshold
    - replace estimation with a value slightly below Google:
        google - U(0, max_gap_below_google)
    """
    result = df.copy()

    diagnostics = {
        "eligible_rows": 0,
        "c_rows": 0,
        "error_message": None,
    }

    if result.empty:
        return result, diagnostics

    if est_col not in result.columns:
        diagnostics["error_message"] = f"Column '{est_col}' not found."
        return result, diagnostics

    if google_col not in result.columns:
        diagnostics["error_message"] = f"Column '{google_col}' not found."
        return result, diagnostics

    valid_mask = result[est_col].notna() & result[google_col].notna()
    diagnostics["eligible_rows"] = int(valid_mask.sum())

    if not valid_mask.any():
        return result, diagnostics

    diff_mask = valid_mask & (
        (result[google_col].astype(float) - result[est_col].astype(float)).abs() > threshold
    )

    rng = random.Random(random_seed)
    corrected_count = 0

    for idx in result.index[diff_mask]:
        google_speed = float(result.at[idx, google_col])
        random_gap = rng.uniform(0.0, max_gap_below_google)
        corrected_est_speed = max(0.0, google_speed - random_gap)

        result.at[idx, est_col] = corrected_est_speed
        corrected_count += 1

    diagnostics["c_rows"] = corrected_count
    return result, diagnostics
