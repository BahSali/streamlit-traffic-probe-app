from __future__ import annotations

import pandas as pd


def build_empty_estimation_diagnostics() -> dict:
    return {
        "estimation_mode": "tmp_baseline",
        "snapshot_found": False,
        "snapshot_time": None,
        "map_has_id_column": False,
        "matched_segments": 0,
        "error_message": None,
    }


def get_snapshot_timestamp(completed_snapshot_df: pd.DataFrame) -> pd.Timestamp | None:
    """
    Extract a single snapshot timestamp from the completed STIB snapshot.
    """
    if completed_snapshot_df.empty or "snapshot_time" not in completed_snapshot_df.columns:
        return None

    valid_times = completed_snapshot_df["snapshot_time"].dropna()
    if valid_times.empty:
        return None

    return pd.to_datetime(valid_times.iloc[0], errors="coerce")


def build_completed_snapshot_speed_lookup(
    completed_snapshot_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Build a segment_id -> final_speed_kmh lookup from the completed STIB snapshot.
    """
    if completed_snapshot_df.empty:
        return {}

    required_columns = {"segment_id", "final_speed_kmh"}
    missing_columns = required_columns - set(completed_snapshot_df.columns)
    if missing_columns:
        raise ValueError(
            f"Completed snapshot is missing required columns: {sorted(missing_columns)}"
        )

    working_df = completed_snapshot_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str).str.strip()
    working_df["final_speed_kmh"] = pd.to_numeric(
        working_df["final_speed_kmh"],
        errors="coerce",
    )

    working_df = working_df.dropna(subset=["final_speed_kmh"]).copy()

    return dict(zip(working_df["segment_id"], working_df["final_speed_kmh"]))


def attach_tmp_estimated_speeds(
    gdf: pd.DataFrame,
    completed_snapshot_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Temporary step-1 estimator for map 2.

    Current behavior
    ----------------
    Uses the completed STIB snapshot final speed as the map-2 estimated speed.
    This removes the random placeholder while keeping the application runnable.

    Later this same file will be extended to:
    - build historical Brussels Mobility features
    - load the trained .pt model
    - run inference
    - replace this temporary baseline
    """
    result = gdf.copy()
    diagnostics = build_empty_estimation_diagnostics()

    diagnostics["map_has_id_column"] = "id" in result.columns

    if "id" not in result.columns:
        diagnostics["error_message"] = "The Brussels map file does not contain an 'id' column."
        result["est_speed"] = pd.NA
        return result, diagnostics

    if completed_snapshot_df.empty:
        diagnostics["error_message"] = "Completed STIB snapshot is empty."
        result["est_speed"] = pd.NA
        return result, diagnostics

    snapshot_time = get_snapshot_timestamp(completed_snapshot_df)
    diagnostics["snapshot_found"] = snapshot_time is not None
    diagnostics["snapshot_time"] = (
        snapshot_time.isoformat() if snapshot_time is not None else None
    )

    speed_lookup = build_completed_snapshot_speed_lookup(completed_snapshot_df)

    result["segment_id_str"] = result["id"].astype(str).str.strip()
    result["est_speed"] = result["segment_id_str"].map(speed_lookup)

    diagnostics["matched_segments"] = int(result["est_speed"].notna().sum())

    return result, diagnostics
