from __future__ import annotations

import pandas as pd

from core.stib_historical import fetch_historical_segment_speeds
from core.stib_live import fetch_live_segment_speeds


def build_historical_fill_values(
    historical_segment_df: pd.DataFrame,
    method: str = "latest",
) -> pd.DataFrame:
    """
    Build one historical fallback value per segment.

    Parameters
    ----------
    historical_segment_df : pd.DataFrame
        Historical segment-level speeds with columns:
        - bucket_time
        - segment_id
        - avg_speed_kmh
        - sample_count
    method : str
        Strategy used to convert the historical time series into one fallback value
        per segment.

        Supported values:
        - "latest": use the most recent available historical value
        - "mean": use the mean over the historical window

    Returns
    -------
    pd.DataFrame
        Columns:
        - segment_id
        - historical_speed_kmh
        - historical_bucket_count
        - interpolation_method
    """
    if historical_segment_df.empty:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "historical_speed_kmh",
                "historical_bucket_count",
                "interpolation_method",
            ]
        )

    working_df = historical_segment_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str).str.strip()

    if method == "latest":
        latest_rows = (
            working_df.sort_values(["segment_id", "bucket_time"])
            .groupby("segment_id", as_index=False)
            .tail(1)
            .copy()
        )

        bucket_count_df = (
            working_df.groupby("segment_id", as_index=False)
            .agg(historical_bucket_count=("avg_speed_kmh", lambda s: s.notna().sum()))
        )

        result = latest_rows[["segment_id", "avg_speed_kmh"]].rename(
            columns={"avg_speed_kmh": "historical_speed_kmh"}
        )

        result = result.merge(bucket_count_df, on="segment_id", how="left")
        result["interpolation_method"] = "historical_latest"
        return result

    if method == "mean":
        result = (
            working_df.groupby("segment_id", as_index=False)
            .agg(
                historical_speed_kmh=("avg_speed_kmh", "mean"),
                historical_bucket_count=("avg_speed_kmh", lambda s: s.notna().sum()),
            )
            .copy()
        )

        result["interpolation_method"] = "historical_mean"
        return result

    raise ValueError(
        f"Unsupported interpolation method: {method}. "
        "Supported values are 'latest' and 'mean'."
    )


def build_completed_stib_snapshot(
    live_segment_df: pd.DataFrame,
    historical_segment_df: pd.DataFrame,
    interpolation_method: str = "latest",
) -> pd.DataFrame:
    """
    Build the completed STIB snapshot.

    Rules
    -----
    1. If live speed exists, keep it.
    2. If live speed is missing, use the historical fallback.
    3. If both are missing, keep the final speed empty.

    Parameters
    ----------
    live_segment_df : pd.DataFrame
        Live segment snapshot. Expected columns:
        - snapshot_time
        - segment_id
        - avg_speed_kmh
        - sample_count

    historical_segment_df : pd.DataFrame
        Historical segment-level time series. Expected columns:
        - bucket_time
        - segment_id
        - avg_speed_kmh
        - sample_count

    interpolation_method : str
        Historical fallback strategy:
        - "latest"
        - "mean"

    Returns
    -------
    pd.DataFrame
        Columns:
        - snapshot_time
        - segment_id
        - speed_kmh
        - live_speed_kmh
        - historical_speed_kmh
        - interpolation
        - interpolation_method
        - live_sample_count
        - historical_bucket_count
    """
    live_df = live_segment_df.copy()
    live_df["segment_id"] = live_df["segment_id"].astype(str).str.strip()

    live_df = live_df.rename(
        columns={
            "avg_speed_kmh": "live_speed_kmh",
            "sample_count": "live_sample_count",
        }
    )

    historical_fill_df = build_historical_fill_values(
        historical_segment_df=historical_segment_df,
        method=interpolation_method,
    )

    result = live_df.merge(
        historical_fill_df,
        on="segment_id",
        how="left",
    )

    result["interpolation"] = result["live_speed_kmh"].isna() & result[
        "historical_speed_kmh"
    ].notna()

    result["speed_kmh"] = result["live_speed_kmh"].where(
        result["live_speed_kmh"].notna(),
        result["historical_speed_kmh"],
    )

    result["interpolation_method"] = result["interpolation_method"].where(
        result["interpolation"],
        "live",
    )

    result = result[
        [
            "snapshot_time",
            "segment_id",
            "speed_kmh",
            "live_speed_kmh",
            "historical_speed_kmh",
            "interpolation",
            "interpolation_method",
            "live_sample_count",
            "historical_bucket_count",
        ]
    ].copy()

    result = result.sort_values("segment_id").reset_index(drop=True)
    return result


def fetch_completed_stib_snapshot(
    token: str,
    gpkg_path: str,
    lookback_minutes: int = 60,
    bucket_minutes: int = 5,
    interpolation_method: str = "latest",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch live and historical STIB data, then build the completed snapshot.

    Parameters
    ----------
    token : str
        MobilityTwin API token.
    gpkg_path : str
        Path to the segment GPKG file.
    lookback_minutes : int
        Historical window size in minutes.
    bucket_minutes : int
        Historical aggregation bucket size in minutes.
    interpolation_method : str
        Historical fallback strategy:
        - "latest"
        - "mean"

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - completed_snapshot_df
        - live_segment_df
        - historical_segment_df
    """
    live_segment_df, _ = fetch_live_segment_speeds(
        token=token,
        gpkg_path=gpkg_path,
    )

    historical_segment_df = fetch_historical_segment_speeds(
        token=token,
        gpkg_path=gpkg_path,
        lookback_minutes=lookback_minutes,
        bucket_minutes=bucket_minutes,
    )

    completed_snapshot_df = build_completed_stib_snapshot(
        live_segment_df=live_segment_df,
        historical_segment_df=historical_segment_df,
        interpolation_method=interpolation_method,
    )

    return completed_snapshot_df, live_segment_df, historical_segment_df
