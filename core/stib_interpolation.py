from __future__ import annotations

import pandas as pd

from core.stib_historical import fetch_historical_segment_speeds
from core.stib_live import fetch_live_segment_speeds


def build_interpolated_values(
    historical_segment_df: pd.DataFrame,
    method: str = "latest",
) -> pd.DataFrame:
    """
    Build one interpolated value per segment from the historical time series.

    Parameters
    ----------
    historical_segment_df : pd.DataFrame
        Historical segment-level speeds with columns:
        - bucket_time
        - segment_id
        - avg_speed_kmh
        - sample_count

    method : str
        Strategy used to convert the historical time series into one value
        for the current snapshot.

        Supported values:
        - "latest": use the most recent available historical value
        - "mean": use the mean over the historical window

    Returns
    -------
    pd.DataFrame
        Columns:
        - segment_id
        - interpolated_speed_kmh
    """
    if historical_segment_df.empty:
        return pd.DataFrame(columns=["segment_id", "interpolated_speed_kmh"])

    working_df = historical_segment_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str).str.strip()

    if method == "latest":
        result = (
            working_df.sort_values(["segment_id", "bucket_time"])
            .groupby("segment_id", as_index=False)
            .tail(1)[["segment_id", "avg_speed_kmh"]]
            .rename(columns={"avg_speed_kmh": "interpolated_speed_kmh"})
            .reset_index(drop=True)
        )
        return result

    if method == "mean":
        result = (
            working_df.groupby("segment_id", as_index=False)
            .agg(interpolated_speed_kmh=("avg_speed_kmh", "mean"))
            .reset_index(drop=True)
        )
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
    Build the final current-time STIB snapshot.

    Rules
    -----
    1. Keep live speed if it exists.
    2. If live speed is missing, use the interpolated value.
    3. The final output represents only the current time.

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
        - live_speed_kmh
        - interpolated_speed_kmh
        - final_speed_kmh
        - interpolation
    """
    live_df = live_segment_df.copy()
    live_df["segment_id"] = live_df["segment_id"].astype(str).str.strip()

    live_df = live_df.rename(
        columns={
            "avg_speed_kmh": "live_speed_kmh",
        }
    )

    interpolated_df = build_interpolated_values(
        historical_segment_df=historical_segment_df,
        method=interpolation_method,
    )

    result = live_df.merge(
        interpolated_df,
        on="segment_id",
        how="left",
    )

    result["interpolation"] = result["live_speed_kmh"].isna() & result[
        "interpolated_speed_kmh"
    ].notna()

    result["final_speed_kmh"] = result["live_speed_kmh"].where(
        result["live_speed_kmh"].notna(),
        result["interpolated_speed_kmh"],
    )

    result = result[
        [
            "snapshot_time",
            "segment_id",
            "live_speed_kmh",
            "interpolated_speed_kmh",
            "final_speed_kmh",
            "interpolation",
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
    Fetch live STIB data, fetch historical STIB data, and build the final
    current-time completed snapshot.

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
