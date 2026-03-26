from __future__ import annotations

import pandas as pd

from core.stib_historical import fetch_historical_segment_speeds
from core.stib_interpolation import build_completed_stib_snapshot
from core.stib_live import fetch_live_segment_speeds


def get_live_stib_snapshot(
    token: str,
    gpkg_path: str,
) -> pd.DataFrame:
    """
    Fetch the current live STIB segment snapshot.

    Parameters
    ----------
    token : str
        MobilityTwin API token.
    gpkg_path : str
        Path to the GPKG file used for segment mapping.

    Returns
    -------
    pd.DataFrame
        Columns:
        - snapshot_time
        - segment_id
        - avg_speed_kmh
        - sample_count
    """
    live_segment_df, _ = fetch_live_segment_speeds(
        token=token,
        gpkg_path=gpkg_path,
    )

    return live_segment_df.copy()


def get_completed_stib_snapshot(
    token: str,
    gpkg_path: str,
    lookback_minutes: int = 60,
    bucket_minutes: int = 5,
    interpolation_method: str = "latest",
) -> pd.DataFrame:
    """
    Build the completed current-time STIB snapshot.

    The completed snapshot keeps live values where available and uses
    historical interpolation only for segments with missing live speed.

    Parameters
    ----------
    token : str
        MobilityTwin API token.
    gpkg_path : str
        Path to the GPKG file used for segment mapping.
    lookback_minutes : int
        Historical lookback window in minutes.
    bucket_minutes : int
        Historical aggregation bucket size in minutes.
    interpolation_method : str
        Historical interpolation strategy.
        Supported values:
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

    return completed_snapshot_df.copy()


def run_stib_pipeline(
    token: str,
    gpkg_path: str,
    lookback_minutes: int = 60,
    bucket_minutes: int = 5,
    interpolation_method: str = "latest",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the STIB pipeline and return both outputs needed by the app.

    Output 1:
        live_snapshot_df
        Used by map 1 only.

    Output 2:
        completed_snapshot_df
        Used for download and later for map 2.

    Parameters
    ----------
    token : str
        MobilityTwin API token.
    gpkg_path : str
        Path to the GPKG file used for segment mapping.
    lookback_minutes : int
        Historical lookback window in minutes.
    bucket_minutes : int
        Historical aggregation bucket size in minutes.
    interpolation_method : str
        Historical interpolation strategy.
        Supported values:
        - "latest"
        - "mean"

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - live_snapshot_df
        - completed_snapshot_df
    """
    live_snapshot_df = get_live_stib_snapshot(
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
        live_segment_df=live_snapshot_df,
        historical_segment_df=historical_segment_df,
        interpolation_method=interpolation_method,
    )

    return live_snapshot_df.copy(), completed_snapshot_df.copy()
