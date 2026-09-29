from __future__ import annotations

import os

import pandas as pd
import geopandas as gpd
import streamlit as st


MOBILITY_TWIN_SECRET_KEY = "MOBILITY_TWIN_TOKEN"


def get_mobility_twin_token() -> str | None:
    """MobilityTwin (api.mobilitytwin.brussels) token from Streamlit secrets or the environment."""
    try:
        token = st.secrets.get(MOBILITY_TWIN_SECRET_KEY)
    except Exception:  # no secrets.toml at all
        token = None
    return token or os.environ.get(MOBILITY_TWIN_SECRET_KEY)


@st.cache_data(show_spinner=False)
def load_csv(path: str, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


@st.cache_data(show_spinner=False)
def load_gpkg(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


@st.cache_data(show_spinner=False, ttl=90)
def load_stib_pipeline_outputs(
    token: str,
    gpkg_path: str,
    lookback_minutes: int = 60,
    bucket_minutes: int = 5,
    interpolation_method: str = "latest",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both STIB pipeline outputs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - live_snapshot_df
        - completed_snapshot_df
    """
    from core.stib_pipeline import run_stib_pipeline

    live_snapshot_df, completed_snapshot_df = run_stib_pipeline(
        token=token,
        gpkg_path=gpkg_path,
        lookback_minutes=lookback_minutes,
        bucket_minutes=bucket_minutes,
        interpolation_method=interpolation_method,
    )

    return live_snapshot_df.copy(), completed_snapshot_df.copy()


@st.cache_data(show_spinner=False, ttl=90)
def load_live_stib_segment_speeds(
    token: str,
    gpkg_path: str,
    lookback_minutes: int = 60,
    bucket_minutes: int = 5,
    interpolation_method: str = "latest",
) -> pd.DataFrame:
    """
    Load the current live STIB segment snapshot for map 1.

    Returns
    -------
    pd.DataFrame
        Columns:
        - snapshot_time
        - segment_id
        - avg_speed_kmh
        - sample_count
    """
    live_snapshot_df, _ = load_stib_pipeline_outputs(
        token=token,
        gpkg_path=gpkg_path,
        lookback_minutes=lookback_minutes,
        bucket_minutes=bucket_minutes,
        interpolation_method=interpolation_method,
    )

    return live_snapshot_df.copy()


@st.cache_data(show_spinner=False, ttl=90)
def load_completed_stib_snapshot(
    token: str,
    gpkg_path: str,
    lookback_minutes: int = 60,
    bucket_minutes: int = 5,
    interpolation_method: str = "latest",
) -> pd.DataFrame:
    """
    Load the completed current-time STIB snapshot.

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
    _, completed_snapshot_df = load_stib_pipeline_outputs(
        token=token,
        gpkg_path=gpkg_path,
        lookback_minutes=lookback_minutes,
        bucket_minutes=bucket_minutes,
        interpolation_method=interpolation_method,
    )

    return completed_snapshot_df.copy()


@st.cache_data(show_spinner=False, ttl=90)
def load_live_stib_segment_speed_lookup(
    token: str,
    gpkg_path: str,
    lookback_minutes: int = 60,
    bucket_minutes: int = 5,
    interpolation_method: str = "latest",
) -> dict[str, float]:
    """
    Build a segment_id -> live speed lookup for map 1.

    Returns
    -------
    dict[str, float]
        Mapping:
        - key   -> segment_id as string
        - value -> avg_speed_kmh
    """
    live_snapshot_df = load_live_stib_segment_speeds(
        token=token,
        gpkg_path=gpkg_path,
        lookback_minutes=lookback_minutes,
        bucket_minutes=bucket_minutes,
        interpolation_method=interpolation_method,
    )

    working_df = live_snapshot_df.dropna(subset=["avg_speed_kmh"]).copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str).str.strip()

    return dict(zip(working_df["segment_id"], working_df["avg_speed_kmh"]))
