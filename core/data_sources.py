from __future__ import annotations

import pandas as pd
import geopandas as gpd
import requests
import streamlit as st
from shapely.geometry import shape

from core.stib_live import (
    build_segment_speed_lookup,
    fetch_live_segment_speeds,
)


@st.cache_data(show_spinner=False)
def load_csv(path: str, sep: str = ";") -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(path, sep=sep)


@st.cache_data(show_spinner=False)
def load_gpkg(path: str) -> gpd.GeoDataFrame:
    """
    Load a GeoPackage file into a GeoDataFrame.
    """
    return gpd.read_file(path)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_stib_shapefile(token: str) -> gpd.GeoDataFrame:
    """
    Fetch the STIB shapefile data from the MobilityTwin API.

    The result is cached for one hour because the geometry layer is relatively
    stable compared to live speed data.
    """
    url = "https://api.mobilitytwin.brussels/stib/shapefile"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    data = response.json()

    geometries = [shape(feature["geometry"]) for feature in data["features"]]
    properties_df = pd.DataFrame(
        [feature["properties"] for feature in data["features"]]
    )

    return gpd.GeoDataFrame(properties_df, geometry=geometries, crs="EPSG:4326")


@st.cache_data(show_spinner=False, ttl=90)
def load_live_stib_segment_speeds(
    token: str,
    gpkg_path: str,
) -> pd.DataFrame:
    """
    Load live STIB speeds aggregated to segment level.

    This function uses the live MobilityTwin speed endpoint, maps point-level
    speeds to segment IDs using the provided GPKG file, and returns a
    segment-level snapshot.

    Parameters
    ----------
    token : str
        MobilityTwin API token.
    gpkg_path : str
        Path to the GPKG file containing the segment metadata.

    Returns
    -------
    pd.DataFrame
        Columns:
        - snapshot_time
        - segment_id
        - avg_speed_kmh
        - sample_count
    """
    return fetch_live_segment_speeds(
        token=token,
        gpkg_path=gpkg_path,
    )


@st.cache_data(show_spinner=False, ttl=90)
def load_live_stib_segment_speed_lookup(
    token: str,
    gpkg_path: str,
) -> dict[str, float]:
    """
    Load live STIB segment speeds as a lookup dictionary.

    This is the most convenient format for mapping speeds onto a GeoDataFrame
    used by the Streamlit map.

    Parameters
    ----------
    token : str
        MobilityTwin API token.
    gpkg_path : str
        Path to the GPKG file containing the segment metadata.

    Returns
    -------
    dict[str, float]
        Mapping:
        - key   -> segment_id as string
        - value -> avg_speed_kmh
    """
    snapshot_df = load_live_stib_segment_speeds(
        token=token,
        gpkg_path=gpkg_path,
    )

    return build_segment_speed_lookup(snapshot_df)
