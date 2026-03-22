from __future__ import annotations

import pandas as pd
import geopandas as gpd
import requests
import streamlit as st
from shapely.geometry import shape


@st.cache_data(show_spinner=False)
def load_csv(path: str, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


@st.cache_data(show_spinner=False)
def load_gpkg(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_stib_shapefile(token: str) -> gpd.GeoDataFrame:
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
    from core.stib_live import fetch_live_segment_speeds

    segment_snapshot, _ = fetch_live_segment_speeds(
        token=token,
        gpkg_path=gpkg_path,
    )

    return segment_snapshot


@st.cache_data(show_spinner=False, ttl=90)
def load_live_stib_segment_speed_lookup(
    token: str,
    gpkg_path: str,
) -> dict[str, float]:
    from core.stib_live import build_segment_speed_lookup

    snapshot_df = load_live_stib_segment_speeds(
        token=token,
        gpkg_path=gpkg_path,
    )

    return build_segment_speed_lookup(snapshot_df)
