import pandas as pd
import geopandas as gpd
import streamlit as st
import requests
from shapely.geometry import shape

@st.cache_data(show_spinner=False)
def load_csv(path: str, sep=";"):
    return pd.read_csv(path, sep=sep)

@st.cache_data(show_spinner=False)
def load_gpkg(path: str):
    return gpd.read_file(path)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_stib_shapefile(token: str):
    url = "https://api.mobilitytwin.brussels/stib/shapefile"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    geoms = [shape(f["geometry"]) for f in data["features"]]
    df = pd.DataFrame([f["properties"] for f in data["features"]])
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
