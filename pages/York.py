import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium

from core.styles import inject_styles
from core.colors import get_speed_color, legend_html
from core.data_sources import load_gpkg
from core.map_render import center_from_bounds
from core.pipelines import run_estimation_pipeline


st.set_page_config(page_title="York", layout="wide")
inject_styles()

st.markdown("<h2 style='color:#009688;'>York</h2>", unsafe_allow_html=True)
st.caption("Road segment visualisation with estimated speeds mapped to network geometry.")

# --- Paths (adjust if needed) ---
GPKG_PATH = "data/York_roads_within_3km.gpkg"
PROXY_PATH = "test_proxy_estimates_filtered.csv"
LINK_PATH = "test_link_speed_timeseries_15min_wide.csv"

# If your pipeline writes results elsewhere, adjust this
PIPELINE_RESULTS_PATH = "results.csv"
PIPELINE_SEP = ";"


# --- Session state ---
if "york_run_token" not in st.session_state:
    st.session_state["york_run_token"] = 0

if "york_last_run_error" not in st.session_state:
    st.session_state["york_last_run_error"] = None

if "york_colorized" not in st.session_state:
    st.session_state["york_colorized"] = False

if "york_rand_boost" not in st.session_state:
    st.session_state["york_rand_boost"] = None


def ensure_files_exist(paths: list[str]) -> tuple[bool, str]:
    for p in paths:
        if not os.path.exists(p):
            return False, p
    return True, ""


@st.cache_data(show_spinner=False)
def load_timeseries(proxy_path: str, link_path: str, run_token: int) -> tuple[pd.Timestamp, dict, dict]:
    """
    run_token is a cache buster to allow refresh after pipeline runs.
    """
    proxy_df = pd.read_csv(proxy_path)
    link_df = pd.read_csv(link_path)

    # Parse timestamps
    proxy_df.iloc[:, 0] = pd.to_datetime(proxy_df.iloc[:, 0], errors="coerce")

    link_time = pd.to_datetime(link_df.iloc[:, 0], errors="coerce", utc=True)
    link_time_fixed = (link_time + pd.Timedelta(hours=1)).dt.tz_convert(None)
    link_df.iloc[:, 0] = link_time_fixed

    latest_time = proxy_df.iloc[:, 0].max()

    proxy_row = proxy_df[proxy_df.iloc[:, 0] == latest_time]
    link_row = link_df[link_df.iloc[:, 0] == latest_time]

    if proxy_row.empty:
        raise ValueError("No matching timestamp found in proxy file after parsing.")
    if link_row.empty:
        raise ValueError("No matching timestamp found in link file after time correction.")

    proxy_data = proxy_row.iloc[0, 1:].to_dict()
    link_data = link_row.iloc[0, 1:].to_dict()

    return latest_time, proxy_data, link_data


@st.cache_data(show_spinner=False)
def prepare_york_geojson(
    gpkg_path: str,
    proxy_data: dict,
    link_data: dict,
    add_boost: bool,
    boost_seed: int,
    run_token: int,
) -> tuple[str, int, int]:
    """
    Returns:
      geojson_str, num_common_ids, num_segments
    """
    gdf = load_gpkg(gpkg_path)

    required_cols = ["NO", "FROMNODENO"]
    for c in required_cols:
        if c not in gdf.columns:
            raise KeyError(f"Missing required column in GPKG: {c}")

    # Build composite segment id
    gdf = gdf.copy()
    gdf["segment_id"] = gdf.apply(lambda r: f"{int(r['NO'])}-{int(r['FROMNODENO'])}", axis=1)

    # Assign data
    gdf["Cariad_speed"] = gdf["segment_id"].map(link_data)
    gdf["Estimated_speed"] = gdf["segment_id"].map(proxy_data)

    # Optional boost (deterministic, for demo/testing)
    if add_boost:
        import numpy as np

        rng = np.random.default_rng(boost_seed)
        boost = rng.uniform(0, 10, len(gdf))
        gdf["Estimated_speed"] = pd.to_numeric(gdf["Estimated_speed"], errors="coerce") + boost

    # Style column
    gdf["__color"] = gdf["Estimated_speed"].apply(get_speed_color)

    # Tooltip fields
    gdf["__estimated_str"] = gdf["Estimated_speed"].apply(
        lambda v: f"{float(v):.2f}" if pd.notna(v) else "N/A"
    )
    gdf["__cariad_str"] = gdf["Cariad_speed"].apply(
        lambda v: f"{float(v):.2f}" if pd.notna(v) else "N/A"
    )

    common_ids = set(proxy_data.keys()) & set(link_data.keys())

    return gdf.to_json(), len(common_ids), len(gdf)


# --- UI controls ---
col_a, col_b, col_c = st.columns([2, 3, 2])
with col_b:
    run_clicked = st.button("Run Traffic Estimation", use_container_width=True)
    force_refresh = st.checkbox("Force refresh (ignore cached results)", value=False)
    show_link_speed = st.checkbox("Show Cariad speed in tooltip", value=False)
    add_boost = st.checkbox("Add deterministic boost (demo)", value=False)

# --- Basic file checks ---
ok, missing = ensure_files_exist([GPKG_PATH, PROXY_PATH, LINK_PATH])
if not ok:
    st.error(f"File not found: {missing}")
    st.stop()

# --- Pipeline trigger ---
if run_clicked:
    st.session_state["york_last_run_error"] = None
    try:
        with st.spinner("Running pipeline..."):
            # This runs your existing scripts; adjust if York uses a different pipeline entrypoint.
            run_estimation_pipeline(
                results_path=PIPELINE_RESULTS_PATH,
                sep=PIPELINE_SEP,
                force=force_refresh,
            )

        # Bust caches for time series / geojson preparation
        st.session_state["york_run_token"] += 1
        st.session_state["york_colorized"] = True

        st.success("Pipeline finished. Refreshing map data...")
    except Exception as ex:
        st.session_state["york_last_run_error"] = str(ex)
        st.error(f"Pipeline failed: {ex}")

if st.session_state["york_last_run_error"]:
    st.warning("Last run failed. You can retry or enable Force refresh.")

# --- Load time series and build geojson ---
try:
    latest_time, proxy_data, link_data = load_timeseries(PROXY_PATH, LINK_PATH, st.session_state["york_run_token"])
except Exception as ex:
    st.error(f"Failed to load time series: {ex}")
    st.stop()

# Use a stable seed, but allow cache bust via run_token
boost_seed = 42

geojson_str, num_common, num_segments = prepare_york_geojson(
    gpkg_path=GPKG_PATH,
    proxy_data=proxy_data,
    link_data=link_data,
    add_boost=add_boost,
    boost_seed=boost_seed,
    run_token=st.session_state["york_run_token"],
)

st.info(f"Latest timestamp: {latest_time} | Common IDs: {num_common} | Segments: {num_segments}")

# --- Map ---
# We need a GeoDataFrame only for bounds-based centering (fast)
gdf_for_center = load_gpkg(GPKG_PATH)
center = center_from_bounds(gdf_for_center)

m = folium.Map(location=center, zoom_start=13, control_scale=True)

tooltip_fields = ["segment_id", "__estimated_str"] + (["__cariad_str"] if show_link_speed else [])
tooltip_aliases = ["Segment:", "Estimated speed:"] + (["Cariad speed:"] if show_link_speed else [])

folium.GeoJson(
    data=geojson_str,
    style_function=lambda feature: {
        "color": feature["properties"].get("__color", "black"),
        "weight": 2.5,
    },
    tooltip=GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=tooltip_aliases,
        sticky=True,
    ),
    name="York",
).add_to(m)

col_map, col_legend = st.columns([4, 1], vertical_alignment="top")
with col_map:
    st_folium(
        m,
        width=850,
        height=550,
        key="york_map",
        returned_objects=[],
    )
with col_legend:
    st.markdown(legend_html(), unsafe_allow_html=True)
