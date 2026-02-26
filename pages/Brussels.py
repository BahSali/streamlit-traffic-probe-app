import streamlit as st
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium

from core.styles import inject_styles
from core.colors import get_speed_color, legend_html
from core.data_sources import fetch_stib_shapefile
from core.map_render import center_from_bounds
from core.menu import sidebar_dropdown

st.set_page_config(page_title="Brussels", layout="wide")
inject_styles()
sidebar_dropdown("Brussels")

# --- Brussels-specific sidebar settings ---
st.sidebar.markdown("### Brussels settings")
show_labels = st.sidebar.checkbox("Show labels", value=True)
demo_mode = st.sidebar.toggle("Demo color mode", value=True)

# ------
st.markdown("<h2 style='color:#009688;'>Brussels</h2>", unsafe_allow_html=True)
st.caption("STIB network visualisation. Tooltip changes before/after colorisation.")

# --- Secrets ---
if "STIB_TOKEN" not in st.secrets:
    st.error("Missing STIB_TOKEN in st.secrets. Add it to .streamlit/secrets.toml or Streamlit Cloud Secrets.")
    st.stop()
token = st.secrets["STIB_TOKEN"]

# --- Session state ---
if "brussels_colorized" not in st.session_state:
    st.session_state["brussels_colorized"] = False

# --- UI controls ---
col_a, col_b, col_c = st.columns([2, 3, 2])
with col_b:
    if st.button("Colorize network (demo)", use_container_width=True):
        st.session_state["brussels_colorized"] = True
    if st.button("Reset", use_container_width=True):
        st.session_state["brussels_colorized"] = False


@st.cache_data(show_spinner=False, ttl=3600)
def build_geojson_with_style(token_value: str, colorized: bool) -> str:
    gdf = fetch_stib_shapefile(token_value)

    # Ensure required tooltip fields exist
    for col in ["ligne", "variante"]:
        if col not in gdf.columns:
            gdf[col] = "N/A"

    if colorized:
        # Deterministic demo speeds: stable across reruns
        speeds = {i: float(3 + (i * 7) % 55) for i in range(len(gdf))}
        gdf["__speed"] = [speeds[i] for i in range(len(gdf))]
        gdf["__speed_str"] = gdf["__speed"].map(lambda v: f"{v:.1f}")
        gdf["__color"] = gdf["__speed"].apply(get_speed_color)
    else:
        gdf["__speed_str"] = ""
        gdf["__color"] = "black"

    return gdf.to_json()


with st.spinner("Loading STIB network geometry..."):
    geojson_str = build_geojson_with_style(token, st.session_state["brussels_colorized"])

# Map center (fast bounds-based)
gdf_center_source = fetch_stib_shapefile(token)
center = center_from_bounds(gdf_center_source)

# Tooltip fields depend on colorized mode
tooltip_fields = ["ligne", "variante"] + (["__speed_str"] if st.session_state["brussels_colorized"] else [])
tooltip_aliases = ["Line:", "Variant:"] + (["Speed:"] if st.session_state["brussels_colorized"] else [])

m = folium.Map(location=center, zoom_start=11, control_scale=True)

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
    name="STIB",
).add_to(m)

col_map, col_legend = st.columns([4, 1], vertical_alignment="top")
with col_map:
    st_folium(
        m,
        width=850,
        height=550,
        key="brussels_map",
        returned_objects=[],
    )
with col_legend:
    st.markdown(legend_html(), unsafe_allow_html=True)
