import streamlit as st
import folium
from streamlit_folium import st_folium

from core.styles import inject_styles
from core.colors import get_speed_color, legend_html
from core.data_sources import fetch_stib_shapefile
from core.map_render import center_from_bounds
from core.nav_panel import render_left_panel
from core.ui.brussels_controls import brussels_left_controls


st.set_page_config(page_title="Brussels", layout="wide")
inject_styles()

settings_box, content_box = render_left_panel("Brussels")

# --- Secrets ---
if "STIB_TOKEN" not in st.secrets:
    with content_box:
        st.error("Missing STIB_TOKEN in st.secrets. Add it to .streamlit/secrets.toml or Streamlit Cloud Secrets.")
    st.stop()
token = st.secrets["STIB_TOKEN"]

# --- Session state ---
if "brussels_colorized" not in st.session_state:
    st.session_state["brussels_colorized"] = False

# --- Left panel controls (mode + colorize button) ---
controls = brussels_left_controls(settings_box)

if controls["colorize_clicked"]:
    st.session_state["brussels_colorized"] = True

mode = controls["mode"]  # not used yet (next step: conditional settings)

@st.cache_data(show_spinner=False, ttl=3600)
def build_geojson_with_style(token_value: str, colorized: bool) -> str:
    gdf = fetch_stib_shapefile(token_value)

    for col in ["ligne", "variante"]:
        if col not in gdf.columns:
            gdf[col] = "N/A"

    if colorized:
        speeds = {i: float(3 + (i * 7) % 55) for i in range(len(gdf))}
        gdf["__speed"] = [speeds[i] for i in range(len(gdf))]
        gdf["__speed_str"] = gdf["__speed"].map(lambda v: f"{v:.1f}")
        gdf["__color"] = gdf["__speed"].apply(get_speed_color)
    else:
        gdf["__speed_str"] = ""
        gdf["__color"] = "black"

    return gdf.to_json()

with content_box:
    st.markdown("<h2 style='color:#009688;'>Brussels</h2>", unsafe_allow_html=True)
    st.caption("STIB network visualisation. Mode selection is ready; mode-specific controls will be added next.")

    with st.spinner("Loading STIB network geometry..."):
        geojson_str = build_geojson_with_style(token, st.session_state["brussels_colorized"])

    gdf_center_source = fetch_stib_shapefile(token)
    center = center_from_bounds(gdf_center_source)

    m = folium.Map(location=center, zoom_start=11, control_scale=True)

    folium.GeoJson(
        data=geojson_str,
        style_function=lambda feature: {
            "color": feature["properties"].get("__color", "black"),
            "weight": 2.5,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["ligne", "variante"] + (["__speed_str"] if st.session_state["brussels_colorized"] else []),
            aliases=["Line:", "Variant:"] + (["Speed:"] if st.session_state["brussels_colorized"] else []),
            sticky=True,
        ),
        name="STIB",
    ).add_to(m)

    col_map, col_legend = st.columns([4, 1], vertical_alignment="top")
    with col_map:
        st_folium(m, width=850, height=550, key="brussels_map", returned_objects=[])
    with col_legend:
        st.markdown(legend_html(), unsafe_allow_html=True)

    # ------------------------------Overview---------------------------------------------
    st.markdown("---")
    st.markdown("### Overview")

    total_features = len(fetch_stib_shapefile(token))
    colorized = st.session_state["brussels_colorized"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Mode", mode)
    col2.metric("Number of segments", 1366)
    col3.metric("Number of bus lines", 34)
