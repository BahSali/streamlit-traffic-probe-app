import streamlit as st
import folium
from streamlit_folium import st_folium

from core.styles import inject_styles
from core.colors import get_speed_color, legend_html
from core.data_sources import fetch_stib_shapefile
from core.map_render import center_from_bounds


st.set_page_config(page_title="Brussels", layout="wide")
inject_styles()

st.markdown(
    "<h2 style='color:#009688;'>Brussels</h2>",
    unsafe_allow_html=True
)
st.caption("STIB network visualisation. (Optional) demo colorisation by synthetic speeds.")

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
    run_clicked = st.button("Colorize network (demo)", use_container_width=True)
    if run_clicked:
        st.session_state["brussels_colorized"] = True

    reset_clicked = st.button("Reset", use_container_width=True)
    if reset_clicked:
        st.session_state["brussels_colorized"] = False

# --- Load STIB geometry (cached with TTL in core.data_sources) ---
with st.spinner("Loading STIB network geometry..."):
    gdf = fetch_stib_shapefile(token)

# --- Build speed mapping (demo) ---
speeds = None
if st.session_state["brussels_colorized"]:
    # Deterministic pseudo-random values for stable UI across reruns
    speeds = {i: float(3 + (i * 7) % 55) for i in range(len(gdf))}

# --- Map ---
center = center_from_bounds(gdf)
m = folium.Map(location=center, zoom_start=11, control_scale=True)

for idx, row in gdf.iterrows():
    if speeds is not None:
        color = get_speed_color(speeds[idx])
    else:
        color = "black"

    tooltip = (
        f"<b>Line:</b> {row.get('ligne', 'N/A')}<br>"
        f"<b>Variant:</b> {row.get('variante', 'N/A')}"
    )

    folium.GeoJson(
        row.geometry.__geo_interface__,
        tooltip=tooltip,
        style_function=lambda _x, color=color: {"color": color, "weight": 2.5},
    ).add_to(m)

col_map, col_legend = st.columns([4, 1], vertical_alignment="top")
with col_map:
    st_folium(m, width=850, height=550)
with col_legend:
    st.markdown(legend_html(), unsafe_allow_html=True)
