
import os
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from core.styles import inject_styles
from core.colors import get_speed_color, legend_html
from core.data_sources import load_csv
from core.pipelines import run_estimation_pipeline, load_results_dict
from core.menu import sidebar_dropdown

st.set_page_config(page_title="Ixelles-Etterbeek", layout="wide")
inject_styles()
sidebar_dropdown("Ixelles-Etterbeek")

# --- Ixelles settings ---
st.sidebar.markdown("### Ixelles settings")
force_refresh = st.sidebar.checkbox("Force refresh model", value=False)
show_raw = st.sidebar.checkbox("Show raw table", value=False)

# ------
st.markdown(
    "<h2 style='color:#009688;'>Ixelles-Etterbeek</h2>",
    unsafe_allow_html=True
)
st.caption("Interactive visualisation of real-time probe-derived bus speeds alongside estimation across road segments.")

DATA_PATH = "data/Brux_net.csv"
RESULTS_PATH = "results.csv"
CSV_SEP = ";"


@st.cache_data(show_spinner=False)
def prepare_segments(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str, str, list[float]]:
    df = df.copy()
    df["id"] = df["id"].astype(str)

    last_four_cols = df.columns[-4:]
    start_lat_col, start_lon_col, end_lat_col, end_lon_col = last_four_cols

    center = [df[start_lat_col].mean(), df[start_lon_col].mean()]
    return df, start_lat_col, start_lon_col, end_lat_col, end_lon_col, center


def build_map(
    df: pd.DataFrame,
    start_lat_col: str,
    start_lon_col: str,
    end_lat_col: str,
    end_lon_col: str,
    center: list[float],
    results_dict: dict,
) -> folium.Map:
    m = folium.Map(location=center, zoom_start=13, control_scale=True)

    for _, row in df.iterrows():
        segment_id = str(row["id"])
        street_name = f"{row[df.columns[1]]} - {row[df.columns[2]]}"

        tooltip_lines = [
            f"Segment ID: {segment_id}",
            f"Name: {street_name}",
        ]

        color = "black"
        if results_dict and segment_id in results_dict:
            r = results_dict[segment_id]
            speed = r.get("Speed", None)
            prediction = r.get("Prediction", None)

            try:
                speed_f = float(speed)
                pred_f = float(prediction)
                tooltip_lines.append(f"STIB Speed: {speed_f:.1f}")
                tooltip_lines.append(f"Estimation: {pred_f:.1f}")
                color = get_speed_color(pred_f)
            except Exception:
                tooltip_lines.append("STIB Speed: N/A")
                tooltip_lines.append("Estimation: N/A")
                color = "gray"

        folium.PolyLine(
            locations=[
                [row[start_lat_col], row[start_lon_col]],
                [row[end_lat_col], row[end_lon_col]],
            ],
            color=color,
            weight=4,
            tooltip="<br>".join(tooltip_lines),
        ).add_to(m)

    return m


# --- Load data ---
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()

df_raw = load_csv(DATA_PATH, sep=CSV_SEP)
df, s_lat, s_lon, e_lat, e_lon, center = prepare_segments(df_raw)

with st.expander("Show raw segment data"):
    st.dataframe(df)

# --- Session state ---
if "ixelles_results_dict" not in st.session_state:
    st.session_state["ixelles_results_dict"] = {}

if "ixelles_last_run_error" not in st.session_state:
    st.session_state["ixelles_last_run_error"] = None

# --- Actions ---
col_a, col_b, col_c = st.columns([2, 3, 2])
with col_b:
    run_clicked = st.button("Run Traffic Estimation", use_container_width=True)
    force_refresh = st.checkbox("Force refresh (ignore cached results)", value=False)

if run_clicked:
    st.session_state["ixelles_last_run_error"] = None
    try:
        with st.spinner("Fetching live bus speeds and running estimator..."):
            run_estimation_pipeline(results_path=RESULTS_PATH, sep=CSV_SEP, force=force_refresh)

        with st.spinner("Loading results..."):
            st.session_state["ixelles_results_dict"] = load_results_dict(results_path=RESULTS_PATH, sep=CSV_SEP)

        st.success("Done. Results loaded.")
    except Exception as ex:
        st.session_state["ixelles_last_run_error"] = str(ex)
        st.error(f"Pipeline failed: {ex}")

if st.session_state["ixelles_last_run_error"]:
    st.warning("Last run failed. You can retry or use Force refresh.")

results_dict = st.session_state["ixelles_results_dict"]

# --- Render map ---
m = build_map(df, s_lat, s_lon, e_lat, e_lon, center, results_dict)

col_map, col_legend = st.columns([4, 1], vertical_alignment="top")
with col_map:
    st_folium(m, width=850, height=550)
with col_legend:
    st.markdown(legend_html(), unsafe_allow_html=True)
