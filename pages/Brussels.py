import streamlit as st
import pandas as pd
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
        st.error("Missing STIB_TOKEN in st.secrets.")
    st.stop()
token = st.secrets["STIB_TOKEN"]

# --- Session state ---
if "brussels_colorized" not in st.session_state:
    st.session_state["brussels_colorized"] = False


@st.cache_data(show_spinner=False, ttl=3600)
def load_stib_gdf(token_value: str):
    return fetch_stib_shapefile(token_value)


def unique_sorted_str(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def apply_or_filters(gdf, mode: str, filters: dict, *, segment_col: str, busid_col: str, street_col: str):
    """
    OR logic across selected filters. If no filter selected -> return gdf unchanged.
    """
    if gdf is None or len(gdf) == 0:
        return gdf

    mask = None

    if mode == "Segment":
        seg_sel = set(filters.get("segment_names", []) or [])
        bus_sel = set(filters.get("bus_ids", []) or [])

        if seg_sel and segment_col in gdf.columns:
            m1 = gdf[segment_col].astype(str).isin(seg_sel)
            mask = m1 if mask is None else (mask | m1)

        if bus_sel and busid_col in gdf.columns:
            m2 = gdf[busid_col].astype(str).isin(bus_sel)
            mask = m2 if mask is None else (mask | m2)

    elif mode == "Street":
        street_sel = set(filters.get("streets", []) or [])
        if street_sel and street_col in gdf.columns:
            m = gdf[street_col].astype(str).isin(street_sel)
            mask = m if mask is None else (mask | m)

    if mask is None:
        return gdf
    return gdf[mask].copy()


@st.cache_data(show_spinner=False, ttl=3600)
def build_geojson_with_style(token_value: str, colorized: bool, mode: str, filters: dict,
                            segment_col: str, busid_col: str, street_col: str) -> tuple[str, dict]:
    """
    Returns:
      geojson_str: styled geojson of filtered gdf
      summary: dict of basic stats for visualization below map
    """
    gdf = load_stib_gdf(token_value)

    # Ensure core fields exist to avoid tooltip errors
    for col in ["ligne", "variante"]:
        if col not in gdf.columns:
            gdf[col] = "N/A"

    gdf_filtered = apply_or_filters(
        gdf,
        mode,
        filters,
        segment_col=segment_col,
        busid_col=busid_col,
        street_col=street_col,
    )

    summary = {
        "mode": mode,
        "total_features": int(len(gdf)),
        "filtered_features": int(len(gdf_filtered)),
    }

    if colorized:
        speeds = {i: float(3 + (i * 7) % 55) for i in range(len(gdf_filtered))}
        gdf_filtered = gdf_filtered.reset_index(drop=True)
        gdf_filtered["__speed"] = [speeds[i] for i in range(len(gdf_filtered))]
        gdf_filtered["__speed_str"] = gdf_filtered["__speed"].map(lambda v: f"{v:.1f}")
        gdf_filtered["__color"] = gdf_filtered["__speed"].apply(get_speed_color)

        summary["avg_speed"] = float(gdf_filtered["__speed"].mean()) if len(gdf_filtered) else None
        summary["min_speed"] = float(gdf_filtered["__speed"].min()) if len(gdf_filtered) else None
        summary["max_speed"] = float(gdf_filtered["__speed"].max()) if len(gdf_filtered) else None
    else:
        gdf_filtered["__speed_str"] = ""
        gdf_filtered["__color"] = "black"
        summary["avg_speed"] = None
        summary["min_speed"] = None
        summary["max_speed"] = None

    return gdf_filtered.to_json(), summary


# --- Load base gdf once (for options + center) ---
gdf_base = load_stib_gdf(token)

# TODO: set these to your real column names (prototype)
SEGMENT_COL = "segment_name"
BUSID_COL = "bus_id"
STREET_COL = "street_name"

# If you don't know columns, temporarily uncomment this line:
# with content_box: st.write(list(gdf_base.columns))

segment_options = unique_sorted_str(gdf_base[SEGMENT_COL]) if SEGMENT_COL in gdf_base.columns else []
bus_id_options = unique_sorted_str(gdf_base[BUSID_COL]) if BUSID_COL in gdf_base.columns else []
street_options = unique_sorted_str(gdf_base[STREET_COL]) if STREET_COL in gdf_base.columns else []

# --- Left panel controls ---
controls = brussels_left_controls(
    settings_box,
    segment_options=segment_options,
    bus_id_options=bus_id_options,
    street_options=street_options,
)

mode = controls["mode"]
filters = controls["filters"]

if controls["colorize_clicked"]:
    st.session_state["brussels_colorized"] = True

# --- Build map data ---
with content_box:
    st.markdown("<h2 style='color:#009688;'>Brussels</h2>", unsafe_allow_html=True)
    st.caption("Map + mode-dependent summary below the map.")

    with st.spinner("Preparing map..."):
        geojson_str, summary = build_geojson_with_style(
            token,
            st.session_state["brussels_colorized"],
            mode,
            filters,
            SEGMENT_COL,
            BUSID_COL,
            STREET_COL,
        )

    center = center_from_bounds(gdf_base)
    m = folium.Map(location=center, zoom_start=11, control_scale=True)

    # Tooltip fields depend on colorized state only (prototype)
    tooltip_fields = ["ligne", "variante"] + (["__speed_str"] if st.session_state["brussels_colorized"] else [])
    tooltip_aliases = ["Line:", "Variant:"] + (["Speed:"] if st.session_state["brussels_colorized"] else [])

    folium.GeoJson(
        data=geojson_str,
        style_function=lambda feature: {
            "color": feature["properties"].get("__color", "black"),
            "weight": 2.5,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            sticky=True,
        ),
        name="STIB",
    ).add_to(m)

    col_map, col_legend = st.columns([4, 1], vertical_alignment="top")
    with col_map:
        st_folium(m, width=850, height=550, key="brussels_map", returned_objects=[])
    with col_legend:
        st.markdown(legend_html(), unsafe_allow_html=True)

    # ---------------- Visualization below the map ----------------
    st.markdown("---")
    st.markdown("### Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", summary["mode"])
    c2.metric("Total features", summary["total_features"])
    c3.metric("Filtered features", summary["filtered_features"])

    if summary["avg_speed"] is None:
        c4.metric("Avg speed", "N/A")
    else:
        c4.metric("Avg speed", f"{summary['avg_speed']:.1f}")

    if summary["avg_speed"] is not None:
        smin, savg, smax = summary["min_speed"], summary["avg_speed"], summary["max_speed"]
        st.write(f"Speed range: {smin:.1f} – {smax:.1f} (avg {savg:.1f})")

    # Show active filters (prototype)
    active = []
    if mode == "Segment":
        if filters["segment_names"]:
            active.append(f"Segments: {len(filters['segment_names'])}")
        if filters["bus_ids"]:
            active.append(f"Bus IDs: {len(filters['bus_ids'])}")
    elif mode == "Street":
        if filters["streets"]:
            active.append(f"Streets: {len(filters['streets'])}")

    st.write("Active filters (OR): " + (", ".join(active) if active else "None"))
