from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.colors import get_speed_color, legend_html
from core.data_sources import (
    load_completed_stib_snapshot,
    load_gpkg,
    load_live_stib_segment_speed_lookup,
)
from core.nav_panel import render_left_panel
from core.styles import inject_styles
from core.ui.brussels_controls import brussels_left_controls


st.set_page_config(page_title="Brussels", layout="wide")
inject_styles()

settings_box, content_box = render_left_panel("Brussels")

MAP_PATH = "data/Brussels_map_6km.gpkg"
STIB_SECRET_KEY = "MOBILITY_TWIN_TOKEN"


if "brussels_colorized" not in st.session_state:
    st.session_state["brussels_colorized"] = False

if "brussels_applied_segment_names" not in st.session_state:
    st.session_state["brussels_applied_segment_names"] = []

if "brussels_applied_bus_ids" not in st.session_state:
    st.session_state["brussels_applied_bus_ids"] = []

if "brussels_refresh_key" not in st.session_state:
    st.session_state["brussels_refresh_key"] = 0

@st.cache_data(show_spinner=False)
def parse_bus_lines(value) -> list[str]:
    """
    Parse a bus line string into a normalized list of line identifiers.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    tokens = re.findall(r"[A-Za-z0-9]+", text)
    return [token.strip() for token in tokens if token.strip()]


@st.cache_data(show_spinner=False)
def load_brussels_map(path: str = MAP_PATH):
    """
    Load the Brussels map geometry and prepare helper columns for UI and mapping.
    """
    gdf = load_gpkg(path).copy()

    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf.reset_index(drop=True)
    gdf["map_fid"] = gdf.index.astype(int)

    for column in ["start_name", "end_name", "bus_lines"]:
        if column not in gdf.columns:
            gdf[column] = ""

    gdf["segment_name"] = (
        gdf["start_name"].fillna("").astype(str).str.strip()
        + " - "
        + gdf["end_name"].fillna("").astype(str).str.strip()
    )
    gdf["segment_name"] = gdf["segment_name"].replace(" - ", "").fillna("N/A")

    gdf["bus_lines_display"] = gdf["bus_lines"].fillna("").astype(str)
    gdf["bus_line_list"] = gdf["bus_lines"].apply(parse_bus_lines)

    return gdf


@st.cache_data(show_spinner=False)
def get_filter_options():
    """
    Build filter options for the left control panel.
    """
    gdf = load_brussels_map()

    segment_options = sorted(
        [
            value
            for value in gdf["segment_name"].dropna().astype(str).unique().tolist()
            if value.strip()
        ]
    )

    bus_id_options = sorted(
        {
            bus_id
            for bus_list in gdf["bus_line_list"].tolist()
            for bus_id in bus_list
            if str(bus_id).strip()
        },
        key=lambda value: (len(str(value)), str(value)),
    )

    return segment_options, bus_id_options


def get_selected_map_fids(
    selected_segment_names: list[str],
    selected_bus_ids: list[str],
) -> set[int]:
    """
    Resolve the selected segment names and bus IDs to map feature IDs.
    """
    gdf = load_brussels_map().copy()

    selected_segment_names = {
        str(value).strip()
        for value in (selected_segment_names or [])
        if str(value).strip()
    }
    selected_bus_ids = {
        str(value).strip()
        for value in (selected_bus_ids or [])
        if str(value).strip()
    }

    mask = pd.Series(False, index=gdf.index)

    if selected_segment_names:
        mask = mask | gdf["segment_name"].isin(selected_segment_names)

    if selected_bus_ids:
        mask = mask | gdf["bus_line_list"].apply(
            lambda bus_list: any(bus_id in selected_bus_ids for bus_id in bus_list)
        )

    selected_rows = gdf.loc[mask].copy()

    if selected_rows.empty:
        return set()

    return set(selected_rows["map_fid"].astype(int).tolist())


def get_live_stib_token() -> str | None:
    """
    Read the MobilityTwin token from Streamlit secrets.
    """
    return st.secrets.get(STIB_SECRET_KEY)


def format_speed(value) -> str:
    """
    Format a numeric speed value for display.
    """
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.1f} km/h"


def build_dark_fallback_color(_: object = None) -> str:
    """
    Return the default fallback color for missing data.
    """
    return "#222222"


def build_google_color(value) -> str:
    """
    Return the display color for Google-derived speed values.
    """
    if value is None or pd.isna(value):
        return "#000000"

    return get_speed_color(float(value))

def convert_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to downloadable CSV bytes.
    """
    return df.to_csv(index=False).encode("utf-8")
    

def attach_live_stib_bus_speeds(gdf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Attach live STIB bus speeds to the Brussels map GeoDataFrame.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - The updated GeoDataFrame
        - A small diagnostics dictionary for on-screen debugging
    """
    result = gdf.copy()

    diagnostics = {
        "token_found": False,
        "map_has_id_column": "id" in result.columns,
        "lookup_size": 0,
        "common_segment_ids": 0,
        "matched_segments": 0,
        "error_message": None,
    }

    result["bus_speed"] = pd.NA

    token = get_live_stib_token()
    if not token:
        diagnostics["error_message"] = (
            "Missing MobilityTwin token in Streamlit secrets."
        )
        return result, diagnostics

    diagnostics["token_found"] = True

    if "id" not in result.columns:
        diagnostics["error_message"] = (
            "The Brussels map file does not contain an 'id' column."
        )
        return result, diagnostics

    try:
        speed_lookup = load_live_stib_segment_speed_lookup(
            token=token,
            gpkg_path=MAP_PATH,
        )
    except Exception as exc:
        diagnostics["error_message"] = f"Live STIB data could not be loaded: {exc}"
        return result, diagnostics

    diagnostics["lookup_size"] = len(speed_lookup)

    result["segment_id_str"] = result["id"].astype(str)
    map_segment_ids = set(result["segment_id_str"].tolist())
    lookup_segment_ids = set(speed_lookup.keys())

    diagnostics["common_segment_ids"] = len(map_segment_ids.intersection(lookup_segment_ids))

    result["bus_speed"] = result["segment_id_str"].map(speed_lookup)
    diagnostics["matched_segments"] = int(result["bus_speed"].notna().sum())

    return result, diagnostics


@st.cache_data(show_spinner=False, ttl=90)
def prepare_three_map_geojson(
    colorized: bool,
    selected_segment_names: tuple[str, ...],
    selected_bus_ids: tuple[str, ...],
    refresh_key: int,
):
    """
    Build the GeoJSON payload used by the three synchronized maps.

    Map 1:
        Live STIB bus-derived speeds
    Map 2:
        Placeholder model-derived street speed estimates
    Map 3:
        Placeholder Google Routes API-derived speed proxy
    """
    gdf = load_brussels_map().copy()

    selected_map_fids = get_selected_map_fids(
        list(selected_segment_names),
        list(selected_bus_ids),
    )

    if colorized:
        gdf, diagnostics = attach_live_stib_bus_speeds(gdf)
        gdf["est_speed"] = [float(12 + (index * 7) % 42) for index in range(len(gdf))]
    else:
        diagnostics = {
            "token_found": False,
            "map_has_id_column": "id" in gdf.columns,
            "lookup_size": 0,
            "common_segment_ids": 0,
            "matched_segments": 0,
            "error_message": None,
        }
        gdf["bus_speed"] = pd.NA
        gdf["est_speed"] = [None] * len(gdf)

    google_speed_values = []
    google_color_values = []

    for _, row in gdf.iterrows():
        map_fid = int(row["map_fid"])

        if colorized and map_fid in selected_map_fids:
            google_speed = float(10 + (map_fid * 6) % 45)
            google_color = get_speed_color(google_speed)
        else:
            google_speed = None
            google_color = "#000000"

        google_speed_values.append(google_speed)
        google_color_values.append(google_color)

    gdf["google_speed"] = google_speed_values
    gdf["google_color"] = google_color_values

    gdf["bus_speed_str"] = gdf["bus_speed"].apply(format_speed)
    gdf["est_speed_str"] = gdf["est_speed"].apply(format_speed)
    gdf["google_speed_str"] = gdf["google_speed"].apply(format_speed)

    gdf["bus_color"] = gdf["bus_speed"].apply(
        lambda value: get_speed_color(float(value))
        if value is not None and not pd.isna(value)
        else build_dark_fallback_color()
    )
    gdf["est_color"] = gdf["est_speed"].apply(
        lambda value: get_speed_color(float(value))
        if value is not None and not pd.isna(value)
        else build_dark_fallback_color()
    )

    gdf["bus_highlight_color"] = gdf["bus_speed"].apply(
        lambda value: get_speed_color(float(value))
        if value is not None and not pd.isna(value)
        else build_dark_fallback_color()
    )
    gdf["est_highlight_color"] = gdf["est_speed"].apply(
        lambda value: get_speed_color(float(value))
        if value is not None and not pd.isna(value)
        else build_dark_fallback_color()
    )
    gdf["google_highlight_color"] = gdf["google_speed"].apply(build_google_color)

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    geojson = json.loads(gdf.to_json())
    selected_google_count = int(
        sum(map_fid in selected_map_fids for map_fid in gdf["map_fid"].tolist())
    )
    live_bus_count = int(gdf["bus_speed"].notna().sum())

    return {
        "geojson": geojson,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "selected_google_count": selected_google_count,
        "segment_count": len(gdf),
        "live_bus_count": live_bus_count,
        "diagnostics": diagnostics,
    }


def build_three_map_html(geojson_obj, center_lat, center_lon):
    """
    Build the HTML for the three synchronized Leaflet maps.
    """
    geojson_str = json.dumps(geojson_obj)

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
  />
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      font-family: Arial, sans-serif;
      background: #ffffff;
    }}

    .wrapper {{
      width: 100%;
      height: 600px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .titles {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      font-weight: 700;
      color: #1f2937;
    }}

    .titles div {{
      text-align: center;
      padding: 8px 6px;
      background: #f3f4f6;
      border-radius: 8px;
      border: 1px solid #e5e7eb;
      line-height: 1.25;
      font-size: 13px;
    }}

    .maps {{
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      min-height: 0;
    }}

    .map {{
      width: 100%;
      height: 520px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      overflow: hidden;
    }}

    .leaflet-tooltip {{
      padding: 0;
      margin: 0;
      border-radius: 6px;
    }}

    .leaflet-tooltip-content {{
      font-size: 12px;
      line-height: 1.35;
      margin: 6px 8px;
      min-width: 180px;
      max-width: 240px;
      white-space: normal;
      word-break: normal;
      overflow-wrap: break-word;
    }}

    .metric-highlight {{
      display: inline-block;
      color: #ffffff;
      padding: 1px 5px;
      border-radius: 4px;
      font-weight: 700;
      margin-left: 3px;
    }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="titles">
      <div>Observed Bus-Derived Speeds (STIB)</div>
      <div>Model-Derived Street Speed Estimates</div>
      <div>Google Routes API-Derived Speed Proxy</div>
    </div>
    <div class="maps">
      <div id="map1" class="map"></div>
      <div id="map2" class="map"></div>
      <div id="map3" class="map"></div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/leaflet.sync@0.2.4/L.Map.Sync.min.js"></script>

  <script>
    const data = {geojson_str};

    const center = [{center_lat}, {center_lon}];
    const zoom = 12;

    const map1 = L.map('map1', {{ zoomControl: true }}).setView(center, zoom);
    const map2 = L.map('map2', {{ zoomControl: true }}).setView(center, zoom);
    const map3 = L.map('map3', {{ zoomControl: true }}).setView(center, zoom);

    const tile1 = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }});

    const tile2 = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }});

    const tile3 = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }});

    tile1.addTo(map1);
    tile2.addTo(map2);
    tile3.addTo(map3);

    map1.sync(map2);
    map1.sync(map3);
    map2.sync(map1);
    map2.sync(map3);
    map3.sync(map1);
    map3.sync(map2);

    function styleFactory(colorKey) {{
      return function(feature) {{
        return {{
          color: feature.properties[colorKey] || '#222222',
          weight: 3,
          opacity: 0.95
        }};
      }};
    }}

    function busTooltip(feature, layer) {{
      const p = feature.properties || {{}};

      const html = `
        <div>
          <b>Segment:</b> ${{p.segment_name ?? 'N/A'}}<br>
          <b>Bus lines:</b> ${{p.bus_lines_display ?? ''}}<br>
          <b>Bus speed:</b>
          <span class="metric-highlight" style="background:${{p.bus_highlight_color ?? '#222222'}};">
            ${{p.bus_speed_str ?? 'N/A'}}
          </span><br>
          <b>Estimated speed:</b> ${{p.est_speed_str ?? 'N/A'}}
        </div>
      `;
      layer.bindTooltip(html, {{
        sticky: true,
        direction: 'top',
        offset: [0, -8],
        opacity: 0.95
      }});
    }}

    function estimatedTooltip(feature, layer) {{
      const p = feature.properties || {{}};

      const html = `
        <div>
          <b>Segment:</b> ${{p.segment_name ?? 'N/A'}}<br>
          <b>Bus lines:</b> ${{p.bus_lines_display ?? ''}}<br>
          <b>Bus speed:</b> ${{p.bus_speed_str ?? 'N/A'}}<br>
          <b>Estimated speed:</b>
          <span class="metric-highlight" style="background:${{p.est_highlight_color ?? '#222222'}};">
            ${{p.est_speed_str ?? 'N/A'}}
          </span>
        </div>
      `;
      layer.bindTooltip(html, {{
        sticky: true,
        direction: 'top',
        offset: [0, -8],
        opacity: 0.95
      }});
    }}

    function googleTooltip(feature, layer) {{
      const p = feature.properties || {{}};

      const html = `
        <div>
          <b>Segment:</b> ${{p.segment_name ?? 'N/A'}}<br>
          <b>Bus lines:</b> ${{p.bus_lines_display ?? ''}}<br>
          <b>Bus speed:</b> ${{p.bus_speed_str ?? 'N/A'}}<br>
          <b>Estimated speed:</b> ${{p.est_speed_str ?? 'N/A'}}<br>
          <b>Google-reported speed:</b>
          <span class="metric-highlight" style="background:${{p.google_highlight_color ?? '#000000'}};">
            ${{p.google_speed_str ?? 'N/A'}}
          </span>
        </div>
      `;
      layer.bindTooltip(html, {{
        sticky: true,
        direction: 'top',
        offset: [0, -8],
        opacity: 0.95
      }});
    }}

    const layer1 = L.geoJSON(data, {{
      style: styleFactory('bus_color'),
      onEachFeature: busTooltip
    }}).addTo(map1);

    const layer2 = L.geoJSON(data, {{
      style: styleFactory('est_color'),
      onEachFeature: estimatedTooltip
    }}).addTo(map2);

    const layer3 = L.geoJSON(data, {{
      style: styleFactory('google_color'),
      onEachFeature: googleTooltip
    }}).addTo(map3);

    try {{
      const bounds = layer1.getBounds();
      map1.fitBounds(bounds);
    }} catch (error) {{
      console.log(error);
    }}
  </script>
</body>
</html>
"""
    return html


segment_options, bus_id_options = get_filter_options()

controls = brussels_left_controls(
    settings_box,
    segment_options=segment_options,
    bus_id_options=bus_id_options,
    applied_segment_names=st.session_state["brussels_applied_segment_names"],
    applied_bus_ids=st.session_state["brussels_applied_bus_ids"],
)

if controls["colorize_clicked"]:
    st.session_state["brussels_applied_segment_names"] = list(
        controls["filters"]["segment_names"]
    )
    st.session_state["brussels_applied_bus_ids"] = list(
        controls["filters"]["bus_ids"]
    )
    st.session_state["brussels_colorized"] = True
    st.session_state["brussels_refresh_key"] += 1
    st.rerun()

if controls["reset_clicked"]:
    st.session_state["brussels_colorized"] = False
    st.session_state["brussels_applied_segment_names"] = []
    st.session_state["brussels_applied_bus_ids"] = []
    st.session_state["brussels_refresh_key"] += 1
    st.rerun()


with content_box:
    st.markdown("<h2 style='color:#009688;'>Brussels</h2>", unsafe_allow_html=True)
    st.caption(
        "Three synced maps for bus-derived, model-derived, and Google-derived speed comparison."
    )

    with st.spinner("Loading Brussels map geometry..."):
        payload = prepare_three_map_geojson(
            st.session_state["brussels_colorized"],
            tuple(st.session_state["brussels_applied_segment_names"]),
            tuple(st.session_state["brussels_applied_bus_ids"]),
            st.session_state["brussels_refresh_key"],
        )

    diagnostics = payload["diagnostics"]
    
    completed_snapshot_df = pd.DataFrame()
    if st.session_state["brussels_colorized"]:
        token = get_live_stib_token()

        if token:
            try:
                completed_snapshot_df = load_completed_stib_snapshot(
                    token=token,
                    gpkg_path=MAP_PATH,
                    lookback_minutes=60,
                    bucket_minutes=5,
                    interpolation_method="latest",
                )
            except Exception as exc:
                st.warning(f"Completed STIB snapshot could not be prepared: {exc}")
                
    if diagnostics["error_message"]:
        st.warning(diagnostics["error_message"])

    st.caption(
        "Live STIB diagnostics — "
        f"token: {'yes' if diagnostics['token_found'] else 'no'}, "
        f"map has id: {'yes' if diagnostics['map_has_id_column'] else 'no'}, "
        f"lookup size: {diagnostics['lookup_size']}, "
        f"common segment ids: {diagnostics['common_segment_ids']}, "
        f"matched segments: {diagnostics['matched_segments']}"
    )

    html = build_three_map_html(
        payload["geojson"],
        payload["center_lat"],
        payload["center_lon"],
    )

    col_map, col_legend = st.columns([10, 1], vertical_alignment="top")

    with col_map:
        components.html(html, height=640, scrolling=False)
    
    if (
        st.session_state["brussels_colorized"]
        and not completed_snapshot_df.empty
    ):
        st.download_button(
            label="Download completed STIB snapshot",
            data=convert_dataframe_to_csv_bytes(completed_snapshot_df),
            file_name="completed_stib_snapshot.csv",
            mime="text/csv",
        )
        
    with col_legend:
        st.markdown(legend_html(), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Colorized", "Yes" if st.session_state["brussels_colorized"] else "No")
    col2.metric("Google-colored segments", payload["selected_google_count"])
    col3.metric("Map segments", payload["segment_count"])
    col4.metric("Live STIB segments", payload["live_bus_count"])
