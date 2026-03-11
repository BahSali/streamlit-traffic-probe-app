import json
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.styles import inject_styles
from core.colors import get_speed_color, legend_html
from core.data_sources import load_gpkg
from core.nav_panel import render_left_panel
from core.ui.brussels_controls import brussels_left_controls


st.set_page_config(page_title="Brussels", layout="wide")
inject_styles()

settings_box, content_box = render_left_panel("Brussels")

MAP_PATH = "data/Brussels_map_6km.gpkg"


if "brussels_colorized" not in st.session_state:
    st.session_state["brussels_colorized"] = False

if "brussels_applied_segment_names" not in st.session_state:
    st.session_state["brussels_applied_segment_names"] = []

if "brussels_applied_bus_ids" not in st.session_state:
    st.session_state["brussels_applied_bus_ids"] = []

@st.cache_data(show_spinner=False)
def parse_bus_lines(value) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    # Supports values such as:
    # "71"
    # "13,88"
    # "71;95"
    # "[71, 95]"
    # "71 95"
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    return [token.strip() for token in tokens if token.strip()]


@st.cache_data(show_spinner=False)
def load_brussels_map(path: str = MAP_PATH):
    gdf = load_gpkg(path).copy()

    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf.reset_index(drop=True)

    # Use row index as the internal map key.
    gdf["map_fid"] = gdf.index.astype(int)

    # Make sure expected columns exist.
    for col in ["start_name", "end_name", "bus_lines"]:
        if col not in gdf.columns:
            gdf[col] = ""

    # Build labels directly from the GPKG.
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
        key=lambda x: (len(str(x)), str(x)),
    )

    return segment_options, bus_id_options


def get_selected_map_fids(
    selected_segment_names: list[str],
    selected_bus_ids: list[str],
) -> set[int]:
    gdf = load_brussels_map().copy()

    selected_segment_names = {
        str(x).strip() for x in (selected_segment_names or []) if str(x).strip()
    }
    selected_bus_ids = {
        str(x).strip() for x in (selected_bus_ids or []) if str(x).strip()
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


@st.cache_data(show_spinner=False, ttl=3600)
def prepare_three_map_geojson(
    colorized: bool,
    selected_segment_names: tuple[str, ...],
    selected_bus_ids: tuple[str, ...],
):
    gdf = load_brussels_map().copy()

    selected_map_fids = get_selected_map_fids(
        list(selected_segment_names),
        list(selected_bus_ids),
    )

    # Fake speeds for now.
    if colorized:
        gdf["bus_speed"] = [float(8 + (i * 5) % 48) for i in range(len(gdf))]
        gdf["est_speed"] = [float(12 + (i * 7) % 42) for i in range(len(gdf))]
    else:
        gdf["bus_speed"] = [None] * len(gdf)
        gdf["est_speed"] = [None] * len(gdf)

    google_speed_values = []
    google_color_values = []

    for _, row in gdf.iterrows():
        fid = int(row["map_fid"])

        if colorized and fid in selected_map_fids:
            google_speed = float(10 + (fid * 6) % 45)
            google_color = get_speed_color(google_speed)
        else:
            google_speed = None
            google_color = "#000000"

        google_speed_values.append(google_speed)
        google_color_values.append(google_color)

    gdf["google_speed"] = google_speed_values
    gdf["google_color"] = google_color_values

    def format_speed(value):
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.1f} km/h"

    gdf["bus_speed_str"] = gdf["bus_speed"].apply(format_speed)
    gdf["est_speed_str"] = gdf["est_speed"].apply(format_speed)
    gdf["google_speed_str"] = gdf["google_speed"].apply(format_speed)

    gdf["bus_color"] = gdf["bus_speed"].apply(
        lambda value: get_speed_color(value) if value is not None else "#222222"
    )
    gdf["est_color"] = gdf["est_speed"].apply(
        lambda value: get_speed_color(value) if value is not None else "#222222"
    )

    gdf["bus_highlight_color"] = gdf["bus_speed"].apply(
        lambda value: get_speed_color(value) if value is not None else "#222222"
    )
    gdf["est_highlight_color"] = gdf["est_speed"].apply(
        lambda value: get_speed_color(value) if value is not None else "#222222"
    )
    gdf["google_highlight_color"] = gdf["google_speed"].apply(
        lambda value: get_speed_color(value) if value is not None else "#000000"
    )

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    geojson = json.loads(gdf.to_json())
    selected_google_count = int(sum(fid in selected_map_fids for fid in gdf["map_fid"].tolist()))

    return {
        "geojson": geojson,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "selected_google_count": selected_google_count,
        "segment_count": len(gdf),
    }


def build_three_map_html(geojson_obj, center_lat, center_lon):
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
      font-size: 12px;
      padding: 6px 8px;
      line-height: 1.45;
    }}

    .metric-highlight {{
      display: inline-block;
      color: #ffffff;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 700;
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
      layer.bindTooltip(html, {{ sticky: true }});
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
      layer.bindTooltip(html, {{ sticky: true }});
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
      layer.bindTooltip(html, {{ sticky: true }});
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
    st.session_state["brussels_applied_segment_names"] = list(controls["filters"]["segment_names"])
    st.session_state["brussels_applied_bus_ids"] = list(controls["filters"]["bus_ids"])
    st.session_state["brussels_colorized"] = True

if controls["reset_clicked"]:
    st.session_state["brussels_colorized"] = False
    st.session_state["brussels_applied_segment_names"] = []
    st.session_state["brussels_applied_bus_ids"] = []
    st.session_state["bru_seg_names"] = []
    st.session_state["bru_bus_ids"] = []


with content_box:
    st.markdown("<h2 style='color:#009688;'>Brussels</h2>", unsafe_allow_html=True)
    st.caption("Three synced maps for bus-derived, model-derived, and Google-derived speed comparison.")

    with st.spinner("Loading Brussels map geometry..."):
        payload = prepare_three_map_geojson(
            st.session_state["brussels_colorized"],
            tuple(st.session_state["brussels_applied_segment_names"]),
            tuple(st.session_state["brussels_applied_bus_ids"]),
        )

    html = build_three_map_html(
        payload["geojson"],
        payload["center_lat"],
        payload["center_lon"],
    )

    col_map, col_legend = st.columns([5, 1], vertical_alignment="top")

    with col_map:
        components.html(html, height=640, scrolling=False)

    with col_legend:
        st.markdown(legend_html(), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Colorized", "Yes" if st.session_state["brussels_colorized"] else "No")
    col2.metric("Google-colored segments", payload["selected_google_count"])
    col3.metric("Map segments", payload["segment_count"])
