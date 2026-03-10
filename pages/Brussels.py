import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.styles import inject_styles
from core.colors import get_speed_color, legend_html
from core.data_sources import fetch_stib_shapefile
from core.nav_panel import render_left_panel
from core.ui.brussels_controls import brussels_left_controls


st.set_page_config(page_title="Brussels", layout="wide")
inject_styles()

settings_box, content_box = render_left_panel("Brussels")

# ---------------------- Secrets ----------------------
if "STIB_TOKEN" not in st.secrets:
    with content_box:
        st.error("Missing STIB_TOKEN in st.secrets.")
    st.stop()

token = st.secrets["STIB_TOKEN"]

# ---------------------- Session state ----------------------
if "brussels_colorized" not in st.session_state:
    st.session_state["brussels_colorized"] = False


# ---------------------- Metadata helpers ----------------------
@st.cache_data(show_spinner=False)
def load_segments_metadata(path: str = "data/segments_metadata.csv") -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    # Build the display label for the segment filter.
    df["segment_name"] = (
        df["start_name"].fillna("").astype(str).str.strip()
        + " - "
        + df["end_name"].fillna("").astype(str).str.strip()
    )

    # Split bus_lines into a list.
    def split_bus_lines(value):
        if pd.isna(value):
            return []
        text = str(value).replace(";", ",")
        return [item.strip() for item in text.split(",") if item.strip()]

    df["bus_line_list"] = df["bus_lines"].apply(split_bus_lines)
    return df


@st.cache_data(show_spinner=False)
def get_filter_options():
    meta = load_segments_metadata()

    segment_options = sorted(
        [
            value
            for value in meta["segment_name"].dropna().astype(str).unique().tolist()
            if value.strip()
        ]
    )

    bus_id_options = sorted(
        {
            bus_id
            for bus_list in meta["bus_line_list"].tolist()
            for bus_id in bus_list
            if str(bus_id).strip()
        },
        key=lambda x: (len(str(x)), str(x)),
    )

    return segment_options, bus_id_options


def build_selected_segment_keys(
    selected_segment_names: list[str],
    selected_bus_ids: list[str],
) -> set[tuple[float, float, float, float]]:
    """
    Create a set of endpoint-based keys for metadata rows selected by the user.
    A row is selected if:
    - its segment_name is selected, OR
    - any of its bus_line_list values is selected
    """
    meta = load_segments_metadata().copy()

    selected_segment_names = set(selected_segment_names or [])
    selected_bus_ids = set(selected_bus_ids or [])

    def row_matches_bus_ids(bus_list):
        if not selected_bus_ids:
            return False
        return any(bus_id in selected_bus_ids for bus_id in bus_list)

    mask = pd.Series(False, index=meta.index)

    if selected_segment_names:
        mask = mask | meta["segment_name"].isin(selected_segment_names)

    if selected_bus_ids:
        mask = mask | meta["bus_line_list"].apply(row_matches_bus_ids)

    selected_meta = meta.loc[mask].copy()

    if selected_meta.empty:
        return set()

    selected_meta["segment_key"] = selected_meta.apply(
        lambda row: (
            round(float(row["start_lon"]), 6),
            round(float(row["start_lat"]), 6),
            round(float(row["end_lon"]), 6),
            round(float(row["end_lat"]), 6),
        ),
        axis=1,
    )

    return set(selected_meta["segment_key"].tolist())


# ---------------------- Data preparation ----------------------
@st.cache_data(show_spinner=False, ttl=3600)
def prepare_three_map_geojson(
    token_value: str,
    colorized: bool,
    selected_segment_names: tuple[str, ...],
    selected_bus_ids: tuple[str, ...],
):
    gdf = fetch_stib_shapefile(token_value).copy()

    # Make sure the geometry is in WGS84.
    try:
        if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass

    for col in ["ligne", "variante"]:
        if col not in gdf.columns:
            gdf[col] = "N/A"

    # Build a geometry-based key for each segment in the STIB GeoDataFrame.
    gdf["segment_key"] = gdf.geometry.apply(
        lambda geom: (
            round(float(geom.coords[0][0]), 6),
            round(float(geom.coords[0][1]), 6),
            round(float(geom.coords[-1][0]), 6),
            round(float(geom.coords[-1][1]), 6),
        )
        if geom is not None and hasattr(geom, "coords")
        else None
    )

    selected_segment_keys = build_selected_segment_keys(
        list(selected_segment_names),
        list(selected_bus_ids),
    )

    # Map 1 and Map 2 are colorized entirely when colorization is enabled.
    if colorized:
        gdf["bus_speed"] = [float(8 + (i * 5) % 48) for i in range(len(gdf))]
        gdf["est_speed"] = [float(12 + (i * 7) % 42) for i in range(len(gdf))]
    else:
        gdf["bus_speed"] = [None] * len(gdf)
        gdf["est_speed"] = [None] * len(gdf)

    # Map 3 is colorized only for the segments selected in the left filter.
    google_speed_values = []
    for i, row in gdf.iterrows():
        if colorized and row["segment_key"] in selected_segment_keys:
            google_speed_values.append(float(10 + (i * 6) % 45))
        else:
            google_speed_values.append(None)

    gdf["google_speed"] = google_speed_values

    def format_speed(value):
        if value is None or pd.isna(value):
            return ""
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
    gdf["google_color"] = gdf["google_speed"].apply(
        lambda value: get_speed_color(value) if value is not None else "transparent"
    )

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    geojson = json.loads(gdf.to_json())

    selected_google_count = int(gdf["google_speed"].notna().sum())

    return {
        "geojson": geojson,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "selected_google_count": selected_google_count,
    }


# ---------------------- HTML builder ----------------------
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
      font-size: 15px;
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
          <b>Line:</b> ${{p.ligne ?? 'N/A'}}<br>
          <b>Variant:</b> ${{p.variante ?? 'N/A'}}<br>
          <b>Bus speed:</b> ${{p.bus_speed_str ?? ''}}
        </div>
      `;
      layer.bindTooltip(html, {{ sticky: true }});
    }}

    function estimatedTooltip(feature, layer) {{
      const p = feature.properties || {{}};
      const html = `
        <div>
          <b>Line:</b> ${{p.ligne ?? 'N/A'}}<br>
          <b>Variant:</b> ${{p.variante ?? 'N/A'}}<br>
          <b>Estimated speed:</b> ${{p.est_speed_str ?? ''}}
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
      style: function(feature) {{
        const color = feature.properties.google_color || 'transparent';

        if (color === 'transparent') {{
          return {{
            color: 'transparent',
            weight: 0,
            opacity: 0
          }};
        }}

        return {{
          color: color,
          weight: 3,
          opacity: 0.95
        }};
      }}
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


# ---------------------- Controls ----------------------
segment_options, bus_id_options = get_filter_options()

controls = brussels_left_controls(
    settings_box,
    segment_options=segment_options,
    bus_id_options=bus_id_options,
)

if controls["colorize_clicked"]:
    st.session_state["brussels_colorized"] = True

if controls["reset_clicked"]:
    st.session_state["brussels_colorized"] = False


# ---------------------- Page ----------------------
with content_box:
    st.markdown("<h2 style='color:#009688;'>Brussels</h2>", unsafe_allow_html=True)
    st.caption("Three synced maps for bus-derived, model-derived, and Google-derived speed comparison.")

    with st.spinner("Loading STIB network geometry..."):
        payload = prepare_three_map_geojson(
            token,
            st.session_state["brussels_colorized"],
            tuple(controls["filters"]["segment_names"]),
            tuple(controls["filters"]["bus_ids"]),
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
    st.markdown("### Export")

    if st.button("Download results (prototype)"):
        st.info("Prototype only — download not implemented yet.")

    st.markdown("---")
    st.markdown("### Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Colorized", "Yes" if st.session_state["brussels_colorized"] else "No")
    col2.metric("Google-colored segments", payload["selected_google_count"])
    col3.metric("Available bus lines", len(bus_id_options))
