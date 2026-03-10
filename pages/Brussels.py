import json
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

# ---------------------- secrets ----------------------
if "STIB_TOKEN" not in st.secrets:
    with content_box:
        st.error("Missing STIB_TOKEN in st.secrets.")
    st.stop()

token = st.secrets["STIB_TOKEN"]

# ---------------------- state ----------------------
if "brussels_colorized" not in st.session_state:
    st.session_state["brussels_colorized"] = False

controls = brussels_left_controls(settings_box)
if controls["colorize_clicked"]:
    st.session_state["brussels_colorized"] = True

mode = controls["mode"]


# ---------------------- helpers ----------------------
@st.cache_data(show_spinner=False, ttl=3600)
def prepare_three_map_geojson(token_value: str, colorized: bool):
    gdf = fetch_stib_shapefile(token_value).copy()

    # ensure lat/lon
    try:
        if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass

    for col in ["ligne", "variante"]:
        if col not in gdf.columns:
            gdf[col] = "N/A"

    # fake speed layers for now
    if colorized:
        gdf["bus_speed"] = [float(8 + (i * 5) % 48) for i in range(len(gdf))]
        gdf["est_speed"] = [float(12 + (i * 7) % 42) for i in range(len(gdf))]
        gdf["google_speed"] = [float(10 + (i * 6) % 45) for i in range(len(gdf))]
    else:
        gdf["bus_speed"] = [None] * len(gdf)
        gdf["est_speed"] = [None] * len(gdf)
        gdf["google_speed"] = [None] * len(gdf)

    def fmt(v):
        if v is None:
            return ""
        return f"{float(v):.1f} km/h"

    gdf["bus_speed_str"] = gdf["bus_speed"].apply(fmt)
    gdf["est_speed_str"] = gdf["est_speed"].apply(fmt)
    gdf["google_speed_str"] = gdf["google_speed"].apply(fmt)

    gdf["bus_color"] = gdf["bus_speed"].apply(lambda x: get_speed_color(x) if x is not None else "#222222")
    gdf["est_color"] = gdf["est_speed"].apply(lambda x: get_speed_color(x) if x is not None else "#222222")
    gdf["google_color"] = gdf["google_speed"].apply(lambda x: get_speed_color(x) if x is not None else "#222222")

    # bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    geojson = json.loads(gdf.to_json())

    return {
        "geojson": geojson,
        "center_lat": center_lat,
        "center_lon": center_lon,
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
      height: 620px;
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
      height: 100%;
      min-height: 540px;
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
      <div>Bus speed</div>
      <div>Estimated speed</div>
      <div>Google proxy</div>
    </div>
    <div class="maps">
      <div>Observed Bus-Derived Speeds (STIB)</div>
      <div>Model-Derived Street Speed Estimates</div>
      <div>Google Routes API-Derived Speeds</div>
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

    const tileA = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }});

    const tileB = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }});

    const tileC = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }});

    tileA.addTo(map1);
    tileB.addTo(map2);
    tileC.addTo(map3);

    // full sync
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

    function estTooltip(feature, layer) {{
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
      onEachFeature: estTooltip
    }}).addTo(map2);

    const layer3 = L.geoJSON(data, {{
      style: styleFactory('google_color')
    }}).addTo(map3);

    // fit once
    try {{
      const bounds = layer1.getBounds();
      map1.fitBounds(bounds);
    }} catch (e) {{
      console.log(e);
    }}
  </script>
</body>
</html>
"""
    return html


# ---------------------- page ----------------------
with content_box:
    st.markdown("<h2 style='color:#009688;'>Brussels</h2>", unsafe_allow_html=True)
    st.caption("Three synced maps: bus speed, estimated speed, and Google proxy placeholder.")

    with st.spinner("Loading STIB network geometry..."):
        payload = prepare_three_map_geojson(token, st.session_state["brussels_colorized"])

    html = build_three_map_html(
        payload["geojson"],
        payload["center_lat"],
        payload["center_lon"],
    )

    col_map, col_legend = st.columns([5, 1], vertical_alignment="top")

    with col_map:
        components.html(html, height=660, scrolling=False)

    with col_legend:
        st.markdown(legend_html(), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Export")

    if st.button("Download results (prototype)"):
        st.info("Prototype only — download not implemented yet.")

    st.markdown("---")
    st.markdown("### Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mode", mode)
    col2.metric("Number of segments", 1366)
    col3.metric("Number of bus lines", 53)@st.cache_data(show_spinner=False, ttl=3600)
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
    st.caption("STIB network visualisation. Mode selection; filter controls; analysis; export")

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

    # ------------------------------Export and Overview---------------------------------------------
    st.markdown("---")
    st.markdown("### Export")

    if st.button("Download results (prototype)"):
        st.info("Prototype only — download not implemented yet.")

    st.markdown("---")
    st.markdown("### Overview")

    total_features = len(fetch_stib_shapefile(token))
    colorized = st.session_state["brussels_colorized"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Mode", mode)
    col2.metric("Number of segments", 1366)
    col3.metric("Number of bus lines", 53)

