"""Three side-by-side Leaflet maps (bus / estimated / Google speeds) kept in sync.

The page markup, styles and tooltips live in synced_maps.html; this module
fills in the data. Each GeoJSON feature must carry the *_color,
*_highlight_color and *_str properties produced by speed_layers.py.
"""
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name("synced_maps.html")


def build_three_map_html(geojson_obj, center_lat, center_lon) -> str:
    return (
        TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("__CENTER_LAT__", str(center_lat))
        .replace("__CENTER_LON__", str(center_lon))
        .replace("__GEOJSON__", json.dumps(geojson_obj))
    )
