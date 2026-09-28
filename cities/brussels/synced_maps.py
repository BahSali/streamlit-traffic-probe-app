"""Three side-by-side Leaflet maps (bus / estimated / Google speeds) kept in sync.

The page markup, styles and tooltips live in synced_maps.html; this module
fills in the data. Each GeoJSON feature must carry the *_color,
*_highlight_color and *_str properties produced by speed_layers.py.
"""
import html
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name("synced_maps.html")


def build_three_map_html(
    geojson_text: str,
    center_lat,
    center_lon,
    estimate_title: str = "Estimated Speeds (Model)",
) -> str:
    return (
        TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("__ESTIMATE_TITLE__", html.escape(estimate_title))
        .replace("__CENTER_LAT__", str(center_lat))
        .replace("__CENTER_LON__", str(center_lon))
        # "<\/" keeps text such as "</script>" in a street name from closing the script tag.
        .replace("__GEOJSON__", geojson_text.replace("</", "<\\/"))
    )
