"""Synced Leaflet maps (bus / estimated / [foundation model] / Google speeds).

Three maps in one row, or, with a foundation model, four maps in a 2x2 grid:
Bus | Estimated, Foundation model | Google.

The page markup, styles and tooltips live in synced_maps.html; this module
fills in the data and the panels. Each GeoJSON feature must carry the *_color,
*_highlight_color and *_str properties produced by speed_layers.py (and the
foundation_model_* ones when a foundation model is shown).
"""
import html
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name("synced_maps.html")

# Height (px) of each map: three maps in one row, or four in two rows.
MAP_HEIGHT_ONE_ROW = 520
MAP_HEIGHT_TWO_ROWS = 420


def _script_json(value) -> str:
    # "<\/" keeps text such as "</script>" from closing the script tag.
    return json.dumps(value).replace("</", "<\\/")


def build_synced_maps_html(
    geojson_text: str,
    center_lat,
    center_lon,
    estimate_title: str = "Estimated Speeds (Model)",
    foundation_model_name: str | None = None,
) -> str:
    """Page with the synced maps; foundation_model_name=None gives the three-map row."""
    # (title, colour property, tooltip kind in synced_maps.html)
    panels = [
        ("Bus Speeds (STIB)", "bus_color", "bus"),
        (estimate_title, "est_color", "estimated"),
        ("Google API Speeds", "google_color", "google"),
    ]
    if foundation_model_name:
        panels.insert(2, (f"{foundation_model_name} Estimate", "foundation_model_color", "foundation"))
        rows = [panels[:2], panels[2:]]
        map_height = MAP_HEIGHT_TWO_ROWS
    else:
        rows = [panels]
        map_height = MAP_HEIGHT_ONE_ROW

    rows_html = []
    number = 0
    for row in rows:
        titles = "\n".join(f"      <div>{html.escape(title)}</div>" for title, _, _ in row)
        maps = []
        for _ in row:
            number += 1
            maps.append(f'      <div id="map{number}" class="map"></div>')
        rows_html.append(
            f'    <div class="titles">\n{titles}\n    </div>\n'
            f'    <div class="maps">\n' + "\n".join(maps) + "\n    </div>"
        )
    panel_config = [
        {"id": f"map{i}", "colorKey": color_key, "tooltip": tooltip}
        for i, (_, color_key, tooltip) in enumerate(panels, start=1)
    ]
    # The name is shown as HTML in the tooltips.
    foundation_label = html.escape(foundation_model_name) if foundation_model_name else None

    return (
        TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("__GRID_COLUMNS__", " ".join(["1fr"] * len(rows[0])))
        .replace("__MAP_HEIGHT__", str(map_height))
        .replace("__PANEL_ROWS__", "\n".join(rows_html))
        .replace("__PANELS__", _script_json(panel_config))
        .replace("__FOUNDATION_MODEL_NAME__", _script_json(foundation_label))
        .replace("__CENTER_LAT__", str(center_lat))
        .replace("__CENTER_LON__", str(center_lon))
        # "<\/" keeps text such as "</script>" in a street name from closing the script tag.
        .replace("__GEOJSON__", geojson_text.replace("</", "<\\/"))
    )


def build_three_map_html(
    geojson_text: str,
    center_lat,
    center_lon,
    estimate_title: str = "Estimated Speeds (Model)",
) -> str:
    """The three-map row (kept for existing callers)."""
    return build_synced_maps_html(geojson_text, center_lat, center_lon, estimate_title=estimate_title)
