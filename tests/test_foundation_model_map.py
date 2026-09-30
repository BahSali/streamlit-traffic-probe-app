"""Optional foundation-model map on the Brussels page (core.config.SHOW_FOUNDATION_MODEL_MAP)."""
import json
import re

import pandas as pd
import pytest
import streamlit as st

from cities.brussels.foundation_model import PLACEHOLDER_SPEED_KMH
from cities.brussels.speed_layers import FOUNDATION_MODEL_MAP_PROPERTIES, MAP_PROPERTIES, finalize_map_columns
from cities.brussels.synced_maps import build_synced_maps_html, build_three_map_html
from core import config
from core.colors import NO_DATA_COLOR, speed_color_or
from tests.conftest import map_features
from tests.test_pages import run_brussels

GEOJSON = '{"type": "FeatureCollection", "features": []}'


def panel_titles(html: str) -> list[str]:
    return re.findall(r"^      <div>(.*)</div>$", html, re.M)


def map_divs(html: str) -> list[str]:
    return re.findall(r'<div id="(map\d+)" class="map"></div>', html)


def panels(html: str) -> list[dict]:
    return json.loads(re.search(r"const panels = (\[.*?\]);\n", html).group(1))


def foundation_name(html: str):
    return json.loads(re.search(r"const foundationModelName = (.*?);\n", html).group(1))


def test_three_maps_without_foundation_model():
    html = build_synced_maps_html(GEOJSON, 50.8, 4.35, estimate_title="Estimated Speeds")
    assert map_divs(html) == ["map1", "map2", "map3"]
    assert panel_titles(html) == ["Bus Speeds (STIB)", "Estimated Speeds", "Google API Speeds"]
    assert [p["colorKey"] for p in panels(html)] == ["bus_color", "est_color", "google_color"]
    assert html.count('<div class="titles">') == 1  # one row
    assert "grid-template-columns: 1fr 1fr 1fr;" in html and "height: 520px;" in html
    assert foundation_name(html) is None
    # The old entry point still gives the same page.
    assert build_three_map_html(GEOJSON, 50.8, 4.35, estimate_title="Estimated Speeds") == html


@pytest.mark.parametrize("name", ["TabPFN", "Chronos"])
def test_four_maps_in_a_2x2_grid_with_the_configured_name(name):
    html = build_synced_maps_html(GEOJSON, 50.8, 4.35, estimate_title="Estimated Speeds", foundation_model_name=name)
    assert map_divs(html) == ["map1", "map2", "map3", "map4"]
    # Bus | Estimated / Foundation model | Google
    assert panel_titles(html) == ["Bus Speeds (STIB)", "Estimated Speeds", f"{name} Estimate", "Google API Speeds"]
    assert [p["colorKey"] for p in panels(html)] == [
        "bus_color", "est_color", "foundation_model_color", "google_color",
    ]
    assert [p["tooltip"] for p in panels(html)] == ["bus", "estimated", "foundation", "google"]
    assert html.count('<div class="titles">') == 2  # two rows of two
    assert "grid-template-columns: 1fr 1fr;" in html and "height: 420px;" in html
    assert foundation_name(html) == name
    # Every tooltip carries the foundation-model line; only its own map highlights it.
    assert html.count("${foundationLine(p, false)}") == 3 and html.count("${foundationLine(p, true)}") == 1
    # All maps are synced with each other.
    assert "maps.forEach(function(map) {" in html and "map.sync(other);" in html
    if name != "TabPFN":
        assert "TabPFN" not in html


def test_foundation_model_name_is_escaped():
    html = build_synced_maps_html(GEOJSON, 50.8, 4.35, foundation_model_name="<b>X</b>")
    assert "<b>X</b>" not in html and "&lt;b&gt;X&lt;/b&gt; Estimate" in html


def test_foundation_model_columns_use_the_estimate_colour_scale():
    speeds = [5.0, 25.0, 55.0, None]
    gdf = pd.DataFrame({"est_speed": [40.0] * 4, "foundation_model_speed": speeds})
    result = finalize_map_columns(gdf)
    expected = [speed_color_or(s, missing_color=NO_DATA_COLOR) for s in speeds]
    assert result["foundation_model_color"].tolist() == expected
    assert result["foundation_model_highlight_color"].tolist() == expected
    assert result["foundation_model_speed_str"].tolist() == ["5.0 km/h", "25.0 km/h", "55.0 km/h", "N/A"]
    # Existing columns are unchanged.
    assert result["est_color"].tolist() == [speed_color_or(40.0, missing_color=NO_DATA_COLOR)] * 4
    assert not set(FOUNDATION_MODEL_MAP_PROPERTIES) & set(finalize_map_columns(gdf[["est_speed"]]).columns)


def run_page(offline, monkeypatch, show, name="TabPFN"):
    monkeypatch.setattr(config, "SHOW_FOUNDATION_MODEL_MAP", show)
    monkeypatch.setattr(config, "FOUNDATION_MODEL_NAME", name)
    st.cache_data.clear()
    before = {k: offline[k] for k in ["google_requests", "stib_fetches", "model_runs"]}
    at, features, table = run_brussels(offline)
    calls = {k: offline[k] - before[k] for k in before}
    return at, features, table, offline["html"][-1], calls


def test_brussels_page_with_and_without_foundation_model_map(offline, monkeypatch):
    _, off_features, off_table, off_html, off_calls = run_page(offline, monkeypatch, show=False)
    on_at, on_features, on_table, on_html, on_calls = run_page(offline, monkeypatch, show=True, name="Chronos")

    # Off: the three-map page, no foundation-model data or text.
    assert map_divs(off_html) == ["map1", "map2", "map3"]
    assert foundation_name(off_html) is None
    assert not [c for c in off_features.columns if c.startswith("foundation_model")]
    assert "Chronos" not in off_html and "TabPFN" not in off_html

    # On: four maps, Bus | Estimated / Chronos | Google.
    assert map_divs(on_html) == ["map1", "map2", "map3", "map4"]
    assert panel_titles(on_html)[2] == "Chronos Estimate"
    assert foundation_name(on_html) == "Chronos"
    assert (on_features["foundation_model_speed"] == PLACEHOLDER_SPEED_KMH).all()
    assert (on_features["foundation_model_speed_str"] == "25.0 km/h").all()
    assert (on_features["foundation_model_color"] == speed_color_or(PLACEHOLDER_SPEED_KMH, missing_color="")).all()

    # The existing maps' data, the table / CSV and the external calls are unchanged.
    pd.testing.assert_frame_equal(on_features[MAP_PROPERTIES[1:]], off_features[MAP_PROPERTIES[1:]])
    pd.testing.assert_frame_equal(on_table, off_table)
    assert on_calls == off_calls
    assert on_calls["stib_fetches"] == 1 and on_calls["model_runs"] == 1


def test_idle_page_with_foundation_model_map(offline, monkeypatch):
    from tests.conftest import app

    monkeypatch.setattr(config, "SHOW_FOUNDATION_MODEL_MAP", True)
    st.cache_data.clear()
    at = app("pages/Brussels.py").run()
    assert not at.exception
    html = offline["html"][-1]
    assert map_divs(html) == ["map1", "map2", "map3", "map4"]
    assert (map_features(html)["foundation_model_speed_str"] == "N/A").all()
