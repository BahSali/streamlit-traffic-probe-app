"""Offline fakes for the page tests: no MobilityTwin, Google or Sheets calls."""
import json
import re
import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class FakeSheet:
    """In-memory stand-in for the Google Sheet holding the monthly request count."""

    def __init__(self):
        self.rows = []

    def get_all_records(self):
        return [dict(r) for r in self.rows]

    def append_row(self, row):
        self.rows.append({"month_key": row[0], "request_count": row[1]})

    def find(self, key):
        for i, r in enumerate(self.rows):
            if r["month_key"] == key:
                return mock.Mock(row=i + 2)

    def update(self, cell, values):
        self.rows[int(cell[1:]) - 2]["request_count"] = values[0][0]


def fake_google_request(api_key, body):
    legs = [
        {"distanceMeters": 300, "duration": f"{30 + i * 7 % 50}s"}
        for i in range(1 + len(body.get("intermediates", [])))
    ]
    return {"routes": [{"legs": legs}]}, None


def fake_stib_pipeline(token, gpkg_path, **_):
    import geopandas as gpd

    ids = gpd.read_file(gpkg_path)["id"].astype(str).tolist()
    t = pd.Timestamp("2026-01-01 08:00")
    live_speed = [None if int(i) % 3 == 0 else 10.0 + int(i) % 30 for i in ids]
    live = pd.DataFrame(
        {"snapshot_time": t, "segment_id": ids, "avg_speed_kmh": live_speed, "sample_count": 1}
    )
    completed = pd.DataFrame(
        {
            "snapshot_time": t,
            "segment_id": ids,
            "live_speed_kmh": live_speed,
            "interpolated_speed_kmh": [12.0 + int(i) % 7 for i in ids],
        }
    )
    completed["final_speed_kmh"] = completed["live_speed_kmh"].fillna(completed["interpolated_speed_kmh"])
    completed["interpolation"] = completed["live_speed_kmh"].isna()
    # The real pipeline sorts by string segment id.
    return live, completed.sort_values("segment_id").reset_index(drop=True)


def fake_model_inference(completed_snapshot_df, token, gpkg_path):
    ids = completed_snapshot_df["segment_id"].astype(str)
    prediction_df = pd.DataFrame({"segment_id": ids, "est_speed": [8.0 + int(i) % 5 for i in ids]})
    return prediction_df, {"estimation_mode": "fake", "model_loaded": True, "error_message": None}


@pytest.fixture
def offline(monkeypatch):
    """Patch every external service and record the maps HTML and downloads."""
    import streamlit as st
    import streamlit.components.v1 as components

    import cities.brussels.model as brussels_model
    import core.google_routes.service as google
    import core.stib_pipeline as stib_pipeline

    record = {"html": [], "downloads": [], "google_requests": 0, "stib_fetches": 0, "model_runs": 0}
    sheet = FakeSheet()

    def counted(key, fn):
        def wrapper(*args, **kwargs):
            record[key] += 1
            return fn(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(components, "html", lambda html, **_: record["html"].append(html))
    monkeypatch.setattr(
        st, "download_button", lambda label, data, **_: record["downloads"].append(data.decode()) and False
    )
    monkeypatch.setattr(google, "get_google_sheet", lambda: sheet)
    monkeypatch.setattr(google, "send_google_route_request", counted("google_requests", fake_google_request))
    monkeypatch.setattr(stib_pipeline, "run_stib_pipeline", counted("stib_fetches", fake_stib_pipeline))
    monkeypatch.setattr(brussels_model, "run_tmp_model_inference", counted("model_runs", fake_model_inference))
    # Cached results from earlier tests would hide the calls counted above.
    st.cache_data.clear()
    monkeypatch.setenv("MOBILITY_TWIN_TOKEN", "fake")
    return record


def app(path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / path), default_timeout=120)
    at.secrets["GOOGLE_MAPS_API_KEY"] = "fake"
    return at


def map_features(html: str) -> pd.DataFrame:
    geojson = json.loads(re.search(r"const data = (\{.*?\});\n", html, re.S).group(1))
    return pd.DataFrame([f["properties"] for f in geojson["features"]])
