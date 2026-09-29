"""Private Google observation log and CSV exports. All values are synthetic."""
import re
from datetime import datetime

import pandas as pd
import pytest

import cities.brussels.session as session
import core.google_routes.service as google
from core.google_routes import observations
from scripts import export_google_observations as export_script
from tests.conftest import FakeSheet, app
from tests.test_google_usage import run

TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def stored(sheet) -> pd.DataFrame:
    return observations.read_observations(sheet)


def test_a_run_stores_one_batch_with_returned_values(offline):
    at = run(app("pages/Brussels.py").run(), bus_ids=["12"])
    df = stored(offline["observation_sheet"])
    assert df["batch_id"].nunique() == 1
    assert (df["segment_id"] != "").all()  # no blank marker row
    [timestamp] = df["batch_timestamp_utc"].unique()
    assert TIMESTAMP.match(timestamp)
    segments = df[df["segment_id"] != ""]
    assert len(segments) == at.session_state["brussels_google_diagnostics"]["selected_segment_count"]
    assert set(segments["distance_m"]) == {"300"}  # the fake legs: 300 m
    speeds = pd.to_numeric(segments["speed_kmh"])
    durations = pd.to_numeric(segments["duration_s"])
    assert (abs(speeds - 0.3 / (durations / 3600)) < 1e-6).all()


def test_reused_and_denied_batches_add_nothing(offline, monkeypatch):
    at = run(app("pages/Brussels.py").run(), bus_ids=["12"])
    rows = len(offline["observation_sheet"].values)
    [button] = [b for b in at.button if b.label == "RUN"]
    at = button.click().run()  # same selection: reused
    assert len(offline["observation_sheet"].values) == rows

    from core import config
    monkeypatch.setattr(config, "GOOGLE_ROUTES_MONTHLY_LIMIT", 0)
    run(at, bus_ids=["71"])  # denied by the limit
    assert len(offline["observation_sheet"].values) == rows


def test_a_batch_where_every_request_failed_still_gets_a_row(offline, monkeypatch):
    monkeypatch.setattr(google, "send_google_route_request", lambda api_key, body: (None, "HTTP 500"))
    run(app("pages/Brussels.py").run(), bus_ids=["12"])
    df = stored(offline["observation_sheet"])
    assert len(df) == 1 and df.iloc[0]["segment_id"] == ""  # the batch marker only
    exports = observations.build_wide_exports(df, ["1", "2"])
    assert len(exports["speed_kmh"]) == 1 and exports["speed_kmh"][["1", "2"]].isna().all().all()


def synthetic_batch(segment_values, when="2026-01-15T07:07:31.123456+00:00"):
    frame = pd.DataFrame(segment_values, columns=["segment_id", "distance_m", "duration_s", "speed_kmh"])
    return observations.new_batch(datetime.fromisoformat(when), frame)


def test_retries_are_idempotent_and_never_overwrite():
    sheet = FakeSheet(header=None)
    observations.ensure_header(sheet)
    batch = synthetic_batch([("1", 100, 20, 18.0), ("2", 50, None, None)])
    assert batch["batch_timestamp_utc"] == "2026-01-15T07:07:31.123Z"

    # The append succeeds but the response is lost: the retry must not duplicate it.
    real_append = sheet.append_rows
    calls = {"n": 0}

    def append_then_fail(rows, **kwargs):
        calls["n"] += 1
        real_append(rows, **kwargs)
        if calls["n"] == 1:
            raise ConnectionError("response lost")

    sheet.append_rows = append_then_fail
    observations.persist_batch(sheet, batch, backoff_seconds=0)
    observations.persist_batch(sheet, batch, backoff_seconds=0)  # a later retry
    df = stored(sheet)
    assert (df["batch_id"] == batch["batch_id"]).sum() == 2  # the 2 segments, once
    assert calls["n"] == 1


def unsaved_warning(at) -> str:
    return " ".join(e.value for e in at.error if "Google observations NOT saved" in e.value)


def test_a_failed_write_is_reported_kept_and_retried(offline, monkeypatch):
    sheet = offline["observation_sheet"]
    real_append = sheet.append_rows
    monkeypatch.setattr(observations.time, "sleep", lambda s: None)
    sheet.append_rows = lambda rows, **kw: (_ for _ in ()).throw(ConnectionError("offline"))
    at = run(app("pages/Brussels.py").run(), bus_ids=["12"])

    # RUN completed with Google data, and the failure is reported, naming the batch.
    assert offline["google_requests"] > 0
    assert any(m.label == "Google segments" and m.value != "0" for m in at.metric)
    [pending] = at.session_state["brussels_pending_observation_batches"]
    text = unsaved_warning(at)
    assert pending["batch_id"] in text and pending["batch_timestamp_utc"] in text
    assert "ConnectionError while writing to Google Sheets" in text and "lost if the session ends" in text
    assert stored(sheet).empty

    sheet.append_rows = real_append
    at.run()  # any rerun retries; the batch is stored once
    assert stored(sheet)["batch_id"].unique().tolist() == [pending["batch_id"]]
    assert at.session_state["brussels_pending_observation_batches"] == []
    assert unsaved_warning(at) == ""


def test_missing_configuration_keeps_the_batch_and_says_so(offline, monkeypatch):
    def not_configured():
        raise observations.ObservationStoreError(
            "[google_observations] spreadsheet_id is missing from the app's secrets"
        )

    monkeypatch.setattr(session, "_observation_worksheet", not_configured)
    at = run(app("pages/Brussels.py").run(), bus_ids=["12"])
    assert offline["google_requests"] > 0  # RUN still used Google
    assert len(at.session_state["brussels_pending_observation_batches"]) == 1
    assert "spreadsheet_id is missing" in unsaved_warning(at)


def test_the_usage_spreadsheet_is_refused_as_observation_target(monkeypatch):
    import streamlit as st

    fake_secrets = {
        "google_observations": {"spreadsheet_id": "same-id", "worksheet_name": "observations"},
        "sheets": {"spreadsheet_id": "same-id", "worksheet_name": "usage"},
        "gcp_service_account": {"type": "fake"},
    }
    monkeypatch.setattr(st, "secrets", fake_secrets)
    with pytest.raises(observations.ObservationStoreError, match="monthly usage spreadsheet"):
        observations.get_observation_worksheet()


def test_exports_are_aligned_wide_tables():
    sheet = FakeSheet(header=None)
    observations.ensure_header(sheet)
    later = synthetic_batch([("3", 120, 30, 14.4), ("1", 60, 12, 18.0)], when="2026-07-15T06:07:31+00:00")
    earlier = synthetic_batch([("2", 90, None, None)], when="2026-01-15T07:07:31+00:00")
    for batch in [later, earlier]:
        observations.persist_batch(sheet, batch)
    sheet.values.append([later["batch_id"], later["batch_timestamp_utc"], "3", "999", "999", "999"])  # stray duplicate

    segment_ids = ["1", "2", "3", "4"]
    exports = observations.build_wide_exports(stored(sheet), segment_ids)
    assert set(exports) == {"distance_m", "duration_s", "speed_kmh"}
    for frame in exports.values():
        assert list(frame.columns) == ["timestamp_utc", *segment_ids]
        assert list(frame["timestamp_utc"]) == ["2026-01-15T07:07:31.000Z", "2026-07-15T06:07:31.000Z"]
    assert exports["distance_m"].loc[0, "2"] == 90 and pd.isna(exports["speed_kmh"].loc[0, "2"])
    assert exports["distance_m"].loc[1, "3"] == 120  # the first stored value wins
    assert exports["duration_s"].loc[1, "1"] == 12 and exports["speed_kmh"].loc[1, "1"] == 18.0
    assert exports["speed_kmh"][["4"]].isna().all().all()


def test_export_script_writes_three_files_for_all_segments(tmp_path, monkeypatch):
    sheet = FakeSheet(header=None)
    observations.ensure_header(sheet)
    observations.persist_batch(sheet, synthetic_batch([("1", 100, 20, 18.0)]))
    paths = export_script.export(sheet, tmp_path)
    assert sorted(p.name for p in paths) == ["google_distance_m.csv", "google_duration_s.csv", "google_speed_kmh.csv"]
    headers = {p.read_text().splitlines()[0] for p in paths}
    assert len(headers) == 1  # identical columns in all three
    columns = headers.pop().split(",")
    assert columns[0] == "timestamp_utc" and len(columns) == 1 + 1366
    assert columns[1:] == sorted(columns[1:], key=int)


def test_no_marker_rows_and_older_marker_rows_still_read():
    """Rows saved in the old format (with a marker row) stay untouched and readable."""
    sheet = FakeSheet(header=None)
    observations.ensure_header(sheet)
    legacy = [
        ["a" * 32, "2026-01-15T07:07:31.000Z", "", "", "", ""],          # old marker row
        ["a" * 32, "2026-01-15T07:07:31.000Z", "1", 100, 20, 18],
        ["a" * 32, "2026-01-15T07:07:31.000Z", "2", 90, "", ""],
        ["b" * 32, "2026-02-15T07:07:31.000Z", "", "", "", ""],          # old batch, nothing returned
    ]
    sheet.values.extend(legacy)
    before = [list(map(str, row)) for row in sheet.values]

    new = synthetic_batch([("1", 60, 12, 18.0), ("3", 120, 30, 14.4)], when="2026-07-15T06:07:31+00:00")
    observations.persist_batch(sheet, new)
    observations.persist_batch(sheet, new)  # retry: the batch_id check still prevents duplicates

    after = sheet.get_all_values()
    assert after[: len(before)] == before  # nothing deleted or overwritten
    added = after[len(before):]
    assert [row[2] for row in added] == ["1", "3"]  # no blank marker row
    assert all(row[0] == new["batch_id"] for row in added)

    exports = observations.build_wide_exports(stored(sheet), ["1", "2", "3"])
    speed, distance = exports["speed_kmh"], exports["distance_m"]
    assert list(speed["timestamp_utc"]) == [
        "2026-01-15T07:07:31.000Z", "2026-02-15T07:07:31.000Z", "2026-07-15T06:07:31.000Z",
    ]
    assert speed.loc[0, "1"] == 18 and distance.loc[0, "2"] == 90 and pd.isna(speed.loc[0, "2"])
    assert speed.loc[1, ["1", "2", "3"]].isna().all()  # the old empty batch keeps its row
    assert speed.loc[2, "1"] == 18.0 and distance.loc[2, "3"] == 120 and pd.isna(distance.loc[2, "2"])


def test_a_new_batch_with_nothing_returned_is_one_row():
    sheet = FakeSheet(header=None)
    observations.ensure_header(sheet)
    empty = synthetic_batch([])
    observations.persist_batch(sheet, empty)
    observations.persist_batch(sheet, empty)
    assert sheet.get_all_values()[1:] == [[empty["batch_id"], empty["batch_timestamp_utc"], "", "", "", ""]]
