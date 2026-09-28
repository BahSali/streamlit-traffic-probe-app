"""Monthly Google Routes limit, using the existing usage worksheet. No real calls."""
import threading

import geopandas as gpd
import pytest

import core.google_routes.service as google
from core import config
from tests.conftest import FakeSheet, app

MONTH = google.get_current_month_key()


def single_segment_name() -> str:
    gdf = gpd.read_file("data/Brussels_map_6km.gpkg")
    names = gdf["start_name"].astype(str).str.strip() + " - " + gdf["end_name"].astype(str).str.strip()
    counts = names.value_counts()
    return counts[counts == 1].index[0]


def run(at, *, segment_names=(), bus_ids=()):
    at.multiselect(key="bru_seg_names").set_value(list(segment_names))
    at.multiselect(key="bru_bus_ids").set_value(list(bus_ids)).run()
    [button] = [b for b in at.button if b.label == "RUN"]
    at = button.click().run()
    assert not at.exception
    return at


def metrics(at) -> dict:
    return {m.label: m.value for m in at.metric}


@pytest.fixture
def limit_10(offline, monkeypatch):
    monkeypatch.setattr(config, "APPLY_DEMO_SPEED_CORRECTION", True)
    monkeypatch.setattr(config, "GOOGLE_ROUTES_MONTHLY_LIMIT", 10)
    offline["usage_sheet"].values.append([MONTH, 9])
    return offline


def test_limit_is_one_setting_shown_in_overview_and_diagnostics(limit_10):
    at = app("pages/Brussels.py").run()
    assert metrics(at)["Google used"] == "9"
    assert metrics(at)["Google left"] == "1"
    assert any("monthly limit: 10" in c.value for c in at.caption)


def test_9_of_10_allows_a_one_request_batch_then_10_of_10_blocks(limit_10):
    at = run(app("pages/Brussels.py").run(), segment_names=[single_segment_name()])
    assert limit_10["google_requests"] == 1
    assert limit_10["usage_sheet"].month_total(MONTH) == 10
    assert metrics(at)["Google used"] == "10" and metrics(at)["Google left"] == "0"

    stored_before = len(limit_10["observation_sheet"].values)
    at = run(at, segment_names=[], bus_ids=["12"])  # a different selection: not a cache reuse
    assert limit_10["google_requests"] == 1  # nothing sent
    assert limit_10["usage_sheet"].month_total(MONTH) == 10
    warning = " ".join(w.value for w in at.warning)
    assert "10 of 10" in warning and MONTH in warning and "No Google request was sent" in warning
    assert len(limit_10["observation_sheet"].values) == stored_before  # no observation row
    assert metrics(at)["Google left"] == "0"


def test_a_batch_crossing_the_limit_sends_nothing(limit_10):
    at = run(app("pages/Brussels.py").run(), bus_ids=["12"])  # needs several requests
    assert limit_10["google_requests"] == 0
    assert limit_10["usage_sheet"].month_total(MONTH) == 9  # the denied reservation is released
    warning = " ".join(w.value for w in at.warning)
    assert "9 of 10" in warning and "leaving 1" in warning
    assert metrics(at)["Google used"] == "9"


def test_failed_requests_count_and_cached_results_do_not(limit_10, monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_ROUTES_MONTHLY_LIMIT", 5000)
    monkeypatch.setattr(google, "send_google_route_request", lambda api_key, body: (None, "HTTP 500"))
    at = run(app("pages/Brussels.py").run(), bus_ids=["12"])
    planned = int(next(c.value for c in at.caption if "planned requests" in c.value)
                  .split("planned requests: ")[1].split(",")[0])
    assert planned > 1
    assert limit_10["usage_sheet"].month_total(MONTH) == 9 + planned

    [button] = [b for b in at.button if b.label == "RUN"]
    at = button.click().run()  # same selection within 90 s: reused, not sent
    assert limit_10["usage_sheet"].month_total(MONTH) == 9 + planned


def test_two_concurrent_batches_cannot_both_take_the_last_request():
    """Both sessions append their reservation before either reads the sheet."""
    sheet = FakeSheet()
    sheet.values.append([MONTH, 9])
    barrier = threading.Barrier(2)
    original_read = sheet.get_all_values

    def read_after_both_appended():
        barrier.wait(timeout=5)
        return original_read()

    sheet.get_all_values = read_after_both_appended
    results = []
    original_get = google.get_google_sheet
    google.get_google_sheet = lambda: sheet
    try:
        threads = [
            threading.Thread(target=lambda: results.append(google.reserve_google_requests(1, limit=10)))
            for _ in range(2)
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]
    finally:
        google.get_google_sheet = original_get

    assert sorted(r["allowed"] for r in results) == [False, True]
    assert sheet.month_total(MONTH) == 10
    assert all(kw.get("insert_data_option") == "INSERT_ROWS" for _, kw in sheet.calls)


def test_usage_is_the_sum_of_the_month_rows():
    values = [["month_key", "request_count"], ["2026-08", 4000], [MONTH, 7], [MONTH, 2], [MONTH, "0"], [MONTH, ""]]
    assert google._month_usage(values, MONTH) == 9
    assert google._month_usage(values, MONTH, before_row=4) == 7
