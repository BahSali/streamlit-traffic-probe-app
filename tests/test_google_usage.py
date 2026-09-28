"""Strict monthly Google Routes limit. Fake GCS and Google: no paid or network calls."""
import multiprocessing
import random
import threading

import geopandas as gpd
import pytest

import core.google_routes.service as google
from core import config
from core.google_routes.usage_store import GcsUsageStore, UsageStoreError, next_month_start
from tests.conftest import app
from tests.fake_gcs import FileBucket, MemoryBucket

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


def warnings(at) -> str:
    return " ".join(w.value for w in at.warning)


@pytest.fixture
def limit_10(offline, monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_ROUTES_MONTHLY_LIMIT", 10)
    offline["usage_sheet"].values.append([MONTH, 9])  # the legacy sheet already says 9 this month
    return offline


def used(record) -> int:
    return record["usage_store"].used(MONTH)


def test_limit_setting_and_legacy_count_show_in_overview(limit_10):
    at = app("pages/Brussels.py").run()
    assert metrics(at)["Google used"] == "9" and metrics(at)["Google left"] == "1"
    assert any("monthly limit: 10" in c.value for c in at.caption)


def test_9_of_10_then_10_of_10(limit_10):
    at = run(app("pages/Brussels.py").run(), segment_names=[single_segment_name()])
    assert limit_10["google_requests"] == 1
    assert used(limit_10) == 10
    assert metrics(at)["Google used"] == "10" and metrics(at)["Google left"] == "0"

    at = run(at, segment_names=[], bus_ids=["12"])  # a different selection: not a reuse
    assert limit_10["google_requests"] == 1  # nothing more sent
    assert used(limit_10) == 10
    text = warnings(at)
    assert "10 of 10" in text and MONTH in text and next_month_start(MONTH) in text
    assert "No Google request was sent" in text


def test_a_batch_crossing_the_limit_sends_nothing(limit_10):
    at = run(app("pages/Brussels.py").run(), bus_ids=["12"])
    assert limit_10["google_requests"] == 0
    assert used(limit_10) == 9
    assert "9 of 10" in warnings(at) and "leaving 1" in warnings(at)
    assert metrics(at)["Google used"] == "9"


def test_failed_requests_count_and_reused_results_do_not(limit_10, monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_ROUTES_MONTHLY_LIMIT", 5000)
    monkeypatch.setattr(google, "send_google_route_request", lambda api_key, body: (None, "HTTP 500"))
    at = run(app("pages/Brussels.py").run(), bus_ids=["12"])
    planned = int(next(c.value for c in at.caption if "planned requests" in c.value)
                  .split("planned requests: ")[1].split(",")[0])
    assert planned > 1 and used(limit_10) == 9 + planned

    [button] = [b for b in at.button if b.label == "RUN"]
    at = button.click().run()  # same selection within 90 s: reused, not sent, not counted
    assert used(limit_10) == 9 + planned


def test_unavailable_store_sends_nothing_and_says_why(offline, monkeypatch):
    def broken():
        raise UsageStoreError("the service account has no access to the usage bucket (HTTP 403)")

    monkeypatch.setattr(google, "get_usage_store", broken)
    at = app("pages/Brussels.py").run()
    assert metrics(at)["Google used"] == "N/A" and metrics(at)["Google left"] == "N/A"
    at = run(at, bus_ids=["12"])
    assert offline["google_requests"] == 0
    text = warnings(at)
    assert "HTTP 403" in text and "No Google request was sent" in text and MONTH in text and "5000" in text


def test_partial_batch_counts_only_started_requests(offline, monkeypatch):
    real_body = google.build_google_request_body
    seen = {"n": 0}
    lock = threading.Lock()

    def body_fails_once(group_df):
        with lock:
            seen["n"] += 1
            fail = seen["n"] == 2
        if fail:
            raise RuntimeError("crash before sending this request")
        return real_body(group_df)

    monkeypatch.setattr(google, "build_google_request_body", body_fails_once)
    at = run(app("pages/Brussels.py").run(), bus_ids=["12"])
    sent = offline["google_requests"]
    assert sent == seen["n"] - 1 and sent > 0
    assert used(offline) == sent  # the unstarted request was given back
    assert "Google Routes execution failed" in warnings(at)
    doc = offline["usage_bucket"].document(f"google-routes-usage/{MONTH}.json")
    [reservation] = doc["reservations"].values()
    # Requests queued behind the failure never start (Executor.map cancels them):
    # reserved in full, then only the started ones stay counted.
    assert reservation["planned"] >= seen["n"] and reservation["attempted"] == sent


def test_the_usage_spreadsheet_is_never_written(limit_10):
    run(app("pages/Brussels.py").run(), segment_names=[single_segment_name()])
    assert limit_10["usage_sheet"].calls == []  # reads only
    assert limit_10["usage_sheet"].values == [["month_key", "request_count"], [MONTH, 9]]


# --- the store itself --------------------------------------------------------

def test_concurrent_reservations_never_exceed_the_limit():
    bucket = MemoryBucket(max_delay=0.002)
    store = GcsUsageStore(bucket, seed=lambda month: 3)
    granted = []
    lock = threading.Lock()

    def worker(seed):
        planned = random.Random(seed).randint(1, 3)
        result = store.reserve(MONTH, planned, limit=10)
        if result["allowed"]:
            with lock:
                granted.append(planned)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert 3 + sum(granted) <= 10
    assert store.used(MONTH) == 3 + sum(granted)


def test_concurrent_runs_through_the_fetch_path_never_exceed_the_limit(monkeypatch):
    bucket = MemoryBucket(max_delay=0.002)
    store = GcsUsageStore(bucket)
    sends = []
    lock = threading.Lock()

    def fake_send(api_key, body):
        with lock:
            sends.append(1)
        return {"routes": [{"legs": [{"distanceMeters": 100, "duration": "10s"}]}]}, None

    monkeypatch.setattr(google, "get_usage_store", lambda: store)
    monkeypatch.setattr(google, "send_google_route_request", fake_send)
    monkeypatch.setattr(google, "get_google_routes_api_key", lambda: "fake")
    gdf = gpd.read_file("data/Brussels_map_6km.gpkg").reset_index(drop=True)
    gdf["map_fid"] = gdf.index
    far_apart = [gdf.iloc[[i]] for i in range(0, 1300, 60)]  # 22 one-request RUNs

    threads = [
        threading.Thread(target=google.fetch_google_speeds_for_selected_segments, args=(one,), kwargs={"monthly_limit": 7})
        for one in far_apart
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(sends) == 7 == store.used(MONTH)


def _instance(root, n_runs, seed, out):
    """One 'app instance' (process) pressing RUN n_runs times."""
    store = GcsUsageStore(FileBucket(root, max_delay=0.002))
    rng = random.Random(seed)
    granted = 0
    for _ in range(n_runs):
        planned = rng.randint(1, 4)
        reservation = store.reserve(MONTH, planned, limit=25)
        if reservation["allowed"]:
            attempted = rng.randint(0, planned)  # some requests may never start
            store.settle(reservation, attempted)
            granted += attempted
    out.put(granted)


def test_separate_processes_share_one_strict_counter(tmp_path):
    out = multiprocessing.Queue()
    procs = [multiprocessing.Process(target=_instance, args=(tmp_path, 8, i, out)) for i in range(4)]
    [p.start() for p in procs]
    [p.join(timeout=60) for p in procs]
    attempted = sum(out.get(timeout=5) for _ in procs)
    store = GcsUsageStore(FileBucket(tmp_path))
    assert store.used(MONTH) == attempted <= 25


def test_settle_is_idempotent_and_a_new_month_is_created_once():
    bucket = MemoryBucket(max_delay=0.002)
    seeds = []
    store = GcsUsageStore(bucket, seed=lambda month: seeds.append(month) or 2)
    results = []
    threads = [threading.Thread(target=lambda: results.append(store.reserve(MONTH, 3, 10))) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert store.used(MONTH) == 8  # 2 from the sheet + 3 + 3, the month created exactly once
    store.settle(results[0], 1)
    store.settle(results[0], 0)  # repeated: ignored
    assert store.used(MONTH) == 6


def test_store_errors_are_specific_and_safe():
    from google.api_core import exceptions as gexc

    class Broken(MemoryBucket):
        def get_blob(self, name):
            raise gexc.Forbidden("secret details")

    with pytest.raises(UsageStoreError, match=r"no access to the usage bucket \(HTTP 403\)$"):
        GcsUsageStore(Broken()).reserve(MONTH, 1, 10)
