"""TabPFN foundation-model inference: feature construction, alignment and failure isolation.

The TabPFN service is always faked: no network call, no token, no fit().
"""
import sys
import types

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from cities.brussels import foundation_features as ff
from cities.brussels import foundation_model as fm
from cities.brussels.map_data import MAP_PATH, load_brussels_map
from cities.brussels.speed_layers import build_run_payload
from core import config
from tests.conftest import map_features
from tests.test_pages import run_brussels

SNAPSHOT_TIME = pd.Timestamp("2026-01-01 08:07")


@pytest.fixture(scope="module")
def static():
    return ff.load_static_inputs()


def snapshot_df(static):
    """Current STIB snapshot (original ids as strings, like the real pipeline)."""
    ids = static.original_id
    speeds = [10.0 + int(i) % 30 for i in ids]
    return pd.DataFrame({"snapshot_time": SNAPSHOT_TIME, "segment_id": ids, "live_speed_kmh": speeds,
                         "final_speed_kmh": speeds})


def history_df(static):
    buckets = ff.lag_bucket_times(SNAPSHOT_TIME)
    return pd.DataFrame({b: [50.0 + k + int(i) % 7 for i in static.original_id] for k, b in enumerate(buckets)},
                        index=pd.Index(static.original_id, name="segment_id"))


class FakeRegressor:
    """Stands in for TabPFNRegressor: fit() must never be called; predicts segment_id + 1000."""

    def __init__(self):
        self.predict_calls = []

    def fit(self, *args, **kwargs):
        raise AssertionError("fit() must not be called in the app")

    def predict(self, X):
        self.predict_calls.append(X)
        return X["segment_id"].to_numpy(dtype=float) + 1000.0


@pytest.fixture(autouse=True)
def fresh_prediction_cache():
    fm.clear_prediction_cache()
    yield
    fm.clear_prediction_cache()


@pytest.fixture
def tabpfn(monkeypatch, static):
    reg = FakeRegressor()
    tokens = []
    monkeypatch.setitem(sys.modules, "tabpfn_client", types.SimpleNamespace(set_access_token=tokens.append))
    monkeypatch.setenv("TABPFN_TOKEN", "fake-tabpfn-token")
    monkeypatch.setattr(fm, "load_regressor", lambda *a, **k: reg)
    monkeypatch.setattr(fm, "fetch_history", lambda token, snapshot_time: history_df(static))
    reg.tokens = tokens
    return reg


def run_prediction(gdf, static):
    return fm.predict_foundation_model_speeds(gdf, snapshot_df(static), "mobility-token")


# 1. segment-ID alignment -----------------------------------------------------------------------

def test_predictions_are_attached_by_segment_id_not_row_order(tabpfn, static):
    gdf = load_brussels_map().sample(frac=1, random_state=1)  # shuffled rows
    result = run_prediction(gdf, static)
    fid_of_id = dict(zip(static.original_id, static.fid))
    expected = gdf["id"].astype(str).map(lambda i: fid_of_id[i] - 1 + 1000.0)
    assert result.index.equals(gdf.index)
    np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())
    assert len(tabpfn.predict_calls) == 1 and len(tabpfn.predict_calls[0]) == len(static.fid) == len(gdf)


def test_fid_and_original_id_namespaces(static):
    gdf = gpd.read_file(MAP_PATH, fid_as_index=True)
    assert (static.fid == np.arange(1, len(gdf) + 1)).all()
    assert (static.original_id == gdf["id"].astype(str).to_numpy()).all()
    assert (static.fid != gdf["id"].astype(int).to_numpy()).any()  # the ids have gaps


# 2. adjacency lookup / 3. metadata ------------------------------------------------------------

def test_neighbours_follow_hops_then_manhattan_distance_and_use_minus_one_when_unreachable():
    # chain 0-1-...-11, plus an isolated segment 12
    n_seg = 13
    adjacency = np.zeros((n_seg, n_seg), dtype=int)
    for a in range(11):
        adjacency[a, a + 1] = adjacency[a + 1, a] = 1
    lon = np.arange(n_seg) * 0.001
    lat = np.zeros(n_seg)
    n = ff.neighbour_positions(adjacency, lon, lat, lon, lat)
    assert n.shape == (n_seg, ff.N_NEIGHBOURS)
    assert n[0].tolist() == list(range(1, 11))  # by hops
    assert n[5, :4].tolist() == [4, 6, 3, 7]  # one hop: both sides; ties by distance, then index
    assert 5 not in n[5]  # never the segment itself
    assert (n[12] == -1).all()  # isolated: no reachable neighbour
    assert (n[11][:10] >= 0).all()


def test_real_adjacency_is_labelled_by_gpkg_fid_and_neighbours_are_valid(static):
    adjacency = pd.read_csv(ff.ADJACENCY_PATH, index_col=0)
    assert (adjacency.index.to_numpy() == static.fid).all() and (adjacency.columns.astype(int) == static.fid).all()
    assert static.neighbours.shape == (len(static.fid), ff.N_NEIGHBOURS)
    assert static.neighbours.max() < len(static.fid)
    first = static.neighbours[0][static.neighbours[0] >= 0]
    assert adjacency.to_numpy()[0, first[0]] == 1  # the closest neighbour is adjacent


def test_adjacency_label_mismatch_is_rejected():
    segments = pd.DataFrame(
        {"id": [5, 9], "direction": [1, 1], "bus_lines": ["12", "13"], "start_lon": [4.0, 4.1],
         "start_lat": [50.0, 50.1], "end_lon": [4.1, 4.2], "end_lat": [50.1, 50.2]},
        index=pd.Index([1, 2], name="fid"))
    wrong = pd.DataFrame(np.zeros((2, 2), dtype=int), index=[5, 9], columns=[5, 9])
    with pytest.raises(ff.FoundationInputError):
        ff.build_static_inputs(segments, wrong)


def test_static_metadata_comes_from_the_gpkg_and_segments_metadata_csv_is_not_read(monkeypatch):
    read = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda path, *a, **k: read.append(str(path)) or real(path, *a, **k))
    st.cache_resource.clear()
    static = ff.load_static_inputs()
    assert not [p for p in read if "segments_metadata" in p]
    gdf = gpd.read_file(MAP_PATH, fid_as_index=True)
    assert len(static.metadata) == len(gdf) == 1366
    np.testing.assert_allclose(static.metadata["start_lon"], gdf["start_lon"])
    assert (static.metadata["direction"].to_numpy() == gdf["direction"].to_numpy()).all()
    assert static.metadata["bus_lines"].notna().all() and static.metadata["bus_lines"].dtype.kind == "i"


# 4. temporal lags -------------------------------------------------------------------------------

def test_lag_buckets_are_nine_previous_quarter_hours_and_cross_midnight():
    buckets = ff.lag_bucket_times("2026-01-02 00:07")
    assert len(buckets) == ff.N_LAGS - 1 == 9
    assert buckets[0] == pd.Timestamp("2026-01-01 23:45") and buckets[-1] == pd.Timestamp("2026-01-01 21:45")
    assert all(b - a == pd.Timedelta(minutes=-15) for a, b in zip(buckets, buckets[1:]))


# 5. feature order + values ---------------------------------------------------------------------

def test_feature_schema_and_order():
    assert len(ff.FEATURES) == 119 and len(set(ff.FEATURES)) == 119
    assert ff.FEATURES[:9] == ["segment_id", "direction", "bus_lines", "start_lon", "start_lat", "end_lon",
                               "end_lat", "minute_of_day", "weekday"]
    assert ff.FEATURES[9:21] == ["stib_self_t-0"] + [f"stib_n{j}_t-0" for j in range(10)] + ["stib_self_t-1"]
    assert ff.FEATURES[-1] == "stib_n9_t-9"
    assert [ff.FEATURES.index(c) for c in ff.CATEGORICAL_FEATURES] == [1, 2]  # model record's [1, 2]


def test_feature_frame_values(static):
    current = snapshot_df(static).set_index("segment_id")["live_speed_kmh"]
    history = history_df(static)
    X = ff.build_feature_frame(static, current, history, SNAPSHOT_TIME)
    assert list(X.columns) == ff.FEATURES and len(X) == len(static.fid)
    assert (X["segment_id"] == static.fid - 1).all()
    assert (X["minute_of_day"] == 8 * 60).all() and (X["weekday"] == SNAPSHOT_TIME.weekday()).all()
    pos = 10
    nb = static.neighbours[pos]
    assert X.loc[pos, "stib_self_t-0"] == current[static.original_id[pos]]
    assert X.loc[pos, "stib_n0_t-0"] == current[static.original_id[nb[0]]]
    bucket = ff.lag_bucket_times(SNAPSHOT_TIME)[2]  # t-45
    assert X.loc[pos, "stib_n3_t-3"] == history.loc[static.original_id[nb[3]], ff.lag_bucket_times(SNAPSHOT_TIME)[2]]
    assert X.loc[pos, "stib_self_t-3"] == history.loc[static.original_id[pos], bucket]
    # a missing history bucket gives NaN, never a made-up value
    X2 = ff.build_feature_frame(static, current, history.drop(columns=[bucket]), SNAPSHOT_TIME)
    assert X2["stib_self_t-3"].isna().all()


# 8. token / 10. no fit / 11 failures ------------------------------------------------------------

def test_missing_token_disables_tabpfn_without_calling_it(monkeypatch, static):
    loaded = []
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.setattr(fm, "load_regressor", lambda *a, **k: loaded.append(1))
    with pytest.raises(fm.FoundationUnavailable, match="TabPFN token"):
        run_prediction(load_brussels_map(), static)
    assert loaded == []


def test_token_prefers_the_environment_then_secrets_and_is_never_logged(monkeypatch, caplog):
    secrets = {"TABPFN_TOKEN": "from-secrets"}
    monkeypatch.setattr(st, "secrets", secrets)
    monkeypatch.setenv("TABPFN_TOKEN", "from-env")
    with caplog.at_level("INFO", logger="estimator.timing"):
        assert fm.get_tabpfn_token() == "from-env"
        monkeypatch.delenv("TABPFN_TOKEN")
        assert fm.get_tabpfn_token() == "from-secrets"  # no env var: falls back to st.secrets
        monkeypatch.setattr(st, "secrets", {})
        assert fm.get_tabpfn_token() is None
    assert "TABPFN_TOKEN available: True (from environment)" in caplog.text
    assert "TABPFN_TOKEN available: False" in caplog.text
    assert "from-env" not in caplog.text and "from-secrets" not in caplog.text


def test_prediction_sets_the_token_and_never_fits(tabpfn, static):
    run_prediction(load_brussels_map(), static)
    assert tabpfn.tokens == ["fake-tabpfn-token"]  # FakeRegressor.fit raises if it were called


@pytest.mark.parametrize("bad", ["shape", "error"])
def test_service_errors_and_bad_shapes_become_foundation_unavailable(tabpfn, static, monkeypatch, bad):
    if bad == "shape":
        monkeypatch.setattr(tabpfn, "predict", lambda X: np.zeros(3))
    else:
        monkeypatch.setattr(tabpfn, "predict", lambda X: (_ for _ in ()).throw(RuntimeError("quota exceeded")))
    with pytest.raises(fm.FoundationUnavailable):
        run_prediction(load_brussels_map(), static)


# 9. no historical Google file is read ------------------------------------------------------------

def test_inference_reads_no_google_or_stib_history_file(tabpfn, static, monkeypatch):
    read = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda path, *a, **k: read.append(str(path)) or real(path, *a, **k))
    st.cache_resource.clear()
    run_prediction(load_brussels_map(), static)
    assert not [p for p in read if "oogle" in p or "STIB" in p or "segments_metadata" in p]


# 7 / 11 / 12. build_run_payload and the page -----------------------------------------------------

def run_payload(offline_fixture, monkeypatch, show=True):
    monkeypatch.setattr(config, "SHOW_FOUNDATION_MODEL_MAP", show)
    st.cache_data.clear()
    at, features, table = run_brussels(offline_fixture)
    at = settle(at)
    return at, map_features(offline_fixture["html"][-1]).set_index("id"), table


def test_missing_tabpfn_token_keeps_the_page_working_and_shows_a_warning(offline, monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    at, features, table = run_payload(offline, monkeypatch)
    assert (features["foundation_model_speed_str"] == "N/A").all()
    assert not features["foundation_model_speed"].notna().any()
    assert features["est_speed"].notna().any() and features["bus_speed"].notna().any()  # other maps unaffected
    assert len(table) > 0
    assert any("TabPFN estimates are unavailable for this run." in w.value for w in at.warning)


def test_build_run_payload_does_not_run_tabpfn(offline, monkeypatch, tabpfn):
    monkeypatch.setattr(config, "SHOW_FOUNDATION_MODEL_MAP", True)
    st.cache_data.clear()
    payload = build_run_payload(pd.DataFrame(), refresh_key=0, include_foundation_model=True)
    assert payload["foundation_shown"] is True
    assert tabpfn.predict_calls == [] and tabpfn.tokens == []  # the RUN never waits for TabPFN
    features = map_features("const data = " + payload["geojson_text"] + ";\n")
    assert features["foundation_model_speed"].isna().all() and features["est_speed"].notna().any()


def test_successful_prediction_reaches_the_foundation_map(offline, monkeypatch, tabpfn):
    at, features, table = run_payload(offline, monkeypatch)
    assert features["foundation_model_speed"].notna().all()
    assert not (features["foundation_model_speed"] == 25.0).any()  # the old placeholder is gone
    assert not [w for w in at.warning if "TabPFN" in w.value]
    assert len(tabpfn.predict_calls) == 1  # one batch request for all segments


def test_disabled_foundation_model_makes_no_tabpfn_call(offline, monkeypatch, tabpfn):
    at, features, table = run_payload(offline, monkeypatch, show=False)
    assert tabpfn.predict_calls == [] and tabpfn.tokens == []
    assert not [c for c in features.columns if c.startswith("foundation_model")]


# Prediction cache ---------------------------------------------------------------------------------

def test_identical_inputs_call_the_remote_predict_only_once(tabpfn, static, caplog):
    gdf = load_brussels_map()
    with caplog.at_level("INFO", logger="estimator.timing"):
        first = run_prediction(gdf, static)
        second = run_prediction(gdf, static)
    assert len(tabpfn.predict_calls) == 1
    pd.testing.assert_series_equal(first, second)
    assert caplog.text.index("cache: miss") < caplog.text.index("cache: hit")
    assert caplog.text.count("cache: miss") == 1 and caplog.text.count("cache: hit") == 1


def test_changed_inputs_or_a_new_bucket_predict_again(tabpfn, static, monkeypatch):
    gdf = load_brussels_map()
    run_prediction(gdf, static)
    changed = history_df(static) + 1.0  # different STIB history
    monkeypatch.setattr(fm, "fetch_history", lambda token, snapshot_time: changed)
    run_prediction(gdf, static)
    assert len(tabpfn.predict_calls) == 2
    later = snapshot_df(static).assign(snapshot_time=SNAPSHOT_TIME + pd.Timedelta(minutes=15))
    fm.predict_foundation_model_speeds(gdf, later, "mobility-token")  # next bucket
    assert len(tabpfn.predict_calls) == 3


def test_failed_predictions_are_not_cached(tabpfn, static, monkeypatch):
    gdf = load_brussels_map()
    good = tabpfn.predict
    monkeypatch.setattr(tabpfn, "predict", lambda X: (_ for _ in ()).throw(RuntimeError("quota")))
    with pytest.raises(fm.FoundationUnavailable):
        run_prediction(gdf, static)
    monkeypatch.setattr(tabpfn, "predict", good)
    run_prediction(gdf, static)  # not served from a cached failure
    assert len(tabpfn.predict_calls) == 1
    monkeypatch.setattr(tabpfn, "predict", lambda X: np.zeros(3))  # bad shape: not cached either
    fm.clear_prediction_cache()
    with pytest.raises(fm.FoundationUnavailable):
        run_prediction(gdf, static)
    assert not fm._prediction_cache


def test_the_cache_holds_predictions_only_never_the_token(tabpfn, static):
    run_prediction(load_brussels_map(), static)
    stored = repr(list(fm._prediction_cache.keys())) + repr(list(fm._prediction_cache.values()))
    assert "fake-tabpfn-token" not in stored and "mobility-token" not in stored
    assert all(isinstance(v, np.ndarray) for v in fm._prediction_cache.values())


# Sidebar toggle ------------------------------------------------------------------------------------

def foundation_job_of(at):
    return at.session_state["brussels_foundation"]["job"]


def settle(at, timeout=60):
    """Wait for the TabPFN job of this session, then rerun the page (as the page's poller does)."""
    job = foundation_job_of(at)
    if job is not None:
        assert job.wait(timeout)
        at = at.run()
    assert not at.exception
    return at


def page_run(offline_fixture, monkeypatch, *, default, toggle, click_run=True, wait=True):
    from tests.conftest import app

    monkeypatch.setattr(config, "SHOW_FOUNDATION_MODEL_MAP", default)
    st.cache_data.clear()
    at = app("pages/Brussels.py").run()
    assert not at.exception
    at.toggle(key="bru_show_foundation").set_value(toggle).run()
    if click_run:
        at.multiselect(key="bru_bus_ids").set_value(["12", "71"]).run()
        [run] = [b for b in at.button if b.label == "RUN"]
        at = run.click().run()
        assert not at.exception
        if wait:
            at = settle(at)
    return at


def maps_in(html):
    import re

    return re.findall(r'<div id="(map\d+)" class="map"></div>', html)


def test_toggle_off_shows_three_maps_and_runs_no_tabpfn(offline, monkeypatch, tabpfn):
    at = page_run(offline, monkeypatch, default=True, toggle=False)
    assert maps_in(offline["html"][-1]) == ["map1", "map2", "map3"]
    assert tabpfn.predict_calls == [] and tabpfn.tokens == []
    assert not [c for c in map_features(offline["html"][-1]).columns if c.startswith("foundation_model")]
    assert not [w for w in at.warning if "TabPFN" in w.value]


def test_toggle_on_shows_four_maps_and_uses_tabpfn(offline, monkeypatch, tabpfn):
    at = page_run(offline, monkeypatch, default=False, toggle=True)  # config default off, session on
    assert maps_in(offline["html"][-1]) == ["map1", "map2", "map3", "map4"]
    assert len(tabpfn.predict_calls) == 1
    assert map_features(offline["html"][-1])["foundation_model_speed"].notna().all()


def test_config_default_sets_the_initial_toggle_value(offline, monkeypatch):
    from tests.conftest import app

    for default in (True, False):
        monkeypatch.setattr(config, "SHOW_FOUNDATION_MODEL_MAP", default)
        st.cache_data.clear()
        at = app("pages/Brussels.py").run()
        assert at.toggle(key="bru_show_foundation").value is default
        assert at.toggle(key="bru_show_foundation").label == "Show TabPFN map"
        assert (len(maps_in(offline["html"][-1])) == 4) is default  # idle page follows the toggle


def test_toggling_after_a_run_does_not_call_tabpfn_again_until_run(offline, monkeypatch, tabpfn):
    at = page_run(offline, monkeypatch, default=True, toggle=True)
    assert len(tabpfn.predict_calls) == 1
    at.toggle(key="bru_show_foundation").set_value(False).run()
    assert maps_in(offline["html"][-1]) == ["map1", "map2", "map3", "map4"]  # applied at the next RUN
    assert any("Click 'Run' to apply" in c.value for c in at.caption)
    assert len(tabpfn.predict_calls) == 1
    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert maps_in(offline["html"][-1]) == ["map1", "map2", "map3"]
    assert len(tabpfn.predict_calls) == 1  # OFF: no TabPFN call


def test_second_run_with_identical_inputs_reuses_the_cached_prediction(offline, monkeypatch, tabpfn):
    at = page_run(offline, monkeypatch, default=True, toggle=True)
    st.cache_data.clear()
    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert not at.exception and len(tabpfn.predict_calls) == 1


def test_existing_sidebar_controls_are_unchanged(offline, monkeypatch):
    from tests.conftest import app

    monkeypatch.setattr(config, "SHOW_FOUNDATION_MODEL_MAP", True)
    st.cache_data.clear()
    at = app("pages/Brussels.py").run()
    assert [m.label for m in at.multiselect] == ["Segment name(s)", "Bus ID(s)"]
    assert [b.label for b in at.button] == ["RUN", "Reset colorization"]
    assert [t.label for t in at.toggle] == ["Show TabPFN map"]
    assert [s.label for s in at.selectbox] == ["Choose an area"]


# Non-blocking TabPFN job and freshness ------------------------------------------------------------

def gate_predict(tabpfn, monkeypatch):
    """Make the fake remote predict() wait until the returned event is set."""
    import threading

    gate, real = threading.Event(), tabpfn.predict
    monkeypatch.setattr(tabpfn, "predict", lambda X: (gate.wait(30), real(X))[1])
    return gate


def foundation_titles(html):
    import re

    return re.findall(r"^      <div>(.*)</div>$", html, re.M)


def foundation_captions(at):
    return [c.value for c in at.caption if "prediction" in c.value]


def test_run_finishes_without_waiting_for_tabpfn_and_the_label_shows_the_snapshot(offline, monkeypatch, tabpfn):
    gate = gate_predict(tabpfn, monkeypatch)
    at = page_run(offline, monkeypatch, default=True, toggle=True, wait=False)
    # The RUN is complete: the other maps are drawn, the TabPFN map is empty and its job is running.
    features = map_features(offline["html"][-1])
    assert features["foundation_model_speed"].isna().all() and features["est_speed"].notna().any()
    assert not foundation_job_of(at).done
    assert foundation_captions(at) == ["Updating TabPFN prediction for 08:00…"]
    # The TabPFN map itself says it is updating (and is dimmed); the other panels are unchanged.
    assert foundation_titles(offline["html"][-1]) == [
        "Bus Speeds (STIB)", "Estimated Speeds", "TabPFN Estimate \u2014 updating for 08:00\u2026", "Google API Speeds",
    ]
    assert "#map3 { opacity: 0.5;" in offline["html"][-1]
    assert not [b for b in at.button if b.label == "RUN"][0].disabled  # RUN stays usable
    gate.set()
    at = settle(at)
    features = map_features(offline["html"][-1])
    assert features["foundation_model_speed"].notna().all()
    # 08:00 is the STIB snapshot of the RUN, not the time the answer arrived.
    assert foundation_captions(at) == ["TabPFN prediction based on STIB snapshot: 08:00 (Brussels time)"]
    assert foundation_titles(offline["html"][-1])[2] == "TabPFN Estimate (STIB 08:00)"
    assert "#map3 { opacity" not in offline["html"][-1]  # no longer dimmed
    assert len(tabpfn.predict_calls) == 1


def test_a_cached_prediction_is_shown_immediately_on_the_next_run(offline, monkeypatch, tabpfn):
    at = page_run(offline, monkeypatch, default=True, toggle=True)
    st.cache_data.clear()
    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    # Shown in the same script run that finishes the RUN, before the new job (a cache hit) ends.
    assert map_features(offline["html"][-1])["foundation_model_speed"].notna().all()
    assert "TabPFN prediction based on STIB snapshot: 08:00 (Brussels time)" in foundation_captions(at)
    assert foundation_job_of(at) is None  # same snapshot as the shown prediction: no new job
    assert not [c for c in foundation_captions(at) if c.startswith("Updating")]
    settle(at)
    assert len(tabpfn.predict_calls) == 1


def test_an_older_prediction_stays_labelled_while_a_newer_one_runs(offline, monkeypatch, tabpfn):
    import core.stib_pipeline as stib_pipeline
    from tests.conftest import fake_stib_pipeline

    at = page_run(offline, monkeypatch, default=True, toggle=True)  # 08:00 shown

    def later_snapshot(token, gpkg_path, **kwargs):
        live, completed = fake_stib_pipeline(token, gpkg_path, **kwargs)
        for frame in (live, completed):
            frame["snapshot_time"] = frame["snapshot_time"] + pd.Timedelta(minutes=15)
        return live, completed

    monkeypatch.setattr(stib_pipeline, "run_stib_pipeline", later_snapshot)
    gate = gate_predict(tabpfn, monkeypatch)
    st.cache_data.clear()
    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert map_features(offline["html"][-1])["foundation_model_speed"].notna().all()  # older values kept
    assert foundation_captions(at) == [
        "TabPFN prediction based on STIB snapshot: 08:00 (Brussels time)",
        "Updating TabPFN prediction for 08:15…",
    ]
    assert foundation_titles(offline["html"][-1])[2] == "TabPFN Estimate \u2014 updating for 08:15\u2026"
    gate.set()
    at = settle(at)
    assert foundation_titles(offline["html"][-1])[2] == "TabPFN Estimate (STIB 08:15)"
    assert foundation_captions(at) == ["TabPFN prediction based on STIB snapshot: 08:15 (Brussels time)"]
    assert len(tabpfn.predict_calls) == 2


def test_a_failed_tabpfn_job_warns_keeps_other_maps_and_is_retried_by_run(offline, monkeypatch, tabpfn):
    real = tabpfn.predict
    monkeypatch.setattr(tabpfn, "predict", lambda X: (_ for _ in ()).throw(ConnectionError("service down")))
    at = page_run(offline, monkeypatch, default=True, toggle=True)
    assert any("TabPFN estimates are unavailable for this run." in w.value for w in at.warning)
    assert map_features(offline["html"][-1])["est_speed"].notna().any()
    assert foundation_job_of(at) is None
    at = at.run()  # an ordinary rerun does not start another job
    assert foundation_job_of(at) is None
    monkeypatch.setattr(tabpfn, "predict", real)
    st.cache_data.clear()
    [run] = [b for b in at.button if b.label == "RUN"]
    at = settle(run.click().run())
    assert not [w for w in at.warning if "TabPFN" in w.value]
    assert map_features(offline["html"][-1])["foundation_model_speed"].notna().all()


def test_toggle_off_starts_no_job_and_shows_no_tabpfn_status(offline, monkeypatch, tabpfn):
    at = page_run(offline, monkeypatch, default=True, toggle=False, wait=False)
    assert foundation_job_of(at) is None and tabpfn.predict_calls == []
    assert foundation_captions(at) == []


def test_reset_forgets_the_shown_prediction(offline, monkeypatch, tabpfn):
    at = page_run(offline, monkeypatch, default=True, toggle=True)
    assert at.session_state["brussels_foundation"]["shown"] is not None
    [reset] = [b for b in at.button if b.label == "Reset colorization"]
    at = reset.click().run()
    assert at.session_state["brussels_foundation"]["shown"] is None
    assert foundation_captions(at) == []


def test_with_foundation_speeds_patches_the_map_data_by_segment_id():
    import json

    from cities.brussels.speed_layers import with_foundation_speeds

    features = [{"type": "Feature", "properties": {"id": i}, "geometry": None} for i in (7, 11, 12)]
    payload = {"geojson_text": json.dumps({"type": "FeatureCollection", "features": features}), "html": "old"}
    with_foundation_speeds(payload, {"11": 30.0, "12": float("nan")})
    props = [f["properties"] for f in json.loads(payload["geojson_text"])["features"]]
    assert [p["foundation_model_speed"] for p in props] == [None, 30.0, None]
    assert [p["foundation_model_speed_str"] for p in props] == ["N/A", "30.0 km/h", "N/A"]
    assert props[1]["foundation_model_color"] == props[1]["foundation_model_highlight_color"] != props[0]["foundation_model_color"]
    assert payload["html"] is None  # rebuilt on the next render


def test_concurrent_identical_jobs_share_one_remote_call(tabpfn, static):
    import threading
    import time

    X = ff.build_feature_frame(
        static, snapshot_df(static).set_index("segment_id")["live_speed_kmh"], history_df(static), SNAPSHOT_TIME
    )
    calls, gate, results = [], threading.Event(), []

    def slow_remote(features):
        calls.append(1)
        gate.wait(10)
        return np.arange(len(features), dtype=float)

    threads = [
        threading.Thread(target=lambda: results.append(fm.cached_predict(X, SNAPSHOT_TIME, slow_remote)))
        for _ in range(2)  # two sessions with the same input
    ]
    for t in threads:
        t.start()
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)  # the second session is waiting for the first
    gate.set()
    for t in threads:
        t.join(10)
    assert len(calls) == 1 and len(results) == 2
    np.testing.assert_array_equal(results[0], results[1])
    assert not fm._key_locks  # no lock left behind


def test_synced_maps_mark_the_foundation_panel_only_when_it_is_updating():
    from cities.brussels.synced_maps import build_synced_maps_html

    geojson = '{"type": "FeatureCollection", "features": []}'
    plain = build_synced_maps_html(geojson, 50.8, 4.35, foundation_model_name="TabPFN")
    shown = build_synced_maps_html(geojson, 50.8, 4.35, foundation_model_name="TabPFN", foundation_status="(STIB 14:30)")
    busy = build_synced_maps_html(
        geojson, 50.8, 4.35, foundation_model_name="TabPFN", foundation_status="updating", foundation_updating=True
    )
    assert foundation_titles(plain)[2] == "TabPFN Estimate" and "#map3 {" not in plain
    assert foundation_titles(shown)[2] == "TabPFN Estimate (STIB 14:30)" and "#map3 {" not in shown
    assert foundation_titles(busy)[2] == "TabPFN Estimate updating" and "#map3 { opacity: 0.5;" in busy
    # The three-map layout ignores the foundation options.
    three = build_synced_maps_html(geojson, 50.8, 4.35, foundation_status="x", foundation_updating=True)
    assert "#map3 {" not in three and len(foundation_titles(three)) == 3


# Every map shows the loading state from the moment RUN is pressed ----------------------------------

UPDATING = "— updating…"


def test_synced_maps_dim_and_mark_every_panel_while_a_run_updates():
    from cities.brussels.synced_maps import build_synced_maps_html

    geojson = '{"type": "FeatureCollection", "features": []}'
    four = build_synced_maps_html(geojson, 50.8, 4.35, foundation_model_name="TabPFN", updating=True)
    assert foundation_titles(four) == [
        f"Bus Speeds (STIB) {UPDATING}", f"Estimated Speeds (Model) {UPDATING}",
        f"TabPFN Estimate {UPDATING}", f"Google API Speeds {UPDATING}",
    ]
    assert "#map1, #map2, #map3, #map4 { opacity: 0.5;" in four
    three = build_synced_maps_html(geojson, 50.8, 4.35, updating=True)
    assert [t.endswith(UPDATING) for t in foundation_titles(three)] == [True, True, True]
    assert "#map1, #map2, #map3 { opacity: 0.5;" in three
    idle = build_synced_maps_html(geojson, 50.8, 4.35, foundation_model_name="TabPFN")
    assert UPDATING not in idle and "opacity: 0.5" not in idle


def test_every_map_is_loading_from_the_run_click_until_its_data_arrives(offline, monkeypatch, tabpfn):
    gate = gate_predict(tabpfn, monkeypatch)
    page_run(offline, monkeypatch, default=True, toggle=True, wait=False)
    pages = offline["html"]
    # Rendered right after RUN was pressed (before any new data): all four maps loading.
    started = [h for h in pages if foundation_titles(h)[0].endswith(UPDATING)]
    assert started and all(t.endswith(UPDATING) for t in foundation_titles(started[-1]))
    assert "#map1, #map2, #map3, #map4 { opacity: 0.5;" in started[-1]
    # RUN finished: Bus / Primary / Google are done; only the TabPFN map is still loading.
    final = pages[-1]
    assert foundation_titles(final) == [
        "Bus Speeds (STIB)", "Estimated Speeds", f"TabPFN Estimate — updating for 08:00…", "Google API Speeds",
    ]
    assert "<style>#map3 { opacity: 0.5;" in final  # only that map is dimmed
    gate.set()


def test_the_three_map_layout_is_loading_from_the_run_click_too(offline, monkeypatch, tabpfn):
    page_run(offline, monkeypatch, default=True, toggle=False)
    pages = offline["html"]
    started = [h for h in pages if foundation_titles(h) and foundation_titles(h)[0].endswith(UPDATING)]
    assert started and len(foundation_titles(started[-1])) == 3
    assert "#map1, #map2, #map3 { opacity: 0.5;" in started[-1]
    assert UPDATING not in pages[-1] and "opacity: 0.5" not in pages[-1]  # all loaded
