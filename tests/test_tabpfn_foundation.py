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
    return run_brussels(offline_fixture)


def test_missing_tabpfn_token_keeps_the_page_working_and_shows_a_warning(offline, monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    at, features, table = run_payload(offline, monkeypatch)
    assert (features["foundation_model_speed_str"] == "N/A").all()
    assert not features["foundation_model_speed"].notna().any()
    assert features["est_speed"].notna().any() and features["bus_speed"].notna().any()  # other maps unaffected
    assert len(table) > 0
    assert any("TabPFN estimates are unavailable for this run." in w.value for w in at.warning)


def test_tabpfn_failure_does_not_crash_build_run_payload(offline, monkeypatch):
    monkeypatch.setattr(config, "SHOW_FOUNDATION_MODEL_MAP", True)
    monkeypatch.setattr("cities.brussels.speed_layers.predict_foundation_model_speeds",
                        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("service down")))
    st.cache_data.clear()
    payload = build_run_payload(pd.DataFrame(), refresh_key=0)
    assert payload["foundation_warning"].startswith("TabPFN estimates are unavailable for this run.")
    features = map_features("const data = " + payload["geojson_text"] + ";\n")
    assert features["foundation_model_speed"].isna().all()
    assert features["est_speed"].notna().any()


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

def page_run(offline_fixture, monkeypatch, *, default, toggle, click_run=True):
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
