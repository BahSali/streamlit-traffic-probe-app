"""Foundation-model speed estimates for the optional fourth Brussels map.

Used by speed_layers.py only when core.config.SHOW_FOUNDATION_MODEL_MAP is
True. The display name comes from core.config.FOUNDATION_MODEL_NAME.

TabPFN: the fitted model saved in data/tabpfn_model.json is loaded with
TabPFNRegressor.load_model() (no fit() here) and predicts all segments in one
request. It is a remote model: prediction needs the TabPFN service and a token
(Streamlit secret / environment variable TABPFN_TOKEN). Any failure raises
FoundationUnavailable; the caller then shows no estimates, never made-up ones.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from cities.brussels import foundation_features as ff
from cities.brussels.map_data import MAP_PATH
from core.timing import timed

logger = logging.getLogger("estimator.timing")  # same stderr handler as core.timing

TABPFN_SECRET_KEY = "TABPFN_TOKEN"
UNAVAILABLE_MESSAGE = "TabPFN estimates are unavailable for this run."


# Successful predictions only (never the token), keyed by model, 15-minute bucket and
# the exact feature matrix. Shared by all sessions of the process; the newest entries
# are kept. Each is 1366 floats.
PREDICTION_CACHE_SIZE = 16
_prediction_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
_cache_lock = threading.Lock()
_key_locks: dict[tuple, threading.Lock] = {}


class FoundationUnavailable(Exception):
    """TabPFN could not produce estimates; the message says why."""


def get_tabpfn_token() -> str | None:
    """TabPFN token from the environment, then Streamlit secrets (never hard-coded).

    Root-level Streamlit secrets are also exposed as environment variables, which
    works in the background update thread where st.secrets may not. The token is
    never logged; only whether one was found, and where.
    """
    token, source = os.environ.get(TABPFN_SECRET_KEY), "environment"
    if not (token and str(token).strip()):
        try:
            token, source = st.secrets.get(TABPFN_SECRET_KEY), "secrets"
        except Exception:  # no secrets.toml at all, or no script context in this thread
            token = None
    token = str(token).strip() if token else None
    logger.info("TABPFN_TOKEN available: %s%s", bool(token), f" (from {source})" if token else "")
    return token or None


@st.cache_resource(show_spinner=False)
def load_regressor(model_path: str = str(ff.MODEL_PATH)):
    """The fitted remote TabPFN regressor, restored without fit()."""
    from tabpfn_client import TabPFNRegressor  # imported lazily: opening the page stays light

    return TabPFNRegressor.load_model(model_path)


def model_key(model_path: str = str(ff.MODEL_PATH)) -> str:
    """Identity of the saved model: its model_id and a digest of the whole record."""
    raw = Path(model_path).read_bytes()
    return f"{json.loads(raw).get('model_id')}:{hashlib.sha256(raw).hexdigest()[:16]}"


def prediction_cache_key(features: pd.DataFrame, snapshot_time) -> tuple:
    matrix = np.ascontiguousarray(features[ff.FEATURES].to_numpy(dtype="float64"))
    digest = hashlib.sha256(matrix.tobytes()).hexdigest()
    bucket = pd.Timestamp(snapshot_time).floor(f"{ff.LAG_MINUTES}min").isoformat()
    return (model_key(), bucket, matrix.shape, digest)


def clear_prediction_cache() -> None:
    with _cache_lock:
        _prediction_cache.clear()


def cached_predict(features: pd.DataFrame, snapshot_time, predict) -> np.ndarray:
    """predict(features) unless this exact input was already predicted successfully.

    A failure is raised, never cached. Concurrent sessions with the same input
    wait for one remote call instead of making two.
    """
    key = prediction_cache_key(features, snapshot_time)
    with _cache_lock:
        key_lock = _key_locks.setdefault(key, threading.Lock())
    with key_lock:
        with _cache_lock:
            hit = _prediction_cache.get(key)
            if hit is not None:
                _prediction_cache.move_to_end(key)
        if hit is not None:
            logger.info("TabPFN prediction cache: hit (bucket=%s key=%s)", key[1], key[3][:12])
            return hit.copy()
        logger.info("TabPFN prediction cache: miss (bucket=%s key=%s)", key[1], key[3][:12])
        try:
            predictions = predict(features)
            with _cache_lock:
                _prediction_cache[key] = predictions.copy()
                while len(_prediction_cache) > PREDICTION_CACHE_SIZE:
                    _prediction_cache.popitem(last=False)
            return predictions
        finally:
            with _cache_lock:
                _key_locks.pop(key, None)


def fetch_history(token: str, snapshot_time) -> pd.DataFrame:
    """STIB speeds (index: original segment id, columns: bucket times) for the lag buckets.

    Reuses the primary model's MobilityTwin fetcher and its caches.
    """
    # Imported here: cities.brussels.model loads PyTorch.
    with timed("tabpfn.import_model_module"):
        from cities.brussels.model import fetch_segment_snapshots_for_multiple_buckets

    with timed("tabpfn.lag_schedule") as log:
        buckets = ff.lag_bucket_times(snapshot_time)
        log["buckets"] = len(buckets)
        log["first"] = buckets[-1].isoformat()
        log["last"] = buckets[0].isoformat()

    with timed("tabpfn.history_total") as log:
        history, info = fetch_segment_snapshots_for_multiple_buckets(
            token, MAP_PATH, tuple(b.isoformat() for b in buckets)
        )
        log["windows"] = info.get("historical_fetch_group_count")
        log["window_spans"] = [(w["start"], w["end"]) for w in info.get("historical_fetch_windows", [])]
        log["shape"] = history.shape
    return history


def current_stib_speeds(completed_snapshot_df: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp]:
    """(live STIB speed per original segment id, snapshot time) of the current snapshot."""
    if completed_snapshot_df is None or completed_snapshot_df.empty:
        raise FoundationUnavailable("The current STIB snapshot is unavailable.")
    times = completed_snapshot_df["snapshot_time"].dropna()
    if times.empty:
        raise FoundationUnavailable("The current STIB snapshot has no time.")
    column = "live_speed_kmh" if "live_speed_kmh" in completed_snapshot_df.columns else "final_speed_kmh"
    ids = completed_snapshot_df["segment_id"].astype(str).str.strip()
    speeds = pd.Series(completed_snapshot_df[column].to_numpy(), index=ids)
    return speeds[~speeds.index.duplicated(keep="last")], pd.Timestamp(times.iloc[0])


def attach_by_segment_id(predictions: np.ndarray, static: ff.StaticInputs, gdf: pd.DataFrame) -> pd.Series:
    """Predictions (static/fid order) as a series aligned to gdf.index through the original id."""
    if len(predictions) != len(static.fid):
        raise FoundationUnavailable("TabPFN returned an unexpected number of predictions.")
    by_id = pd.Series(predictions, index=static.original_id)
    mapped = gdf["id"].astype(str).str.strip().map(by_id)
    if mapped.notna().sum() == 0:
        raise FoundationUnavailable("No TabPFN prediction matches a segment of the map.")
    return pd.Series(mapped.to_numpy(dtype=float), index=gdf.index)


def predict_foundation_model_speeds(
    gdf: pd.DataFrame, completed_snapshot_df: pd.DataFrame, token: str | None
) -> pd.Series:
    """Estimated speed (km/h) for each row of the Brussels map gdf, aligned to gdf.index.

    token: the MobilityTwin token (for the lag history). Raises FoundationUnavailable.
    """
    tabpfn_token = get_tabpfn_token()
    if not tabpfn_token:
        raise FoundationUnavailable("Missing TabPFN token (secret TABPFN_TOKEN).")
    if not token:
        raise FoundationUnavailable("Missing MobilityTwin token.")
    try:
        with timed("tabpfn.total") as total:
            current, snapshot_time = current_stib_speeds(completed_snapshot_df)
            with timed("tabpfn.load_static_inputs"):
                static = ff.load_static_inputs()
            history = fetch_history(token, snapshot_time)
            with timed("tabpfn.build_features") as log:
                features = ff.build_feature_frame(static, current, history, snapshot_time)
                log["shape"] = features.shape
            if len(features) != len(static.fid) or list(features.columns) != ff.FEATURES:
                raise FoundationUnavailable("TabPFN features do not match the expected schema.")

            def remote_predict(X: pd.DataFrame) -> np.ndarray:
                with timed("tabpfn.import_client"):
                    import tabpfn_client
                with timed("tabpfn.load_model_json"):  # cached after the first RUN
                    regressor = load_regressor()
                with timed("tabpfn.set_access_token"):
                    tabpfn_client.set_access_token(tabpfn_token)
                with timed("tabpfn.predict", rows=len(X), columns=X.shape[1]):
                    # The first predict() also authenticates and may poll the remote service.
                    result = np.asarray(regressor.predict(X[ff.FEATURES]), dtype=float)
                if result.shape != (len(X),):
                    raise FoundationUnavailable("TabPFN returned an unexpected prediction shape.")
                return result  # only a valid result reaches the cache

            predictions = cached_predict(features, snapshot_time, remote_predict)
            if predictions.shape != (len(features),):
                raise FoundationUnavailable("TabPFN returned an unexpected prediction shape.")
            with timed("tabpfn.map_to_segment_ids"):
                result = attach_by_segment_id(predictions, static, gdf)
            total["mapped"] = int(result.notna().sum())
            return result
    except FoundationUnavailable:
        raise
    except Exception as exc:  # network, quota, model, files, ...: never crash the workflow
        raise FoundationUnavailable(f"{type(exc).__name__}: {exc}") from exc
