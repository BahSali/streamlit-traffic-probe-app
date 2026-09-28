"""The whole-day-file cache must give the same model inputs as downloading every time."""
from unittest import mock

import pandas as pd
import pytest
import requests
import streamlit as st

import cities.brussels.model as model
from tests import fake_mobilitytwin as fake

GPKG = "data/Brussels_map_6km.gpkg"
PLAN = dict(recent_steps=4, use_daily_lag=True, use_weekly_lag=True)


@pytest.fixture(scope="module")
def ids():
    lookup = model.build_segment_lookup_from_gpkg(GPKG)
    fake.set_pairs(lookup, n=15)
    return list(lookup["segment_id"].unique())


def features(snapshot, ids):
    return model.build_historical_feature_matrix("t", GPKG, pd.Timestamp(snapshot), ids, **PLAN)[0]


def without_cache(snapshot, ids):
    st.cache_data.clear()
    return features(snapshot, ids)


def test_file_identity_ignores_only_signature_parameters():
    a = model.file_identity("https://f.example/stib/day_2026-07-14.parquet?X-Amz-Signature=abc&X-Amz-Expires=900&part=2")
    b = model.file_identity("https://f.example/stib/day_2026-07-14.parquet?part=2&X-Amz-Signature=zzz&X-Amz-Expires=60")
    c = model.file_identity("https://f.example/stib/day_2026-07-15.parquet?X-Amz-Signature=abc")
    d = model.file_identity("https://f.example/stib/day_2026-07-14.parquet?part=3&X-Amz-Signature=abc")
    assert a == b and a != c and a != d
    assert "Signature" not in a and "part=2" in a


@pytest.mark.parametrize(
    "snapshot",
    [
        "2026-07-15 14:52:10",  # summer
        "2026-01-15 08:07:31",  # winter
        "2026-07-15 01:52:00",  # recent window crosses UTC midnight
        "2026-01-15 00:20:00",  # winter, just before UTC midnight
        "2026-03-30 02:10:00",  # 1-day lag in March's skipped hour
        "2025-10-26 02:10:00",  # October's repeated hour
        "2025-11-02 02:20:00",  # 1-week lag in October's repeated hour
    ],
)
def test_cached_day_files_give_identical_inputs(snapshot, ids):
    with mock.patch.object(requests, "get", fake.fake_get):
        first = without_cache(snapshot, ids)
        later = pd.Timestamp(snapshot) + pd.Timedelta(minutes=15)
        expected_later = without_cache(later, ids)

        st.cache_data.clear()
        assert features(snapshot, ids).equals(first)  # cold
        fake.CALLS["download"] = 0
        cached_later = features(later, ids)  # new windows, new signatures, same day files
        assert cached_later.equals(expected_later)
        assert first.notna().sum().sum() > 0



def test_a_repeat_run_reuses_the_older_days(ids):
    snapshot = pd.Timestamp("2026-07-15 14:52:10")
    with mock.patch.object(requests, "get", fake.fake_get):
        st.cache_data.clear()
        fake.CALLS.update(list=0, download=0)
        features(snapshot, ids)
        first_downloads = fake.CALLS["download"]
        fake.CALLS.update(list=0, download=0)
        features(snapshot + pd.Timedelta(minutes=15), ids)
        assert first_downloads == 5  # 4 older days + the recent day
        assert fake.CALLS["list"] == 5  # every window still asks for its current file list
        assert fake.CALLS["download"] == 0  # all five day files reused
