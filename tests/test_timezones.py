"""MobilityTwin history windows must be requested in UTC, not naive local time."""
import io
import re
from unittest import mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import streamlit as st

import cities.brussels.model as model

GPKG = "data/Brussels_map_6km.gpkg"
PLAN = dict(recent_steps=4, use_daily_lag=True, use_weekly_lag=True)


def requested_windows(snapshot_local: str, parquet_for=None):
    """Run the real history-feature builder; return requested UTC windows and the result."""
    st.cache_data.clear()
    windows = []

    def fake_auth(url, token):
        start = float(re.search(r"start_timestamp=([\d.]+)", url).group(1))
        end = float(re.search(r"end_timestamp=([\d.]+)", url).group(1))
        windows.append((pd.Timestamp(start, unit="s", tz="UTC"), pd.Timestamp(end, unit="s", tz="UTC")))
        return {"results": [f"fake://{start}/{end}"]} if parquet_for else {"results": []}

    def fake_download(url):
        start, end = map(float, url[len("fake://"):].split("/"))
        return parquet_for(pd.Timestamp(start, unit="s"), pd.Timestamp(end, unit="s"))

    ids = list(model.build_segment_lookup_from_gpkg(GPKG)["segment_id"].unique())
    with mock.patch.object(model, "auth_request", fake_auth), \
         mock.patch.object(model, "_download_single_parquet", fake_download):
        features, meta = model.build_historical_feature_matrix(
            token="fake", gpkg_path=GPKG, snapshot_time=pd.Timestamp(snapshot_local),
            ordered_segment_ids=ids, **PLAN,
        )
    return windows, features, meta


def utc(local: str) -> pd.Timestamp:
    return pd.Timestamp(local).tz_localize("Europe/Brussels").tz_convert("UTC")


@pytest.mark.parametrize(
    "snapshot, recent_local, lag_local",
    [
        # summer time (UTC+2)
        ("2026-07-15 08:07:31", ("2026-07-15 06:30", "2026-07-15 08:00"), ("2026-07-14 07:30", "2026-07-14 08:15")),
        # winter time (UTC+1)
        ("2026-01-15 08:07:31", ("2026-01-15 06:30", "2026-01-15 08:00"), ("2026-01-14 07:30", "2026-01-14 08:15")),
    ],
)
def test_windows_are_brussels_local_converted_to_utc(snapshot, recent_local, lag_local):
    windows, _, _ = requested_windows(snapshot)
    assert (utc(recent_local[0]), utc(recent_local[1])) in windows
    assert (utc(lag_local[0]), utc(lag_local[1])) in windows
    assert len(windows) == 5  # recent, 1 day, 1-3 weeks


def test_weekly_lag_across_the_spring_change_uses_winter_offset():
    # 2026-03-29 is the switch to summer time; the 1-week lag is still winter time.
    windows, _, _ = requested_windows("2026-04-01 08:07:31")
    assert (utc("2026-03-25 07:30"), utc("2026-03-25 08:15")) in windows
    assert utc("2026-03-25 07:30") == pd.Timestamp("2026-03-25 06:30", tz="UTC")
    assert (utc("2026-04-01 06:30"), utc("2026-04-01 08:00")) in windows
    assert utc("2026-04-01 06:30") == pd.Timestamp("2026-04-01 04:30", tz="UTC")


def test_repeated_and_skipped_hours_do_not_fail():
    # 02:30 happens twice on 2026-10-25 and never on 2026-03-29.
    ambiguous = pd.Timestamp("2026-10-25 02:30")
    start = model.brussels_local_to_epoch_seconds(ambiguous, window_end=False)
    end = model.brussels_local_to_epoch_seconds(ambiguous, window_end=True)
    assert end - start == 3600  # start takes the earlier instant, end the later one
    skipped = model.brussels_local_to_epoch_seconds(pd.Timestamp("2026-03-29 02:30"), window_end=False)
    assert skipped == pd.Timestamp("2026-03-29 01:00", tz="UTC").timestamp()


def synthetic_positions(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pa.Table:
    """Synthetic bus pings (UTC) every 20 s over the requested window for a few stops."""
    lookup = model.build_segment_lookup_from_gpkg(GPKG).head(200)
    times = pd.date_range(start_utc, end_utc, freq="20s")
    frames = [
        pd.DataFrame({
            "lineId": row.lineId, "pointId": int(row.pointId), "directionId": 1,
            "distanceFromPoint": [80.0 * i for i in range(len(times))], "date": times,
        })
        for row in lookup.itertuples()
    ]
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(pd.concat(frames, ignore_index=True)), buf)
    return pq.read_table(io.BytesIO(buf.getvalue()))


@pytest.mark.parametrize("snapshot", ["2026-07-15 08:07:31", "2026-01-15 08:07:31"])
def test_every_history_column_finds_data_at_the_right_time(snapshot):
    _, features, meta = requested_windows(snapshot, parquet_for=synthetic_positions)
    counts = meta["historical_non_null_counts"]
    assert set(counts) == {
        "recent_t_minus_60m", "recent_t_minus_45m", "recent_t_minus_30m", "recent_t_minus_15m",
        "daily_t_minus_1d", "weekly_t_minus_1w", "weekly_t_minus_2w", "weekly_t_minus_3w",
    }
    assert all(n > 0 for n in counts.values()), counts
