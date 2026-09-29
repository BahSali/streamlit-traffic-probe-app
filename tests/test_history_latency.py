"""Why recent history columns can be 0: MobilityTwin data that stops before "now".

The recent window (t-60 .. t-15) is one request processed by one query, so
the code cannot fill t-60/t-45 and drop t-30/t-15 unless the returned pings
stop early. These tests reproduce that pattern from data latency alone and
check the log line that reports it. Synthetic data only.
"""
import logging

import pandas as pd
import pytest

from core.timing import logger as timing_logger
from tests.test_timezones import requested_windows, synthetic_positions

SNAPSHOT = "2026-07-15 14:52:10"  # Brussels local (summer, UTC+2)
SNAPSHOT_UTC = pd.Timestamp(SNAPSHOT).tz_localize("Europe/Brussels").tz_convert("UTC").tz_localize(None)
RECENT = ["recent_t_minus_60m", "recent_t_minus_45m", "recent_t_minus_30m", "recent_t_minus_15m"]
LAGS = ["daily_t_minus_1d", "weekly_t_minus_1w", "weekly_t_minus_2w", "weekly_t_minus_3w"]


def published_until(latest_utc: pd.Timestamp):
    """A MobilityTwin stand-in that only has pings up to latest_utc."""
    def parquet_for(start, end):
        return synthetic_positions(start, min(end, latest_utc))
    return parquet_for


@pytest.fixture
def timing_log(caplog):
    timing_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="estimator.timing")
    yield caplog
    timing_logger.removeHandler(caplog.handler)


def test_pings_up_to_the_snapshot_fill_all_history_columns():
    _, _, meta = requested_windows(SNAPSHOT, parquet_for=published_until(SNAPSHOT_UTC))
    assert all(meta["historical_non_null_counts"][label] > 0 for label in RECENT + LAGS)


def test_pings_stopping_38_minutes_early_leave_exactly_t30_and_t15_empty(timing_log):
    latest = SNAPSHOT_UTC - pd.Timedelta(minutes=38)  # 12:14 UTC = 14:14 Brussels, before t-30 starts
    _, _, meta = requested_windows(SNAPSHOT, parquet_for=published_until(latest))
    counts = meta["historical_non_null_counts"]

    assert counts["recent_t_minus_60m"] > 0 and counts["recent_t_minus_45m"] > 0
    assert counts["recent_t_minus_30m"] == 0 and counts["recent_t_minus_15m"] == 0
    assert all(counts[label] > 0 for label in LAGS)

    # The log shows the cause: the recent request reaches 12:45 UTC, the pings stop at 12:14.
    recent_line = next(
        r.getMessage() for r in timing_log.records
        if "caller=model" in r.getMessage() and "window_start=2026-07-15T13:15" in r.getMessage()
    )
    assert "requested_utc=11:15-12:45" in recent_line
    assert "pings_utc=11:15-12:14" in recent_line
    assert "empty_minutes_at_window_end=31.0" in recent_line


def test_map_1_last_hour_fetch_logs_the_age_of_the_newest_ping(monkeypatch, timing_log):
    from datetime import datetime, timezone

    import core.stib_historical as stib_historical

    now = datetime(2026, 7, 15, 12, 52, 10, tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    latest = pd.Timestamp("2026-07-15 12:15:10")
    table = synthetic_positions(pd.Timestamp("2026-07-15 11:52:10"), latest)
    monkeypatch.setattr(stib_historical, "datetime", FixedDatetime)
    monkeypatch.setattr(stib_historical, "auth_request", lambda url, token: {"results": ["fake://1"]})
    monkeypatch.setattr(stib_historical, "download_parquets", lambda urls: table)
    stib_historical.fetch_historical_segment_speeds(token="fake", gpkg_path="data/Brussels_map_6km.gpkg")

    line = next(r.getMessage() for r in timing_log.records if "caller=stib_historical" in r.getMessage())
    assert "latest_ping_utc=12:15:10" in line and "latest_ping_age_min=37.0" in line
