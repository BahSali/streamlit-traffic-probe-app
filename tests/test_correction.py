import pandas as pd
import pytest

from core import config
from core.estimation.correction import (
    DEMO_CORRECTION_MAX_GAP_KMH,
    DEMO_CORRECTION_THRESHOLD_KMH,
    demo_corrected_speeds,
    displayed_estimates,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "segment_id": ["1", "2", "3", "4", "5"],
            "est": [10.0, 30.0, 20.0, None, 12.0],
            "google": [40.0, 31.0, None, 25.0, 3.0],
        }
    )


def corrected(df):
    return demo_corrected_speeds(df, est_col="est", google_col="google", id_col="segment_id")


def test_input_is_not_modified(df):
    before = df.copy()
    corrected(df)
    pd.testing.assert_frame_equal(df, before)


def test_only_rows_beyond_threshold_change(df):
    result = corrected(df)
    changed = (result - df["est"]).abs() > 1e-9
    # row 0: |40-10| > 8.5, row 4: |3-12| > 8.5; row 1 is within, rows 2/3 lack a value
    assert changed.tolist() == [True, False, False, False, True]
    assert pd.isna(result.iloc[3])


def test_corrected_values_sit_just_below_google(df):
    result = corrected(df)
    for i in (0, 4):
        google = df.at[i, "google"]
        assert max(0.0, google - DEMO_CORRECTION_MAX_GAP_KMH) <= result.iloc[i] <= google
    assert DEMO_CORRECTION_THRESHOLD_KMH == 8.5


def test_same_segment_gets_same_value_regardless_of_row_order(df):
    shuffled = df.iloc[[4, 2, 0, 3, 1]].reset_index(drop=True)
    a = corrected(df).set_axis(df["segment_id"])
    b = corrected(shuffled).set_axis(shuffled["segment_id"])
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_flag_selects_corrected_or_original(df, monkeypatch):
    monkeypatch.setattr(config, "APPLY_DEMO_SPEED_CORRECTION", False)
    off = displayed_estimates(df, est_col="est", google_col="google", id_col="segment_id")
    pd.testing.assert_series_equal(off, df["est"])

    monkeypatch.setattr(config, "APPLY_DEMO_SPEED_CORRECTION", True)
    on = displayed_estimates(df, est_col="est", google_col="google", id_col="segment_id")
    pd.testing.assert_series_equal(on, corrected(df))


def test_demo_default_is_on():
    assert config.APPLY_DEMO_SPEED_CORRECTION is True
