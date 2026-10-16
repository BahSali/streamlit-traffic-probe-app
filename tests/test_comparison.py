"""Brussels chart data (cities/brussels/comparison.py) and the dashboard's adaptation to its sources."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from cities.brussels import comparison as cmp


def snapshot(n=40, google_every=2):
    """Segments 0..n-1: Primary = 20 + i % 5, STIB missing every 4th, Google on every google_every-th."""
    i = np.arange(n)
    return pd.DataFrame(
        {
            "segment_id": i.astype(str),
            "segment_name": [f"Street {j % 6}" for j in i],
            "live_speed_kmh": np.where(i % 4 == 0, np.nan, 25.0 + i % 3),
            "estimated_speed": 20.0 + i % 5,
            "google_speed_kmh": np.where(i % google_every == 0, 30.0 + i % 7, np.nan),
        }
    )


def test_tabpfn_column_only_when_its_speeds_are_given():
    assert "tabpfn" not in cmp.build_frame(snapshot()).columns
    # TabPFN only knows half of the segments.
    frame = cmp.build_frame(snapshot(), {str(i): 22.0 for i in range(0, 40, 2)})
    assert frame["tabpfn"].notna().sum() == 20


def test_models_share_one_sample_and_stib_has_its_own():
    frame = cmp.build_frame(snapshot(), {str(i): 22.0 for i in range(0, 40, 6)})  # TabPFN on every 6th
    n = cmp.error_summary(cmp.reference_errors(frame))["n"]
    # Google on every 2nd, STIB missing on every 4th: the models' sample is not cut down by STIB.
    assert (n["primary"], n["tabpfn"], n["stib"]) == (7, 7, 10)


def test_reference_needs_enough_google_segments():
    assert cmp.has_reference(cmp.build_frame(snapshot()))  # 20 Google segments
    assert not cmp.has_reference(cmp.build_frame(snapshot(google_every=1000)))  # 1 Google segment
    no_google = cmp.build_frame(snapshot().drop(columns="google_speed_kmh"))
    assert not cmp.has_reference(no_google)
    # Without Google the sources are still comparable with each other.
    assert list(cmp.source_pairs(no_google)) == [("stib", "primary")]


@pytest.mark.parametrize("tabpfn", [False, True])
@pytest.mark.parametrize("google_every", [2, 1000, None])
def test_dashboard_renders_for_every_combination(tabpfn, google_every):
    def page(tabpfn, google_every):
        from cities.brussels.charts import render_brussels_results_visualisation
        from tests.test_comparison import snapshot

        df = snapshot().drop(columns="google_speed_kmh") if google_every is None else snapshot(google_every=google_every)
        render_brussels_results_visualisation(df, {str(i): 21.0 + i % 4 for i in range(40)} if tabpfn else None)

    at = AppTest.from_function(page, args=(tabpfn, google_every), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    tabs = [t.label for t in at.tabs]
    assert ("Absolute Error Distribution" in tabs) == (google_every == 2)  # error charts only with a Google reference
    assert "Street-Level Error" in tabs or "Largest Speed Differences" in tabs


def test_sources_keep_their_order_and_streets_are_ranked_by_the_chosen_model(monkeypatch):
    monkeypatch.setattr(cmp, "TOP_STREETS", 2)
    frame = cmp.build_frame(snapshot(google_every=1), {str(i): 22.0 + i % 9 for i in range(40)})
    errors = cmp.reference_errors(frame)
    assert list(errors["source"].drop_duplicates()) == ["stib", "primary", "tabpfn"]
    assert list(cmp.error_summary(errors).index) == ["stib", "primary", "tabpfn"]

    tabpfn_mae = errors[errors["source"] == "tabpfn"].groupby("segment_name")["abs_error"].mean().sort_values()
    highest = cmp.street_errors(errors, "tabpfn", highest=True)
    lowest = cmp.street_errors(errors, "tabpfn", highest=False)
    assert set(highest["street"]) == set(tabpfn_mae.index[-2:])
    assert set(lowest["street"]) == set(tabpfn_mae.index[:2])
    assert set(highest["source"]) == {"stib", "primary", "tabpfn"}  # the other sources are shown beside it
