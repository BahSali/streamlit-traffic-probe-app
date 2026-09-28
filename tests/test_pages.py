import io

import pandas as pd
import pytest

from cities.ixelles_etterbeek.pipeline import RESULTS_CSV
from core import config
from tests.conftest import app, map_features


def test_home_page_renders(offline):
    at = app("app.py").run()
    assert not at.exception
    assert any("Urban Area Average Speed Estimator" in m.value for m in at.markdown)


def test_ixelles_page_loads_committed_results(offline):
    at = app("pages/Ixelles_Etterbeek.py").run()
    assert not at.exception
    [button] = [b for b in at.button if b.label == "Run Traffic Estimation"]
    at = button.click().run()
    assert not at.exception
    committed = pd.read_csv(RESULTS_CSV, sep=";")
    assert len(at.session_state["ixelles_results_dict"]) == committed["SegmentID"].nunique()
    assert any(s.value == "Done. Results loaded." for s in at.success)


def run_brussels(offline):
    at = app("pages/Brussels.py").run()
    assert not at.exception
    at.multiselect(key="bru_bus_ids").set_value(["12", "71"]).run()
    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert not at.exception
    features = map_features(offline["html"][-1]).set_index("id")
    table = pd.read_csv(io.StringIO(offline["downloads"][-1])).set_index("segment_id")
    return at, features, table


@pytest.mark.parametrize("flag", [True, False])
def test_brussels_estimates_match_across_map_and_download(offline, monkeypatch, flag):
    monkeypatch.setattr(config, "APPLY_DEMO_SPEED_CORRECTION", flag)
    at, features, table = run_brussels(offline)

    with_google = features[features["google_speed"].notna()]
    assert len(with_google) > 0
    shared = with_google.index.intersection(table.index)
    pd.testing.assert_series_equal(
        with_google.loc[shared, "est_speed"],
        table.loc[shared, "estimated_speed"],
        check_names=False,
    )

    model = pd.Series([8.0 + i % 5 for i in with_google.index], index=with_google.index)
    changed = (with_google["est_speed"] - model).abs() > 1e-9
    if flag:
        assert changed.any()
        assert (changed == ((with_google["google_speed"] - model).abs() > 8.5)).all()
    else:
        assert not changed.any()


@pytest.mark.parametrize("flag", [True, False])
def test_brussels_public_output_hides_correction_details(offline, monkeypatch, flag):
    monkeypatch.setattr(config, "APPLY_DEMO_SPEED_CORRECTION", flag)
    at, features, table = run_brussels(offline)

    assert not [c for c in features.columns if "model" in c]
    assert not [c for c in table.columns if "model" in c]
    captions = " ".join(c.value for c in at.caption)
    assert "el_row" not in captions and "co_row" not in captions

    html = offline["html"][-1]
    if flag:
        assert "<div>Estimated Speeds</div>" in html and "(Model)" not in html
        assert "model-derived" not in captions
    else:
        assert "<div>Estimated Speeds (Model)</div>" in html
