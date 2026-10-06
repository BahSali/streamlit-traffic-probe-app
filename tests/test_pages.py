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
        assert "model-derived" not in captions and "Three synced maps" not in captions
    else:
        assert "<div>Estimated Speeds (Model)</div>" in html


def test_brussels_run_is_one_pass_and_later_reruns_reuse_it(offline):
    at, features, table = run_brussels(offline)
    assert offline["google_requests"] > 0
    assert offline["stib_fetches"] == 1 and offline["model_runs"] == 1
    assert any(c.value.startswith("Maps last updated at") for c in at.caption)
    maps_after_run = offline["html"][-1]
    calls = dict(offline)

    # Editing a filter does not refetch, re-estimate or rebuild the maps.
    at.multiselect(key="bru_bus_ids").set_value(["12"]).run()
    assert not at.exception
    assert offline["html"][-1] == maps_after_run
    for key in ["google_requests", "stib_fetches", "model_runs"]:
        assert offline[key] == calls[key]
    assert any("Filters changed" in w.value for w in at.warning)


def test_brussels_repeat_run_does_not_repeat_google_requests(offline):
    at, _, first_table = run_brussels(offline)
    sent = offline["google_requests"]

    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert not at.exception
    assert offline["google_requests"] == sent
    assert any("no new Google request was sent" in i.value for i in at.info)
    second_table = pd.read_csv(io.StringIO(offline["downloads"][-1])).set_index("segment_id")
    pd.testing.assert_series_equal(first_table["google_speed_kmh"], second_table["google_speed_kmh"])

    # A different selection is a new request.
    at.multiselect(key="bru_bus_ids").set_value(["12"]).run()
    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert offline["google_requests"] > sent


def test_brussels_timings_go_to_the_log_not_the_page(offline, caplog):
    import logging

    from core.timing import logger

    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="estimator.timing"):
            at, _, _ = run_brussels(offline)
    finally:
        logger.removeHandler(caplog.handler)

    stages = {record.getMessage().split()[1] for record in caplog.records}
    assert {"brussels.run_update", "google_routes.requests", "brussels.model_estimation",
            "brussels.geojson_serialize", "brussels.charts"} <= stages
    page_text = " ".join(e.value for e in [*at.caption, *at.markdown, *at.info])
    assert "timing" not in page_text and "brussels.run_update" not in page_text


def test_brussels_run_twice_without_reset(offline):
    """RUN, wait for the results, RUN again: no duplicate element, one download button."""

    def assert_run_succeeded(at):
        assert not at.exception, [e.value for e in at.exception]
        page_text = " ".join(e.value for e in [*at.markdown, *at.caption])
        assert "Duplicate" not in page_text and "Update failed" not in page_text
        assert any(c.value.startswith("Maps last updated at") for c in at.caption)
        assert len(at.get("download_button")) == 1

    at, _, _ = run_brussels(offline)
    assert_run_succeeded(at)
    first = offline["downloads"][-1]

    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()  # same selection, no Reset
    assert_run_succeeded(at)
    assert offline["downloads"][-1] == first  # Google reused within 90 s: same results

    at.multiselect(key="bru_bus_ids").set_value(["71"]).run()  # a new selection, RUN a third time
    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert_run_succeeded(at)
    assert offline["downloads"][-1] != first

    at.multiselect(key="bru_bus_ids").set_value(["12"]).run()  # an unrelated rerun keeps one button
    assert len(at.get("download_button")) == 1 and not at.exception


INTERNAL_TERMS = ["Google Routes diagnostics", "Three synced maps", "Map 2 estimation", "pt_inference",
                  "snapshot time", "bucket time", "model loaded", "fallback window", "planned requests"]


def page_text(at) -> str:
    return " ".join(e.value for e in [*at.markdown, *at.caption, *at.info, *at.warning])


def test_brussels_visitor_text_before_and_after_run(offline):
    at = app("pages/Brussels.py").run()
    before = page_text(at)
    assert not any(term in before for term in INTERNAL_TERMS)
    assert "Model estimates" not in before  # nothing to report before RUN
    assert "Historical coverage — {}" in before
    assert {m.label for m in at.metric} >= {"Google used", "Google left"}

    at, _, _ = run_brussels(offline)
    after = page_text(at)
    assert not any(term in after for term in INTERNAL_TERMS), [t for t in INTERNAL_TERMS if t in after]
    assert "Model estimates available for 1,366 road segments." in after
    coverage = next(c.value for c in at.caption if c.value.startswith("Historical coverage — "))
    diagnostics = at.session_state["brussels_payload"]["estimation_diagnostics"]
    assert coverage == f"Historical coverage — {diagnostics.get('historical_non_null_counts', {})}"
    assert offline["google_requests"] > 0 and {m.label: m.value for m in at.metric}["Google used"] != "N/A"


def test_brussels_warns_when_model_estimates_are_unavailable(offline, monkeypatch):
    import cities.brussels.model as brussels_model

    def broken(*args, **kwargs):
        raise RuntimeError("checkpoint street count mismatch (internal detail)")

    monkeypatch.setattr(brussels_model, "run_tmp_model_inference", broken)
    at = app("pages/Brussels.py").run()
    at.multiselect(key="bru_bus_ids").set_value(["12"]).run()
    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "Model estimates are not available for this run" in warnings
    assert "internal detail" not in page_text(at) and "Model estimates available" not in page_text(at)
    assert any(c.value.startswith("Historical coverage — ") for c in at.caption)


def status_lines(at) -> list[str]:
    """The captions above the maps, in page order (the left panel's captions excluded)."""
    wanted = ("Model estimates", "Live STIB diagnostics", "Historical coverage")
    return [c.value for c in at.caption if c.value.startswith(wanted)]


def expected_lines(at) -> list[str]:
    payload = at.session_state["brussels_payload"]
    live = payload["diagnostics"]
    estimation = payload["estimation_diagnostics"]
    return [
        f"Model estimates available for {estimation['matched_segments']:,} road segments.",
        "Live STIB diagnostics — "
        f"token: {'yes' if live['token_found'] else 'no'}, "
        f"map has id: {'yes' if live['map_has_id_column'] else 'no'}, "
        f"lookup size: {live['lookup_size']}, "
        f"common segment ids: {live['common_segment_ids']}, "
        f"matched segments: {live['matched_segments']}",
        f"Historical coverage — {estimation.get('historical_non_null_counts', {})}",
    ]


def click_run(at):
    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert not at.exception
    return at


def test_status_lines_after_run_are_in_order_with_this_runs_values(offline):
    at, _, _ = run_brussels(offline)
    assert status_lines(at) == expected_lines(at)
    assert status_lines(at)[0] == "Model estimates available for 1,366 road segments."


def test_run_with_no_google_segments_shows_no_message_and_the_same_order(offline):
    at = app("pages/Brussels.py").run()
    at = click_run(at)  # nothing selected
    assert offline["google_requests"] == 0
    shown = " ".join(e.value for e in [*at.info, *at.warning, *at.caption, *at.markdown])
    assert "No segments selected" not in shown and "No Google request was sent" not in shown
    assert not [w for w in at.warning if "TabPFN" not in w.value]  # the TabPFN notice is separate
    # Only the charts' own empty-state notes below the maps remain.
    assert {i.value for i in at.info} <= {"No overlapping Estimated and Google speed data are available."}
    assert status_lines(at) == expected_lines(at)
    assert at.session_state["brussels_payload"]["selected_google_count"] == 0


def test_the_previous_runs_status_is_not_shown_while_a_run_is_in_progress(offline, monkeypatch):
    import streamlit as st

    at, _, _ = run_brussels(offline)  # first RUN: its status lines are now on the page
    captions = []
    real_caption = st.caption

    def recording_caption(body, *args, **kwargs):
        captions.append(str(body))
        return real_caption(body, *args, **kwargs)

    monkeypatch.setattr(st, "caption", recording_caption)
    at = click_run(at)  # second RUN, same selection

    model_lines = [c for c in captions if c.startswith("Model estimates")]
    # Drawn first as "updating…", then once with this RUN's values; never the old values first.
    assert model_lines == ["Model estimates — updating…", "Model estimates available for 1,366 road segments."]
    first_live = next(c for c in captions if c.startswith("Live STIB diagnostics"))
    first_coverage = next(c for c in captions if c.startswith("Historical coverage"))
    assert first_live == "Live STIB diagnostics — updating…"
    assert first_coverage == "Historical coverage — updating…"
    assert status_lines(at) == expected_lines(at)


def test_limit_warning_still_shows_with_the_status_lines(offline, monkeypatch):
    from core import config

    monkeypatch.setattr(config, "GOOGLE_ROUTES_MONTHLY_LIMIT", 0)
    at = app("pages/Brussels.py").run()
    at.multiselect(key="bru_bus_ids").set_value(["12"]).run()
    at = click_run(at)
    assert "Google Routes monthly limit reached" in " ".join(w.value for w in at.warning)
    assert status_lines(at) == expected_lines(at)
