"""Brussels "Live Traffic Overview" (cities/brussels/live_traffic.py)."""
import pytest
import streamlit as st

from cities.brussels.live_traffic import SECRET_NAME, build_live_traffic_html
from tests.conftest import app


@pytest.fixture
def live_pages(monkeypatch):
    """The live-traffic pages (st.html bodies) drawn so far."""
    pages, real_html = [], st.html

    def recording_html(body, *args, **kwargs):
        if "TrafficLayer" in str(body):
            pages.append(str(body))
        return real_html(body, *args, **kwargs)

    monkeypatch.setattr(st, "html", recording_html)
    return lambda: pages


def test_without_the_key_the_page_shows_one_caption_and_no_live_map(offline, live_pages):
    at = app("pages/Brussels.py").run()
    assert not at.exception and not live_pages()
    assert [c.value for c in at.caption if SECRET_NAME in c.value]


def test_live_map_shows_before_the_first_run_and_reloads_only_for_a_new_run(offline, live_pages):
    at = app("pages/Brussels.py")
    at.secrets[SECRET_NAME] = 'fake"key'  # quotes must not break the page
    at = at.run()
    assert not at.exception
    before = live_pages()[-1]  # visible before any RUN
    assert '"fake\\"key"' in before

    at.multiselect(key="bru_bus_ids").set_value(["12"]).run()  # unrelated rerun: the same page, so no reload
    assert live_pages()[-1] == before

    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert not at.exception
    after = live_pages()[-1]
    assert after != before and at.session_state["brussels_payload"]["result_id"] in after  # a new RUN refreshes it
    assert build_live_traffic_html('fake"key', "x") == build_live_traffic_html('fake"key', "x")
