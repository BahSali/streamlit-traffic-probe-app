"""Brussels "Live Traffic Overview" (cities/brussels/live_traffic.py)."""
from cities.brussels.live_traffic import SECRET_NAME, build_live_traffic_html
from tests.conftest import app


def live_pages(offline):
    return [html for html in offline["html"] if "TrafficLayer" in html]


def test_without_the_key_the_page_shows_one_caption_and_no_live_map(offline):
    at = app("pages/Brussels.py").run()
    assert not at.exception and not live_pages(offline)
    assert [c.value for c in at.caption if SECRET_NAME in c.value]


def test_live_map_shows_before_the_first_run_and_reloads_only_for_a_new_run(offline):
    at = app("pages/Brussels.py")
    at.secrets[SECRET_NAME] = 'fake"key'  # quotes must not break the page
    at = at.run()
    assert not at.exception
    before = live_pages(offline)[-1]  # visible before any RUN
    assert '"fake\\"key"' in before

    at.multiselect(key="bru_bus_ids").set_value(["12"]).run()  # unrelated rerun: the same page, so no reload
    assert live_pages(offline)[-1] == before

    [run] = [b for b in at.button if b.label == "RUN"]
    at = run.click().run()
    assert not at.exception
    after = live_pages(offline)[-1]
    assert after != before and at.session_state["brussels_payload"]["result_id"] in after  # a new RUN refreshes it
    assert build_live_traffic_html('fake"key', "x") == build_live_traffic_html('fake"key', "x")
