"""One update at a time per browser session, without blocking other sessions.

A RUN's work runs in the session's UpdateJob. Here the work is held at a gate
so reruns can happen while it is in progress (AppTest stops a script run that
exceeds its timeout, as Streamlit does when a rerun replaces it).
"""
import threading

import pytest

import cities.brussels.session as session
from cities.brussels.update_job import UpdateJob
from tests.conftest import app

PAGE = "pages/Brussels.py"


@pytest.fixture
def gated(offline, monkeypatch):
    """compute_update waits for gates[selection bus ids] (open by default) and counts calls."""
    real = session.compute_update
    gates = {}
    calls = []

    def compute(inputs, report):
        calls.append(inputs["selection"])
        gate = gates.get(inputs["selection"][1])
        if gate is not None:
            assert gate.wait(60), "gate never opened"
        return real(inputs, report)

    monkeypatch.setattr(session, "compute_update", compute)
    return {"gates": gates, "calls": calls, "record": offline}


def button(at, key):
    [found] = [b for b in at.button if b.key == key]
    return found


def run_timing_out(element_or_app):
    """One script run that is still showing the update when it is replaced."""
    with pytest.raises(RuntimeError, match="timed out"):
        element_or_app.run(timeout=1.5)


def test_filter_change_and_second_click_during_a_run_start_nothing(gated):
    gate = gated["gates"][("12",)] = threading.Event()
    at = app(PAGE).run()
    at.multiselect(key="bru_bus_ids").set_value(["12"]).run()
    requests_before = gated["record"]["google_requests"]

    run_timing_out(button(at, "bru_colorize_btn").click())
    job = at.session_state["brussels_update_job"]
    assert job is not None and not job.done

    # While it runs (RUN and Reset are drawn disabled; AppTest keeps showing the
    # elements of the last completed run), a filter edit reruns the page...
    run_timing_out(at.multiselect(key="bru_bus_ids").set_value(["71"]))
    # ...and clicks that still arrive (e.g. sent just before the buttons were disabled) are ignored.
    run_timing_out(button(at, "bru_colorize_btn").click())
    run_timing_out(button(at, "bru_reset_colorize_btn").click())
    assert at.session_state["brussels_update_job"] is job

    gate.set()
    at.run()
    assert not at.exception
    assert gated["calls"] == [((), ("12",))]  # one update, for the selection applied at RUN
    assert at.session_state["brussels_update_job"] is None
    assert at.session_state["brussels_applied_bus_ids"] == ["12"]
    assert at.session_state["brussels_payload"]["result_id"] == job.result["payload"]["result_id"]
    # Google was asked once, for line 12's segments only.
    assert gated["record"]["google_requests"] - requests_before == job.result["google"]["diagnostics"]["request_count_sent"] > 0
    # The edited filter is kept, not applied, and RUN is enabled again.
    assert at.multiselect(key="bru_bus_ids").value == ["71"]
    assert any("Filters changed" in w.value for w in at.warning)
    assert not button(at, "bru_colorize_btn").disabled
    assert not [b for b in at.button if b.key.endswith("_busy")]
    assert len(gated["record"]["downloads"]) >= 1

    at = button(at, "bru_colorize_btn").click().run()  # the next RUN works normally
    assert not at.exception
    assert gated["calls"] == [((), ("12",)), ((), ("71",))]


def test_a_running_update_in_one_session_does_not_block_another(gated):
    gate = gated["gates"][("12",)] = threading.Event()
    first = app(PAGE).run()
    first.multiselect(key="bru_bus_ids").set_value(["12"]).run()
    run_timing_out(button(first, "bru_colorize_btn").click())

    second = app(PAGE).run()
    second.multiselect(key="bru_bus_ids").set_value(["71"]).run()
    second = button(second, "bru_colorize_btn").click().run()  # finishes while the first waits
    assert not second.exception
    assert second.session_state["brussels_update_job"] is None
    assert second.session_state["brussels_payload"] is not None
    assert not first.session_state["brussels_update_job"].done

    gate.set()
    first.run()
    assert not first.exception
    assert first.session_state["brussels_payload"]["result_id"] != second.session_state["brussels_payload"]["result_id"]
    assert sorted(gated["calls"]) == [((), ("12",)), ((), ("71",))]


def test_a_finished_update_is_applied_once():
    job = UpdateJob(lambda inputs, report: report("stage") or 42, {}).start()
    job.wait(5)
    applied = []
    threads = [threading.Thread(target=job.apply_once, args=(applied.append,)) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert applied == [job] and job.result == 42 and job.stages == ["stage"]


def test_a_failed_update_is_reported_and_run_works_again(gated, monkeypatch):
    at = app(PAGE).run()
    real = session.compute_update

    def broken(inputs, report):
        raise RuntimeError("bus data unavailable")

    monkeypatch.setattr(session, "compute_update", broken)
    at = button(at, "bru_colorize_btn").click().run()
    assert not at.exception
    assert any(s.label == "Update failed" for s in at.status)
    assert at.session_state["brussels_update_job"] is None
    assert not button(at, "bru_colorize_btn").disabled

    monkeypatch.setattr(session, "compute_update", real)
    at = button(at, "bru_colorize_btn").click().run()
    assert not at.exception and at.session_state["brussels_payload"] is not None
