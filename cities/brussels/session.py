"""Brussels page state: applied filters, RUN / Reset, and the Google Routes fetch.

RUN and Reset are button callbacks. RUN starts the update in a background
job owned by this browser session (update_job.py), so a rerun during the
update (e.g. editing a filter) neither restarts nor duplicates it; RUN and
Reset are disabled and ignored until it has finished. The result of each RUN
is kept in session state and redrawn unchanged on unrelated reruns, so maps
are only rebuilt, and data only fetched, when RUN is pressed.

Google Routes is paid per request: it is called once per RUN, and not at all
when the same selection is RUN again within GOOGLE_RESULT_REUSE_SECONDS.
Each batch actually sent is queued for the private observation log
(core/google_routes/observations.py) and written from the queue, so a failed
or interrupted write is retried on the next rerun without duplicates.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from cities.brussels.map_data import get_selected_mask, load_brussels_map
from cities.brussels.speed_layers import build_run_payload
from cities.brussels.update_job import UpdateJob
from core.google_routes import observations
from core.google_routes.service import (
    configured_monthly_limit,
    fetch_google_speeds_for_selected_segments,
    monthly_usage_or_none,
)
from core.timing import timed

logger = logging.getLogger("estimator.observations")
run_logger = logging.getLogger("estimator.brussels")

# Same as the cache lifetime of the live STIB data, so a quick repeated RUN
# shows all three maps from the same moment.
GOOGLE_RESULT_REUSE_SECONDS = 90

BRUSSELS_TZ = ZoneInfo("Europe/Brussels")


def empty_google_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["segment_id", "google_speed_kmh", "google_duration_seconds"])


def idle_google_diagnostics(used_count: int | None, error_message: str | None = None) -> dict:
    """Google diagnostics when no request was sent in this run (used_count None = unknown)."""
    limit = configured_monthly_limit()
    remaining = None if used_count is None else limit - used_count
    return {
        "selected_segment_count": 0,
        "group_count": 0,
        "request_count_planned": 0,
        "request_count_sent": 0,
        "success_count": 0,
        "failure_count": 0,
        "usage_month_key": None,
        "usage_monthly_limit": limit,
        "usage_used_before_run": used_count,
        "usage_remaining_before_run": remaining,
        "usage_used_after_run": used_count,
        "usage_remaining_after_run": remaining,
        "was_requested": False,
        "error_message": error_message,
        "info_message": None,
    }


def init_session_state() -> None:
    defaults = {
        "brussels_colorized": False,
        "brussels_applied_segment_names": [],
        "brussels_applied_bus_ids": [],
        "brussels_refresh_key": 0,
        # UpdateJob of the RUN in progress (see update_job.py); None when idle.
        "brussels_update_job": None,
        # Result of the last RUN (see speed_layers.build_run_payload); None before any RUN.
        "brussels_payload": None,
        # {"selection": ..., "fetched_at": epoch seconds} of the last real Google fetch.
        "brussels_google_fetch": None,
        # Observation batches not yet stored in the private log.
        "brussels_pending_observation_batches": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "brussels_google_results_df" not in st.session_state:
        st.session_state["brussels_google_results_df"] = empty_google_results_df()

    if "brussels_google_diagnostics" not in st.session_state:
        st.session_state["brussels_google_diagnostics"] = idle_google_diagnostics(monthly_usage_or_none())


def update_job() -> UpdateJob | None:
    """The update started by this session's last RUN, until its result is applied."""
    return st.session_state.get("brussels_update_job")


def on_run_clicked() -> None:
    """RUN callback: apply the current filters and start the update.

    Ignored while this session's previous update is still running (the
    buttons are disabled then, but a click sent just before that can still
    arrive), so one session never runs two updates at once.
    """
    state = st.session_state
    if update_job() is not None:
        run_logger.info("RUN ignored: this session's update is still running")
        return
    state["brussels_applied_segment_names"] = list(state.get("bru_seg_names", []))
    state["brussels_applied_bus_ids"] = list(state.get("bru_bus_ids", []))
    state["brussels_colorized"] = True
    state["brussels_refresh_key"] += 1
    state["brussels_update_job"] = UpdateJob(compute_update, update_inputs()).start()


def on_reset_clicked() -> None:
    """Reset callback: back to the uncoloured network (ignored while an update runs)."""
    state = st.session_state
    if update_job() is not None:
        return
    used_count = state["brussels_google_diagnostics"].get("usage_used_after_run")
    state["brussels_colorized"] = False
    state["brussels_applied_segment_names"] = []
    state["brussels_applied_bus_ids"] = []
    state["brussels_refresh_key"] += 1
    state["brussels_payload"] = None
    state["brussels_google_fetch"] = None
    state["brussels_google_results_df"] = empty_google_results_df()
    state["brussels_google_diagnostics"] = idle_google_diagnostics(used_count)


def applied_selection() -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(sorted(st.session_state["brussels_applied_segment_names"])),
        tuple(sorted(st.session_state["brussels_applied_bus_ids"])),
    )


def update_inputs() -> dict:
    """Everything the update needs from the session, copied when RUN is pressed."""
    state = st.session_state
    return {
        "selection": applied_selection(),
        "refresh_key": state["brussels_refresh_key"],
        "last_google_fetch": state["brussels_google_fetch"],
        "last_google_results_df": state["brussels_google_results_df"],
        "last_google_diagnostics": dict(state["brussels_google_diagnostics"]),
    }


def selected_segments_gdf(selection):
    segment_names, bus_ids = selection
    gdf = load_brussels_map()
    return gdf.loc[get_selected_mask(gdf, list(segment_names), list(bus_ids))].copy()


def reusable_google_age_seconds(inputs: dict) -> float | None:
    """Age of the last Google fetch if it can be reused for this RUN, else None."""
    last = inputs["last_google_fetch"]
    if not last or last["selection"] != inputs["selection"]:
        return None
    age = time.time() - last["fetched_at"]
    return age if age < GOOGLE_RESULT_REUSE_SECONDS else None


def fetch_google_speeds(inputs: dict, report: Callable[[str], None]) -> dict:
    """Google speeds for the selection, or the last fetch's if identical and fresh.

    Returns the new values of the session's Google entries:
    results_df, diagnostics, fetch, and observation_batch (None when nothing was sent).
    """
    reuse_age = reusable_google_age_seconds(inputs)
    if reuse_age is not None:
        report(f"Reusing Google speeds from {int(reuse_age)} s ago (same selection)")
        diagnostics = dict(inputs["last_google_diagnostics"])
        diagnostics["info_message"] = (
            f"Google speeds from the RUN {int(reuse_age)} s ago were reused "
            "(same selection); no new Google request was sent."
        )
        return {
            "results_df": inputs["last_google_results_df"],
            "diagnostics": diagnostics,
            "fetch": inputs["last_google_fetch"],
            "observation_batch": None,
        }

    used_before_run = inputs["last_google_diagnostics"].get("usage_used_after_run")
    try:
        selected_google_gdf = selected_segments_gdf(inputs["selection"])
        if len(selected_google_gdf):
            report(f"Fetching Google speeds for {len(selected_google_gdf)} selected segments")
        # Nothing selected: no Google stage and no message (no request is sent).
        google_result = fetch_google_speeds_for_selected_segments(selected_gdf=selected_google_gdf)
        batch = google_result.get("observation_batch")
        was_requested = google_result["diagnostics"].get("was_requested")
        return {
            "results_df": google_result["google_results_df"],
            "diagnostics": google_result["diagnostics"],
            "fetch": (
                {"selection": inputs["selection"], "fetched_at": time.time()}
                if was_requested
                else inputs["last_google_fetch"]
            ),
            "observation_batch": (
                observations.new_batch(batch["started_at"], batch["observations"]) if batch else None
            ),
        }
    except Exception as exc:
        return {
            "results_df": empty_google_results_df(),
            "diagnostics": idle_google_diagnostics(
                used_before_run,
                error_message=f"Google Routes execution failed: {exc}",
            ),
            "fetch": inputs["last_google_fetch"],
            "observation_batch": None,
        }


def compute_update(inputs: dict, report: Callable[[str], None]) -> dict:
    """The work of one RUN (runs in the UpdateJob thread: no session state here)."""
    stage_labels = {
        "bus_data": "Loading bus data",
        "estimating": "Estimating speeds",
        "updating_maps": "Updating maps",
    }
    with timed("brussels.run_update"):
        with timed("brussels.google_fetch_step"):
            google = fetch_google_speeds(inputs, report)
        payload = build_run_payload(
            google_results_df=google["results_df"],
            refresh_key=inputs["refresh_key"],
            on_stage=lambda name: report(stage_labels[name]),
        )
    payload["updated_at"] = datetime.now(BRUSSELS_TZ)
    payload["result_id"] = uuid.uuid4().hex
    return {"google": google, "payload": payload}


def apply_update(job: UpdateJob) -> None:
    """Store a finished update's result in the session (called once per job)."""
    state = st.session_state
    if job.error is None:
        google = job.result["google"]
        state["brussels_google_results_df"] = google["results_df"]
        state["brussels_google_diagnostics"] = google["diagnostics"]
        state["brussels_google_fetch"] = google["fetch"]
        if google["observation_batch"] is not None:
            state["brussels_pending_observation_batches"].append(google["observation_batch"])
        state["brussels_payload"] = job.result["payload"]
    state["brussels_update_job"] = None


@st.cache_resource(show_spinner=False)
def _observation_worksheet():
    # Only a working worksheet is cached; a failure raises and is retried next time.
    return observations.get_observation_worksheet()


def flush_observation_batches() -> None:
    """Try to store every queued observation batch; failed ones stay queued with a reason."""
    pending = st.session_state.get("brussels_pending_observation_batches") or []
    if not pending:
        return
    try:
        worksheet = _observation_worksheet()
    except Exception as exc:
        reason = observations.describe_error(exc)
        for batch in pending:
            batch["last_error"] = reason
        logger.warning("observation log unavailable (%d batch(es) queued): %s", len(pending), reason)
        return

    still_pending = []
    for batch in pending:
        try:
            observations.persist_batch(worksheet, batch)
        except observations.ObservationStoreError as exc:
            batch["last_error"] = str(exc)
            still_pending.append(batch)
    st.session_state["brussels_pending_observation_batches"] = still_pending


def unsaved_observation_batches() -> list[dict]:
    return list(st.session_state.get("brussels_pending_observation_batches") or [])
