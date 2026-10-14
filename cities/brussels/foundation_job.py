"""The foundation-model (TabPFN) prediction as its own background job.

A TabPFN prediction takes minutes, so a RUN never waits for it. The RUN draws
the Bus / Primary Model / Google maps as before, with the foundation map empty
or showing the last finished prediction. Once the RUN's result is applied,
this module starts one TabPFN job per session (the same UpdateJob as the RUN,
see update_job.py), for the STIB snapshot of that RUN:

- a finished prediction is kept in session state with the time of the STIB
  snapshot its features were built from (not the time the answer arrived) and
  shown, with that time, until a newer one finishes;
- while a job runs, the page says which snapshot it is updating for;
- one job per session at a time: a newer RUN's snapshot is started when the
  running job ends;
- the thread never touches session state; its result is applied once by the
  script run that sees it finished.

A prediction cache hit (foundation_model.cached_predict) makes the job finish
in a few seconds, so an unchanged input is shown almost immediately.
"""
from __future__ import annotations

import logging
import uuid

import pandas as pd
import streamlit as st

from cities.brussels.foundation_model import (
    UNAVAILABLE_MESSAGE,
    FoundationUnavailable,
    predict_with_snapshot,
    snapshot_time_of,
)
from cities.brussels.map_data import load_brussels_map
from cities.brussels.speed_layers import with_foundation_speeds
from cities.brussels.update_job import UpdateJob
from core import config
from core.data_sources import get_mobility_twin_token
from core.timing import timed

logger = logging.getLogger("estimator.brussels")

STATE_KEY = "brussels_foundation"
# How often the page checks whether the job has finished.
POLL_SECONDS = 2


def _fresh_state() -> dict:
    return {
        "job": None,  # UpdateJob in progress, or None
        "running_for": None,  # snapshot time the running job was started for
        "attempted": None,  # snapshot time of the last started job (no automatic retry)
        "shown": None,  # {"id", "snapshot_time", "speeds"} of the last finished prediction
        "error": None,  # why the last job produced nothing, or None
    }


def foundation_state() -> dict:
    state = st.session_state
    if STATE_KEY not in state:
        state[STATE_KEY] = _fresh_state()
    return state[STATE_KEY]


def foundation_job() -> UpdateJob | None:
    return foundation_state()["job"]


def reset_foundation() -> None:
    """Reset button: forget the shown prediction (a running job just fills the cache)."""
    st.session_state[STATE_KEY] = _fresh_state()


def retry_allowed() -> None:
    """RUN pressed: the current snapshot may be tried again (e.g. after a failure)."""
    foundation_state()["attempted"] = None


def format_snapshot_time(value) -> str:
    return pd.Timestamp(value).strftime("%H:%M")


def compute_foundation(inputs: dict, report) -> dict:
    """The work of one TabPFN job (runs in the UpdateJob thread: no session state here)."""
    with timed("brussels.foundation_job"):
        try:
            gdf = load_brussels_map()
            speeds, snapshot_time = predict_with_snapshot(gdf, inputs["completed_snapshot_df"], inputs["token"])
        except FoundationUnavailable as exc:
            return {"error": str(exc)}
        by_id = dict(zip(gdf["id"].astype(str).str.strip(), speeds.to_numpy(dtype=float)))
        return {"speeds": by_id, "snapshot_time": snapshot_time}


def _apply(job: UpdateJob) -> None:
    state = foundation_state()
    state["job"] = None
    state["running_for"] = None
    if job.error is not None:
        state["error"] = f"{type(job.error).__name__}: {job.error}"
    elif "error" in job.result:
        state["error"] = job.result["error"]
    else:
        state["error"] = None
        state["shown"] = {
            "id": uuid.uuid4().hex,
            "snapshot_time": job.result["snapshot_time"],
            "speeds": job.result["speeds"],
        }


def sync_foundation(payload: dict | None) -> None:
    """Apply a finished job, then start one for the payload's snapshot if it needs it.

    Called on every script run and when a RUN's result is applied; cheap when
    there is nothing to do.
    """
    state = foundation_state()
    job = state["job"]
    if job is not None and job.done:
        job.apply_once(_apply)
    if state["job"] is not None or payload is None or not payload.get("foundation_shown"):
        return

    wanted = snapshot_time_of(payload.get("completed_snapshot_df"))
    if wanted is None:
        state["error"] = "The current STIB snapshot is unavailable."
        return
    if wanted == state["attempted"]:
        return
    shown = state["shown"]
    if shown is not None and shown["snapshot_time"] == wanted and state["error"] is None:
        # Already showing the prediction for this exact snapshot (RUN pressed again
        # within the live-data cache window): nothing to recompute.
        state["attempted"] = wanted
        return
    token = get_mobility_twin_token()
    state["attempted"] = wanted
    if not token:
        state["error"] = "Missing MobilityTwin token."
        return
    logger.info("starting %s prediction job for STIB snapshot %s", config.FOUNDATION_MODEL_NAME, wanted)
    state["error"] = None
    state["running_for"] = wanted
    state["job"] = UpdateJob(
        compute_foundation, {"completed_snapshot_df": payload["completed_snapshot_df"], "token": token}
    ).start()


def display_payload(payload: dict | None) -> dict | None:
    """Put the last finished prediction into the payload's foundation map (once per prediction)."""
    shown = foundation_state()["shown"]
    if payload is None or shown is None or not payload.get("foundation_shown"):
        return payload
    if payload.get("foundation_applied_id") != shown["id"]:
        with_foundation_speeds(payload, shown["speeds"])
        payload["foundation_applied_id"] = shown["id"]
    return payload


def render_foundation_status(payload: dict | None) -> None:
    """Which snapshot the foundation map shows, and what is running (page captions)."""
    if payload is None or not payload.get("foundation_shown"):
        return
    state = foundation_state()
    name = config.FOUNDATION_MODEL_NAME
    if state["shown"] is not None:
        st.caption(
            f"{name} prediction based on STIB snapshot: "
            f"{format_snapshot_time(state['shown']['snapshot_time'])} (Brussels time)"
        )
    if state["job"] is not None:
        st.caption(f"Updating {name} prediction for {format_snapshot_time(state['running_for'])}…")
    elif state["error"]:
        logger.warning("%s prediction unavailable: %s", name, state["error"])
        st.warning(f"{UNAVAILABLE_MESSAGE} ({state['error']})")
