"""Brussels page state: applied filters, RUN / Reset, and the Google Routes fetch.

RUN and Reset are button callbacks, so the page updates in the same script
run as the click (no extra rerun). The result of each RUN is kept in session
state and redrawn unchanged on unrelated reruns (e.g. editing a filter), so
maps are only rebuilt, and data only fetched, when RUN is pressed.

Google Routes is paid per request: it is called once per RUN, and not at all
when the same selection is RUN again within GOOGLE_RESULT_REUSE_SECONDS.
Each batch actually sent is queued for the private observation log
(core/google_routes/observations.py) and written from the queue, so a failed
or interrupted write is retried on the next rerun without duplicates.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import streamlit as st

from cities.brussels.map_data import get_selected_mask, load_brussels_map
from core.google_routes import observations
from core.google_routes.service import (
    configured_monthly_limit,
    fetch_google_speeds_for_selected_segments,
    get_monthly_google_request_count,
)

logger = logging.getLogger("estimator.observations")

# Same as the cache lifetime of the live STIB data, so a quick repeated RUN
# shows all three maps from the same moment.
GOOGLE_RESULT_REUSE_SECONDS = 90


def empty_google_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["segment_id", "google_speed_kmh", "google_duration_seconds"])


def idle_google_diagnostics(used_count: int, error_message: str | None = None) -> dict:
    """Google diagnostics when no request was sent in this run."""
    limit = configured_monthly_limit()
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
        "usage_remaining_before_run": limit - used_count,
        "usage_used_after_run": used_count,
        "usage_remaining_after_run": limit - used_count,
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
        "brussels_run_requested": False,
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
        st.session_state["brussels_google_diagnostics"] = idle_google_diagnostics(
            get_monthly_google_request_count()
        )


def on_run_clicked() -> None:
    """RUN callback: apply the current filters and ask the page to update."""
    state = st.session_state
    state["brussels_applied_segment_names"] = list(state.get("bru_seg_names", []))
    state["brussels_applied_bus_ids"] = list(state.get("bru_bus_ids", []))
    state["brussels_colorized"] = True
    state["brussels_refresh_key"] += 1
    state["brussels_run_requested"] = True


def on_reset_clicked() -> None:
    """Reset callback: back to the uncoloured network."""
    state = st.session_state
    used_count = state["brussels_google_diagnostics"].get("usage_used_after_run", 0)
    state["brussels_colorized"] = False
    state["brussels_applied_segment_names"] = []
    state["brussels_applied_bus_ids"] = []
    state["brussels_refresh_key"] += 1
    state["brussels_run_requested"] = False
    state["brussels_payload"] = None
    state["brussels_google_fetch"] = None
    state["brussels_google_results_df"] = empty_google_results_df()
    state["brussels_google_diagnostics"] = idle_google_diagnostics(used_count)


def applied_selection() -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(sorted(st.session_state["brussels_applied_segment_names"])),
        tuple(sorted(st.session_state["brussels_applied_bus_ids"])),
    )


def selected_segment_count() -> int:
    segment_names, bus_ids = applied_selection()
    gdf = load_brussels_map()
    return int(get_selected_mask(gdf, list(segment_names), list(bus_ids)).sum())


def reusable_google_age_seconds() -> float | None:
    """Age of the last Google fetch if it can be reused for this RUN, else None."""
    last = st.session_state["brussels_google_fetch"]
    if not last or last["selection"] != applied_selection():
        return None
    age = time.time() - last["fetched_at"]
    return age if age < GOOGLE_RESULT_REUSE_SECONDS else None


def fetch_google_speeds() -> None:
    """Fetch Google speeds for the applied selection (or reuse a fresh identical fetch)."""
    reuse_age = reusable_google_age_seconds()
    if reuse_age is not None:
        diagnostics = dict(st.session_state["brussels_google_diagnostics"])
        diagnostics["info_message"] = (
            f"Google speeds from the RUN {int(reuse_age)} s ago were reused "
            "(same selection); no new Google request was sent."
        )
        st.session_state["brussels_google_diagnostics"] = diagnostics
        return

    used_before_run = st.session_state["brussels_google_diagnostics"].get("usage_used_after_run", 0)

    try:
        gdf = load_brussels_map().copy()

        selected_mask = get_selected_mask(
            gdf=gdf,
            selected_segment_names=st.session_state["brussels_applied_segment_names"],
            selected_bus_ids=st.session_state["brussels_applied_bus_ids"],
        )

        selected_google_gdf = gdf.loc[selected_mask].copy()

        google_result = fetch_google_speeds_for_selected_segments(selected_gdf=selected_google_gdf)
        if google_result.get("observation_batch"):
            batch = google_result["observation_batch"]
            st.session_state["brussels_pending_observation_batches"].append(
                observations.new_batch(batch["started_at"], batch["observations"])
            )

        st.session_state["brussels_google_results_df"] = google_result["google_results_df"]
        st.session_state["brussels_google_diagnostics"] = google_result["diagnostics"]
        if google_result["diagnostics"].get("was_requested"):
            st.session_state["brussels_google_fetch"] = {
                "selection": applied_selection(),
                "fetched_at": time.time(),
            }

    except Exception as exc:
        st.session_state["brussels_google_results_df"] = empty_google_results_df()
        st.session_state["brussels_google_diagnostics"] = idle_google_diagnostics(
            used_before_run,
            error_message=f"Google Routes execution failed: {exc}",
        )


@st.cache_resource(show_spinner=False)
def _observation_worksheet():
    return observations.get_observation_worksheet()


def flush_observation_batches() -> None:
    """Store queued observation batches in the private log (idempotent)."""
    pending = st.session_state.get("brussels_pending_observation_batches") or []
    if not pending:
        return
    try:
        worksheet = _observation_worksheet()
    except Exception as exc:
        logger.warning("observation worksheet unavailable: %s", type(exc).__name__)
        return
    if worksheet is None:
        logger.warning(
            "[google_observations] is not configured; %d observation batch(es) not stored",
            len(pending),
        )
        st.session_state["brussels_pending_observation_batches"] = []
        return
    st.session_state["brussels_pending_observation_batches"] = [
        batch for batch in pending if not observations.persist_batch(worksheet, batch)
    ]
