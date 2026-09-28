"""Brussels page state: applied filters, RUN / Reset, and the Google Routes fetch.

Google Routes is paid per request, so it is only called once per RUN click
(maybe_execute_google_fetch), and results are kept in session state between
reruns.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from cities.brussels.map_data import get_selected_mask, load_brussels_map
from core.google_routes.service import (
    GOOGLE_ROUTES_MONTHLY_LIMIT,
    fetch_google_speeds_for_selected_segments,
    get_monthly_google_request_count,
)


def empty_google_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["segment_id", "google_speed_kmh", "google_duration_seconds"])


def idle_google_diagnostics(used_count: int, error_message: str | None = None) -> dict:
    """Google diagnostics when no request was sent in this run."""
    return {
        "selected_segment_count": 0,
        "group_count": 0,
        "request_count_planned": 0,
        "request_count_sent": 0,
        "success_count": 0,
        "failure_count": 0,
        "usage_month_key": None,
        "usage_monthly_limit": GOOGLE_ROUTES_MONTHLY_LIMIT,
        "usage_used_before_run": used_count,
        "usage_remaining_before_run": GOOGLE_ROUTES_MONTHLY_LIMIT - used_count,
        "usage_used_after_run": used_count,
        "usage_remaining_after_run": GOOGLE_ROUTES_MONTHLY_LIMIT - used_count,
        "was_requested": False,
        "error_message": error_message,
        "info_message": None,
    }


def init_session_state() -> None:
    if "brussels_colorized" not in st.session_state:
        st.session_state["brussels_colorized"] = False

    if "brussels_applied_segment_names" not in st.session_state:
        st.session_state["brussels_applied_segment_names"] = []

    if "brussels_applied_bus_ids" not in st.session_state:
        st.session_state["brussels_applied_bus_ids"] = []

    if "brussels_refresh_key" not in st.session_state:
        st.session_state["brussels_refresh_key"] = 0

    if "brussels_pending_google_fetch" not in st.session_state:
        st.session_state["brussels_pending_google_fetch"] = False

    if "brussels_google_results_df" not in st.session_state:
        st.session_state["brussels_google_results_df"] = empty_google_results_df()

    if "brussels_google_diagnostics" not in st.session_state:
        st.session_state["brussels_google_diagnostics"] = idle_google_diagnostics(
            get_monthly_google_request_count()
        )


def handle_controls(controls: dict) -> None:
    """Apply RUN / Reset clicks from the left panel (reruns the page)."""
    if controls["colorize_clicked"]:
        st.session_state["brussels_applied_segment_names"] = list(
            controls["filters"]["segment_names"]
        )
        st.session_state["brussels_applied_bus_ids"] = list(
            controls["filters"]["bus_ids"]
        )
        st.session_state["brussels_colorized"] = True
        st.session_state["brussels_refresh_key"] += 1
        st.session_state["brussels_pending_google_fetch"] = True
        st.rerun()

    if controls["reset_clicked"]:
        st.session_state["brussels_colorized"] = False
        st.session_state["brussels_applied_segment_names"] = []
        st.session_state["brussels_applied_bus_ids"] = []
        st.session_state["brussels_refresh_key"] += 1
        st.session_state["brussels_pending_google_fetch"] = False
        st.session_state["brussels_google_results_df"] = empty_google_results_df()
        st.session_state["brussels_google_diagnostics"] = idle_google_diagnostics(
            get_monthly_google_request_count()
        )
        st.rerun()


def maybe_execute_google_fetch() -> None:
    """
    Execute Google Routes only when RUN has been pressed.
    """
    if not st.session_state.get("brussels_pending_google_fetch", False):
        return

    used_before_run = get_monthly_google_request_count()

    try:
        gdf = load_brussels_map().copy()

        selected_mask = get_selected_mask(
            gdf=gdf,
            selected_segment_names=st.session_state["brussels_applied_segment_names"],
            selected_bus_ids=st.session_state["brussels_applied_bus_ids"],
        )

        selected_google_gdf = gdf.loc[selected_mask].copy()

        google_result = fetch_google_speeds_for_selected_segments(
            selected_gdf=selected_google_gdf,
            monthly_limit=GOOGLE_ROUTES_MONTHLY_LIMIT,
        )

        st.session_state["brussels_google_results_df"] = google_result["google_results_df"]
        st.session_state["brussels_google_diagnostics"] = google_result["diagnostics"]

    except Exception as exc:
        st.session_state["brussels_google_results_df"] = empty_google_results_df()
        st.session_state["brussels_google_diagnostics"] = idle_google_diagnostics(
            used_before_run,
            error_message=f"Google Routes execution failed: {exc}",
        )
    finally:
        st.session_state["brussels_pending_google_fetch"] = False
