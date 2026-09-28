"""Brussels page: three synced maps (STIB bus / estimated / Google speeds).

Each script run:
  1. draws the controls and the content slots, filled with the last RUN's
     result (unchanged, so the maps neither fade nor reload);
  2. if RUN was pressed, updates it with a status box under the RUN button:
     Google speeds (session.py), bus data and estimates (speed_layers.py);
  3. refills the same slots once with the new result.
Stage timings go to the server log (core/timing.py), not the page.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from cities.brussels.charts import render_brussels_results_visualisation
from cities.brussels.controls import brussels_left_controls
from cities.brussels.map_data import get_filter_options
from cities.brussels.session import (
    fetch_google_speeds,
    flush_observation_batches,
    init_session_state,
    on_reset_clicked,
    on_run_clicked,
    reusable_google_age_seconds,
    selected_segment_count,
    unsaved_observation_batches,
)
from cities.brussels.speed_layers import build_idle_payload, build_run_payload
from cities.brussels.synced_maps import build_three_map_html
from core.config import APPLY_DEMO_SPEED_CORRECTION
from core.google_routes.service import configured_monthly_limit
from core.layout import page_header, setup_page
from core.map_render import show_map_with_legend
from core.timing import timed

# While the demo correction is on, the middle map is not pure model output,
# so it is not labelled as such.
if APPLY_DEMO_SPEED_CORRECTION:
    ESTIMATE_MAP_TITLE = "Estimated Speeds"
    PAGE_CAPTION = "Three synced maps for bus-derived, estimated, and Google-derived speed comparison."
else:
    ESTIMATE_MAP_TITLE = "Estimated Speeds (Model)"
    PAGE_CAPTION = "Three synced maps for bus-derived, model-derived, and Google-derived speed comparison."

BRUSSELS_TZ = ZoneInfo("Europe/Brussels")
MAP_HEIGHT = 560


def reorder_columns(df: pd.DataFrame, priority_cols: list[str]) -> pd.DataFrame:
    existing_priority = [col for col in priority_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_priority]
    return df[existing_priority + remaining_cols]


def _or_na(value) -> str:
    return "N/A" if value is None else str(value)


def render_diagnostics(payload: dict, google_diagnostics: dict) -> None:
    diagnostics = payload["diagnostics"]
    estimation_diagnostics = payload.get("estimation_diagnostics", {})

    if diagnostics["error_message"]:
        st.warning(diagnostics["error_message"])

    st.caption(
        "Live STIB diagnostics — "
        f"token: {'yes' if diagnostics['token_found'] else 'no'}, "
        f"map has id: {'yes' if diagnostics['map_has_id_column'] else 'no'}, "
        f"lookup size: {diagnostics['lookup_size']}, "
        f"common segment ids: {diagnostics['common_segment_ids']}, "
        f"matched segments: {diagnostics['matched_segments']}"
    )

    if estimation_diagnostics:
        st.caption(
            "Map 2 estimation — "
            f"mode: {estimation_diagnostics.get('estimation_mode', 'unknown')}, "
            f"snapshot found: {'yes' if estimation_diagnostics.get('snapshot_found') else 'no'}, "
            f"snapshot time: {estimation_diagnostics.get('snapshot_time') or 'N/A'}, "
            f"bucket time: {estimation_diagnostics.get('snapshot_bucket_time') or 'N/A'}, "
            f"model loaded: {'yes' if estimation_diagnostics.get('model_loaded') else 'no'}, "
            f"historical ready: {'yes' if estimation_diagnostics.get('historical_window_ready') else 'no'}, "
            f"fallback window: {'yes' if estimation_diagnostics.get('used_fallback_window') else 'no'}, "
            f"matched segments: {estimation_diagnostics.get('matched_segments', 0)}"
        )

        st.caption(
            "Historical coverage — "
            f"{estimation_diagnostics.get('historical_non_null_counts', {})}"
        )

    if google_diagnostics:
        st.caption(
            "Google Routes diagnostics — "
            f"requested: {'yes' if google_diagnostics.get('was_requested') else 'no'}, "
            f"selected segments: {google_diagnostics.get('selected_segment_count', 0)}, "
            f"groups: {google_diagnostics.get('group_count', 0)}, "
            f"planned requests: {google_diagnostics.get('request_count_planned', 0)}, "
            f"sent requests: {google_diagnostics.get('request_count_sent', 0)}, "
            f"success: {google_diagnostics.get('success_count', 0)}, "
            f"failure: {google_diagnostics.get('failure_count', 0)}, "
            f"monthly used after run: {_or_na(google_diagnostics.get('usage_used_after_run'))}, "
            f"monthly remaining: {_or_na(google_diagnostics.get('usage_remaining_after_run'))}, "
            f"monthly limit: {google_diagnostics.get('usage_monthly_limit', configured_monthly_limit())}"
        )

    if estimation_diagnostics.get("error_message"):
        st.warning(estimation_diagnostics["error_message"])

    if google_diagnostics.get("info_message"):
        st.info(google_diagnostics["info_message"])

    if google_diagnostics.get("error_message"):
        st.warning(google_diagnostics["error_message"])



def render_content(slots: dict, payload: dict) -> None:
    """Fill the content slots; the same slots are reused so the layout never moves."""
    google_diagnostics = st.session_state["brussels_google_diagnostics"]
    enriched_snapshot_df = payload.get("enriched_snapshot_df", pd.DataFrame())
    has_results = st.session_state["brussels_colorized"] and not enriched_snapshot_df.empty

    with slots["diagnostics"].container():
        render_diagnostics(payload, google_diagnostics)

    if payload.get("html") is None:
        with timed("brussels.maps_html") as log:
            payload["html"] = build_three_map_html(
                payload["geojson_text"],
                payload["center_lat"],
                payload["center_lon"],
                estimate_title=ESTIMATE_MAP_TITLE,
            )
            log["bytes"] = len(payload["html"])
    with slots["maps"].container():
        show_map_with_legend(
            lambda: components.html(payload["html"], height=MAP_HEIGHT, scrolling=False),
            ratio=(10, 1),
        )

    with slots["analysis"].container():
        if has_results:
            with timed("brussels.charts"):
                render_brussels_results_visualisation(enriched_snapshot_df)

    with slots["results"].container():
        if has_results:
            st.download_button(
                label="Download results",
                data=reorder_columns(
                    enriched_snapshot_df,
                    ["timestamp", "segment_id", "segment_name", "bus_lines"],
                ).to_csv(index=False).encode("utf-8"),
                file_name="results.csv",
                mime="text/csv",
                use_container_width=False,
            )

    # Authoritative monthly usage (core/google_routes/usage_store.py); None = unavailable.
    google_used = google_diagnostics.get("usage_used_after_run")
    google_remaining = google_diagnostics.get("usage_remaining_after_run")
    with slots["overview"].container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Google used", "N/A" if google_used is None else google_used)
        col2.metric("Google left", "N/A" if google_remaining is None else google_remaining)
        col3.metric("Google segments", payload["selected_google_count"])
        col4.metric("STIB live", payload["live_bus_count"])


def render_unsaved_observations(slot) -> None:
    """Warn about Google observation batches that could not be saved (never silently dropped)."""
    batches = unsaved_observation_batches()
    if not batches:
        slot.empty()
        return
    lines = "\n".join(
        f"- batch `{b['batch_id']}` sent {b['batch_timestamp_utc']} "
        f"({len(b['observations'])} segments): {b.get('last_error') or 'not saved yet'}"
        for b in batches
    )
    slot.error(
        f"**Google observations NOT saved ({len(batches)} batch(es)).**\n\n{lines}\n\n"
        "RUN, the maps and the results are not affected. Saving is retried on every rerun "
        "while this browser session stays open; these observations are lost if the session "
        "ends or the app restarts before they are saved."
    )


def run_update(status) -> dict:
    """Fetch and compute a new result, reporting each stage in the status box."""
    stage_labels = {
        "bus_data": "Loading bus data",
        "estimating": "Estimating speeds",
        "updating_maps": "Updating maps",
    }

    def stage(label: str) -> None:
        status.update(label=f"{label}…")
        status.write(label)

    reuse_age = reusable_google_age_seconds()
    segment_count = selected_segment_count()
    if reuse_age is not None:
        stage(f"Reusing Google speeds from {int(reuse_age)} s ago (same selection)")
    elif segment_count:
        stage(f"Fetching Google speeds for {segment_count} selected segments")
    else:
        stage("No segments selected: skipping Google speeds")
    with timed("brussels.google_fetch_step"):
        fetch_google_speeds()
    with timed("google_observations.store"):
        flush_observation_batches()

    payload = build_run_payload(
        google_results_df=st.session_state["brussels_google_results_df"],
        refresh_key=st.session_state["brussels_refresh_key"],
        on_stage=lambda name: stage(stage_labels[name]),
    )
    payload["updated_at"] = datetime.now(BRUSSELS_TZ)
    return payload


settings_box, content_box = setup_page("Brussels")

init_session_state()
# Retry any observation batch whose write failed or was interrupted.
flush_observation_batches()

segment_options, bus_id_options = get_filter_options()

brussels_left_controls(
    settings_box,
    segment_options=segment_options,
    bus_id_options=bus_id_options,
    applied_segment_names=st.session_state["brussels_applied_segment_names"],
    applied_bus_ids=st.session_state["brussels_applied_bus_ids"],
    on_run=on_run_clicked,
    on_reset=on_reset_clicked,
)
with settings_box:
    status_slot = st.empty()
    observation_warning_slot = st.empty()

with content_box:
    page_header("Brussels", PAGE_CAPTION)
    slots = {"diagnostics": st.empty(), "maps": st.empty()}
    st.markdown("---")
    st.markdown("### Performance Analysis")
    slots["analysis"] = st.empty()
    st.markdown("---")
    st.markdown("### Results")
    slots["results"] = st.empty()
    st.markdown("---")
    st.markdown("### Overview")
    slots["overview"] = st.empty()

previous_payload = st.session_state["brussels_payload"]
render_content(slots, previous_payload or build_idle_payload())

if st.session_state["brussels_run_requested"]:
    with status_slot.container():
        status = st.status("Starting update…")
        st.caption(
            "The maps keep showing the previous results until the update finishes."
            if previous_payload
            else "The maps are coloured when the update finishes."
        )
    try:
        with timed("brussels.run_update"):
            payload = run_update(status)
            st.session_state["brussels_payload"] = payload
            render_content(slots, payload)
    except Exception as exc:
        st.session_state["brussels_run_requested"] = False
        with status_slot.container():
            failed = st.status("Update failed", state="error", expanded=True)
            failed.write(f"{type(exc).__name__}: {exc}")
    else:
        st.session_state["brussels_run_requested"] = False
        status_slot.caption(f"Maps last updated at {payload['updated_at']:%H:%M:%S} (Brussels time).")
elif previous_payload:
    status_slot.caption(f"Maps last updated at {previous_payload['updated_at']:%H:%M:%S} (Brussels time).")

render_unsaved_observations(observation_warning_slot)
