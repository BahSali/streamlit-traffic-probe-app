"""Brussels page: three synced maps (STIB bus / estimated / Google speeds),
or four with the optional foundation-model map (sidebar toggle, default
core.config.SHOW_FOUNDATION_MODEL_MAP, applied by RUN).

Each script run:
  1. draws the controls and the content slots, filled with the last RUN's
     result (unchanged, so the maps neither fade nor reload);
  2. while this session's update is running (started by RUN in a background
     job, see cities/brussels/update_job.py), shows its stages in a status
     box under the disabled RUN button: Google speeds (session.py), bus data
     and estimates (speed_layers.py);
  3. when it finishes, refills the same slots once with the new result.
Stage timings go to the server log (core/timing.py), not the page.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import streamlit as st

from cities.brussels.charts import render_brussels_results_visualisation
from cities.brussels.controls import brussels_left_controls, brussels_run_buttons
from cities.brussels.map_data import get_filter_options
from cities.brussels.session import (
    apply_update,
    flush_observation_batches,
    init_session_state,
    on_reset_clicked,
    on_run_clicked,
    unsaved_observation_batches,
    update_job,
)
from cities.brussels.speed_layers import build_idle_payload
from cities.brussels.synced_maps import build_synced_maps_html
from core.config import APPLY_DEMO_SPEED_CORRECTION, FOUNDATION_MODEL_NAME
from core.layout import page_header, setup_page
from core.map_render import show_map_with_legend
from core.timing import timed

logger = logging.getLogger("estimator.brussels")

# While the demo correction is on, the middle map is not pure model output,
# so it is not labelled as such.
if APPLY_DEMO_SPEED_CORRECTION:
    ESTIMATE_MAP_TITLE = "Estimated Speeds"
else:
    ESTIMATE_MAP_TITLE = "Estimated Speeds (Model)"

# Iframe height: one row of three maps, or a 2x2 grid with the foundation-model map.
MAP_HEIGHT_FOUR_MAPS = 940
MAP_HEIGHT_THREE_MAPS = 560
# How often a script run checks the update's progress.
UPDATE_POLL_SECONDS = 0.2


def reorder_columns(df: pd.DataFrame, priority_cols: list[str]) -> pd.DataFrame:
    existing_priority = [col for col in priority_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_priority]
    return df[existing_priority + remaining_cols]


def render_estimation_status(estimation_diagnostics: dict) -> None:
    """One readable line about the model estimates (details go to the server log)."""
    if estimation_diagnostics.get("estimation_mode") == "disabled":
        return  # before the first RUN
    matched = int(estimation_diagnostics.get("matched_segments") or 0)
    error = estimation_diagnostics.get("error_message")
    if error or matched == 0:
        logger.warning("model estimates unavailable: %s", error or "no segment matched")
        st.warning(
            "Model estimates are not available for this run, so the estimated-speed map has no values. "
            "Press RUN to try again."
        )
    else:
        st.caption(f"Model estimates available for {matched:,} road segments.")


UPDATING_LINES = [
    "Model estimates — updating…",
    "Live STIB diagnostics — updating…",
    "Historical coverage — updating…",
]


def render_diagnostics(payload: dict, google_diagnostics: dict, updating: bool = False) -> None:
    """Status lines above the maps: model estimates, Live STIB, Historical coverage.

    While a RUN is in progress (updating=True) the lines say so instead of
    repeating the previous RUN's values.
    """
    if updating:
        for line in UPDATING_LINES:
            st.caption(line)
        return

    diagnostics = payload["diagnostics"]
    estimation_diagnostics = payload.get("estimation_diagnostics", {})

    if estimation_diagnostics:
        render_estimation_status(estimation_diagnostics)

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
            "Historical coverage — "
            f"{estimation_diagnostics.get('historical_non_null_counts', {})}"
        )

    if diagnostics["error_message"]:
        st.warning(diagnostics["error_message"])

    if payload.get("foundation_warning"):
        st.warning(payload["foundation_warning"])

    if google_diagnostics.get("info_message"):
        st.info(google_diagnostics["info_message"])

    if google_diagnostics.get("error_message"):
        st.warning(google_diagnostics["error_message"])


def render_content(slots: dict, payload: dict, updating: bool = False) -> None:
    """Fill the content slots; the same slots are reused so the layout never moves.

    updating=True: a RUN is in progress and payload is the previous result.
    """
    google_diagnostics = st.session_state["brussels_google_diagnostics"]
    enriched_snapshot_df = payload.get("enriched_snapshot_df", pd.DataFrame())
    has_results = st.session_state["brussels_colorized"] and not enriched_snapshot_df.empty

    with slots["diagnostics"].container():
        render_diagnostics(payload, google_diagnostics, updating=updating)

    if payload.get("html") is None:
        with timed("brussels.maps_html") as log:
            payload["html"] = build_synced_maps_html(
                payload["geojson_text"],
                payload["center_lat"],
                payload["center_lon"],
                estimate_title=ESTIMATE_MAP_TITLE,
                foundation_model_name=FOUNDATION_MODEL_NAME if payload.get("foundation_shown") else None,
            )
            log["bytes"] = len(payload["html"])
    with slots["maps"].container():
        show_map_with_legend(
            lambda: st.iframe(
                payload["html"],
                height=MAP_HEIGHT_FOUR_MAPS if payload.get("foundation_shown") else MAP_HEIGHT_THREE_MAPS,
            ),
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
                width="content",
                # During a RUN the previous result is drawn before the new one; one key
                # per result keeps the two buttons from sharing an element id.
                key=f"brussels_download_{payload['result_id']}",
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


settings_box, content_box = setup_page("Brussels")

init_session_state()
# Retry any observation batch whose write failed or was interrupted.
flush_observation_batches()

segment_options, bus_id_options = get_filter_options()

controls = brussels_left_controls(
    settings_box,
    segment_options=segment_options,
    bus_id_options=bus_id_options,
    applied_segment_names=st.session_state["brussels_applied_segment_names"],
    applied_bus_ids=st.session_state["brussels_applied_bus_ids"],
    applied_show_foundation=st.session_state["brussels_applied_show_foundation"],
)
job = update_job()
brussels_run_buttons(controls["buttons_slot"], busy=job is not None, on_run=on_run_clicked, on_reset=on_reset_clicked)
with settings_box:
    status_slot = st.empty()
    observation_warning_slot = st.empty()

with content_box:
    page_header("Brussels")
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
render_content(slots, previous_payload or build_idle_payload(controls["show_foundation"]), updating=job is not None)

if job is not None:
    # The update runs in this session's UpdateJob (started by RUN); this script run
    # only shows its progress. A rerun meanwhile (e.g. a filter edit) stops this
    # script run, and the next one picks up the same job: nothing is started twice.
    with status_slot.container():
        status = st.status("Starting update…")
        st.caption(
            "The maps keep showing the previous results until the update finishes."
            if previous_payload
            else "The maps are coloured when the update finishes."
        )
    # The label also shows the elapsed seconds, so this loop reaches an st call at
    # least once a second: that is where a script run replaced by a newer one
    # (a rerun) stops, instead of polling on until the update ends.
    shown, label = 0, None
    while not job.wait(UPDATE_POLL_SECONDS):
        while shown < len(job.stages):
            status.write(job.stages[shown])
            shown += 1
        current = job.stages[-1] if job.stages else "Starting update"
        new_label = f"{current}… ({int(time.time() - job.started_at)} s)"
        if new_label != label:
            status.update(label=new_label)
            label = new_label

    job.apply_once(apply_update)
    with timed("google_observations.store"):
        flush_observation_batches()
    brussels_run_buttons(controls["buttons_slot"], busy=False, on_run=on_run_clicked, on_reset=on_reset_clicked)
    if job.error is None:
        payload = job.result["payload"]
        render_content(slots, payload)
        status_slot.caption(f"Maps last updated at {payload['updated_at']:%H:%M:%S} (Brussels time).")
    else:
        # The maps still show the last successful result: show its status lines again.
        with slots["diagnostics"].container():
            render_diagnostics(previous_payload or build_idle_payload(controls["show_foundation"]), st.session_state["brussels_google_diagnostics"])
        with status_slot.container():
            failed = st.status("Update failed", state="error", expanded=True)
            failed.write(f"{type(job.error).__name__}: {job.error}")
elif previous_payload:
    status_slot.caption(f"Maps last updated at {previous_payload['updated_at']:%H:%M:%S} (Brussels time).")

render_unsaved_observations(observation_warning_slot)
