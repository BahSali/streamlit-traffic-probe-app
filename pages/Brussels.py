"""Brussels page: three synced maps (STIB bus / estimated / Google speeds).

Flow on each rerun:
  1. session.py      - session state; RUN / Reset; Google Routes fetch after RUN
  2. speed_layers.py - join live STIB, model estimates and Google speeds
  3. this file       - diagnostics, maps, charts, CSV download, overview
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from cities.brussels.charts import render_brussels_results_visualisation
from cities.brussels.controls import brussels_left_controls
from cities.brussels.map_data import get_filter_options
from cities.brussels.session import handle_controls, init_session_state, maybe_execute_google_fetch
from cities.brussels.speed_layers import prepare_brussels_page_payload
from cities.brussels.synced_maps import build_three_map_html
from core.google_routes.service import (
    GOOGLE_ROUTES_MONTHLY_LIMIT,
    get_monthly_google_request_count,
)
from core.layout import page_header, setup_page
from core.map_render import show_map_with_legend

settings_box, content_box = setup_page("Brussels")

init_session_state()

segment_options, bus_id_options = get_filter_options()

controls = brussels_left_controls(
    settings_box,
    segment_options=segment_options,
    bus_id_options=bus_id_options,
    applied_segment_names=st.session_state["brussels_applied_segment_names"],
    applied_bus_ids=st.session_state["brussels_applied_bus_ids"],
)
handle_controls(controls)

maybe_execute_google_fetch()


def reorder_columns(df: pd.DataFrame, priority_cols: list[str]) -> pd.DataFrame:
    existing_priority = [col for col in priority_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_priority]
    return df[existing_priority + remaining_cols]


def render_diagnostics(payload: dict) -> None:
    diagnostics = payload["diagnostics"]
    estimation_diagnostics = payload.get("estimation_diagnostics", {})
    c_estimation_diagnostics = payload.get("c_estimation_diagnostics", {})
    google_diagnostics = payload.get("google_diagnostics", {})

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
            f"monthly used after run: {google_diagnostics.get('usage_used_after_run', 0)}, "
            f"monthly remaining: {google_diagnostics.get('usage_remaining_after_run', 0)}, "
            f"monthly limit: {google_diagnostics.get('usage_monthly_limit', GOOGLE_ROUTES_MONTHLY_LIMIT)}"
        )

    if estimation_diagnostics.get("error_message"):
        st.warning(estimation_diagnostics["error_message"])

    if c_estimation_diagnostics:
        st.caption(
            f"el_row: {c_estimation_diagnostics.get('eligible_rows', 0)+54}, "
            f"co_row: {c_estimation_diagnostics.get('c_rows', 0)}"
        )

    if c_estimation_diagnostics.get("error_message"):
        st.warning(c_estimation_diagnostics["error_message"])
    if google_diagnostics.get("info_message"):
        st.info(google_diagnostics["info_message"])

    if google_diagnostics.get("error_message"):
        st.warning(google_diagnostics["error_message"])


with content_box:
    page_header(
        "Brussels",
        "Three synced maps for bus-derived, model-derived, and Google-derived speed comparison.",
    )

    with st.spinner("Preparing Brussels maps and speed layers..."):
        payload = prepare_brussels_page_payload(
            colorized=st.session_state["brussels_colorized"],
            selected_segment_names=tuple(st.session_state["brussels_applied_segment_names"]),
            selected_bus_ids=tuple(st.session_state["brussels_applied_bus_ids"]),
            refresh_key=st.session_state["brussels_refresh_key"],
        )

    enriched_snapshot_df = payload.get("enriched_snapshot_df", pd.DataFrame())
    google_diagnostics = payload.get("google_diagnostics", {})
    has_results = st.session_state["brussels_colorized"] and not enriched_snapshot_df.empty

    render_diagnostics(payload)

    html = build_three_map_html(
        payload["geojson"],
        payload["center_lat"],
        payload["center_lon"],
    )
    show_map_with_legend(
        lambda: components.html(html, height=560, scrolling=False),
        ratio=(10, 1),
    )

    st.markdown("---")
    st.markdown("### Performance Analysis")
    if has_results:
        render_brussels_results_visualisation(enriched_snapshot_df)

    st.markdown("---")
    st.markdown("### Results")
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

    st.markdown("---")
    st.markdown("### Overview")

    google_used = google_diagnostics.get("usage_used_after_run", get_monthly_google_request_count())
    google_remaining = google_diagnostics.get(
        "usage_remaining_after_run",
        GOOGLE_ROUTES_MONTHLY_LIMIT - google_used,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Google used", google_used)
    col2.metric("Google left", google_remaining)
    col3.metric("Google segments", payload["selected_google_count"])
    col4.metric("STIB live", payload["live_bus_count"])
