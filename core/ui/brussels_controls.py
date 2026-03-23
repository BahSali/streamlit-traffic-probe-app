import streamlit as st


def brussels_left_controls(
    settings_box,
    *,
    segment_options: list[str] | None = None,
    bus_id_options: list[str] | None = None,
    applied_segment_names: list[str] | None = None,
    applied_bus_ids: list[str] | None = None,
) -> dict:
    segment_options = segment_options or []
    bus_id_options = bus_id_options or []
    applied_segment_names = applied_segment_names or []
    applied_bus_ids = applied_bus_ids or []

    with settings_box:
        st.markdown("### Brussels controls")
        st.markdown("**Filters (OR logic)**")

        selected_segments = st.multiselect(
            "Segment name(s)",
            options=segment_options,
            default=applied_segment_names,
            key="bru_seg_names",
        )

        selected_bus_ids = st.multiselect(
            "Bus ID(s)",
            options=bus_id_options,
            default=applied_bus_ids,
            key="bru_bus_ids",
        )

        has_pending_changes = (
            selected_segments != applied_segment_names
            or selected_bus_ids != applied_bus_ids
        )

        if has_pending_changes:
            st.warning("⚠️ Warning: Filters changed. Click 'Run' to apply changes to the maps.")

        st.markdown("---")

        colorize = st.button(
            "RUN",
            use_container_width=True,
            key="bru_colorize_btn",
        )

        reset = st.button(
            "Reset colorization",
            use_container_width=True,
            key="bru_reset_colorize_btn",
        )

    return {
        "filters": {
            "segment_names": selected_segments,
            "bus_ids": selected_bus_ids,
        },
        "has_pending_changes": has_pending_changes,
        "colorize_clicked": colorize,
        "reset_clicked": reset,
    }
