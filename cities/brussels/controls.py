"""Brussels settings column: Google Routes filters, RUN and Reset."""
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
        st.markdown("### Google Routes Filters")
        st.markdown("Select road segments based on name or serving bus lines to fetch Google speeds.")

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
        buttons_slot = st.empty()

    return {
        "filters": {
            "segment_names": selected_segments,
            "bus_ids": selected_bus_ids,
        },
        "has_pending_changes": has_pending_changes,
        "buttons_slot": buttons_slot,
    }


def brussels_run_buttons(slot, *, busy: bool, on_run=None, on_reset=None) -> None:
    """RUN and Reset, drawn into slot; disabled while this session's update runs.

    The disabled pair has its own keys, so the enabled pair can replace it in
    the same script run when the update finishes (no extra rerun needed).
    """
    suffix = "_busy" if busy else ""
    with slot.container():
        st.button(
            "RUN",
            width="stretch",
            key=f"bru_colorize_btn{suffix}",
            on_click=on_run,
            disabled=busy,
        )
        st.button(
            "Reset colorization",
            width="stretch",
            key=f"bru_reset_colorize_btn{suffix}",
            on_click=on_reset,
            disabled=busy,
        )
