import streamlit as st


def brussels_left_controls(
    settings_box,
    *,
    segment_options: list[str] | None = None,
    bus_id_options: list[str] | None = None,
) -> dict:
    segment_options = segment_options or []
    bus_id_options = bus_id_options or []

    with settings_box:
        st.markdown("### Brussels controls")
        st.markdown("**Filters (OR logic)**")

        selected_segments = st.multiselect(
            "Segment name(s)",
            options=segment_options,
            default=[],
            key="bru_seg_names",
        )

        selected_bus_ids = st.multiselect(
            "Bus ID(s)",
            options=bus_id_options,
            default=[],
            key="bru_bus_ids",
        )

        st.markdown("---")

        colorize = st.button(
            "Colorize network",
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
        "colorize_clicked": colorize,
        "reset_clicked": reset,
    }
