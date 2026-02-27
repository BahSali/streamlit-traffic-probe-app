import streamlit as st

MODE_OPTIONS = ["Segment", "Street"]

def brussels_left_controls(
    settings_box,
    *,
    segment_options: list[str] | None = None,
    bus_id_options: list[str] | None = None,
    street_options: list[str] | None = None,
) -> dict:
    """
    Renders Brussels controls in the left panel and returns a dict that can be used
    to filter features in an OR manner.

    - Segment mode: OR across (segment_names, bus_ids). Each can select multiple values.
    - Street mode: placeholder for now.
    """
    segment_options = segment_options or []
    bus_id_options = bus_id_options or []
    street_options = street_options or []

    with settings_box:
        st.markdown("### Brussels controls")

        mode = st.selectbox(
            "Presentation Mode",
            MODE_OPTIONS,
            index=0,
            key="bru_mode",
        )

        st.markdown("---")

        filters = {
            "segment_names": [],
            "bus_ids": [],
            "streets": [],
        }

        if mode == "Segment":
            st.markdown("**Filters (OR logic):**")
            filters["segment_names"] = st.multiselect(
                "Segment name(s)",
                options=segment_options,
                default=[],
                key="bru_seg_names",
                help="Select one or more segment names. OR-ed with Bus ID filter.",
            )
            filters["bus_ids"] = st.multiselect(
                "Bus ID(s)",
                options=bus_id_options,
                default=[],
                key="bru_bus_ids",
                help="Select one or more bus IDs. OR-ed with Segment name filter.",
            )

        elif mode == "Street":
            st.markdown("**Filters (OR logic):**")
            filters["streets"] = st.multiselect(
                "Street(s)",
                options=street_options,
                default=[],
                key="bru_streets",
                help="Prototype placeholder for Street mode.",
            )

        st.markdown("---")

        colorize = st.button(
            "Colorize network",
            use_container_width=True,
            key="bru_colorize_btn",
        )

    return {
        "mode": mode,
        "filters": filters,
        "colorize_clicked": colorize,
    }
