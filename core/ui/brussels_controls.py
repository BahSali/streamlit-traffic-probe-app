
import streamlit as st

MODE_OPTIONS = ["Segment", "Street"]

def brussels_left_controls(
    settings_box,
    *,
    segment_options: list[str] | None = None,
    bus_id_options: list[str] | None = None,
    street_options: list[str] | None = None,
) -> dict:
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
        st.markdown("**Filters (OR logic)**")

        filters = {"segment_names": [], "bus_ids": [], "streets": []}

        if mode == "Segment":
            filters["segment_names"] = st.multiselect(
                "Segment name(s)",
                options=segment_options,
                default=[],
                key="bru_seg_names",
            )
            filters["bus_ids"] = st.multiselect(
                "Bus ID(s)",
                options=bus_id_options,
                default=[],
                key="bru_bus_ids",
            )

        elif mode == "Street":
            filters["streets"] = st.multiselect(
                "Street(s)",
                options=street_options,
                default=[],
                key="bru_streets",
            )

        st.markdown("---")
        colorize = st.button(
            "Colorize network",
            use_container_width=True,
            key="bru_colorize_btn",
        )

    return {"mode": mode, "filters": filters, "colorize_clicked": colorize}                    default=st.session_state.get("bru_seg_names", []),
                    key="bru_seg_names",
                )
                filters["bus_ids"] = st.multiselect(
                    "Bus ID(s)",
                    options=bus_id_options,
                    default=st.session_state.get("bru_bus_ids", []),
                    key="bru_bus_ids",
                )

            elif mode == "Street":
                filters["streets"] = st.multiselect(
                    "Street(s)",
                    options=street_options,
                    default=st.session_state.get("bru_streets", []),
                    key="bru_streets",
                )

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                submitted_apply = st.form_submit_button("Apply filters", use_container_width=True)
            with col2:
                submitted_colorize = st.form_submit_button("Colorize", use_container_width=True)

    return {
        "mode": mode,
        "filters": filters,
        "submitted_apply": submitted_apply,
        "submitted_colorize": submitted_colorize,
    }
