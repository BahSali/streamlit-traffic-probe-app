import streamlit as st

MODE_OPTIONS = [
    "Segment",
    "Street",
]

def brussels_left_controls(settings_box) -> dict:
    """
    Renders Brussels controls in the left panel.
    Returns a dict with selected UI state.
    """
    with settings_box:
        st.markdown("### Brussels controls")

        mode = st.selectbox(
            "Presentation Mode",
            MODE_OPTIONS,
            index=0,
            key="bru_mode",
        )

        st.markdown("---")

        colorize = st.button(
            "Colorize network",
            use_container_width=True,
            key="bru_colorize_btn",
        )

    return {
        "mode": mode,
        "colorize_clicked": colorize,
    }
