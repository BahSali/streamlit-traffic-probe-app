"""Page layout shared by every page: navigation, settings panel and header."""
import streamlit as st

from core.styles import inject_styles

# Map selector entries: label -> page file. Add a line here to publish a new
# city page (see README, "Adding a city").
PAGES = {
    "Home": "app.py",
    "Ixelles-Etterbeek": "pages/Ixelles_Etterbeek.py",
    "Brussels": "pages/Brussels.py",
}

HEADER_COLOR = "#009688"


def setup_page(current_label: str, page_title: str | None = None):
    """Configure the page and draw the left panel.

    Must be the first Streamlit call of a page. Returns (settings_box,
    content_box): put the page's controls in the first, its content in the
    second.
    """
    st.set_page_config(page_title=page_title or current_label, layout="wide")
    inject_styles()
    return render_left_panel(current_label)


def render_left_panel(current_label: str):
    nav_col, content_col = st.columns([1.25, 4.0], gap="large")

    with nav_col:
        st.markdown("### Map selector")

        labels = list(PAGES.keys())
        default_index = labels.index(current_label) if current_label in labels else 0

        selection = st.selectbox(
            "Choose an area",
            labels,
            index=default_index,
            key="left_nav_selector",
        )

        if selection != current_label:
            st.switch_page(PAGES[selection])

        st.markdown("---")
        st.markdown("### Settings")
        settings_box = st.container()

    with content_col:
        content_box = st.container()

    return settings_box, content_box


def page_header(title: str, caption: str | None = None) -> None:
    st.markdown(f"<h2 style='color:{HEADER_COLOR};'>{title}</h2>", unsafe_allow_html=True)
    if caption:
        st.caption(caption)
