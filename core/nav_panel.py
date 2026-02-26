import streamlit as st

PAGES = {
    "Home": "app.py",
    "Ixelles-Etterbeek": "pages/Ixelles_Etterbeek.py",
    "Brussels": "pages/Brussels.py",
}

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
