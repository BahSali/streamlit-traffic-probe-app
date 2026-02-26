import streamlit as st

PAGES = {
    "Ixelles-Etterbeek": "pages/Ixelles_Etterbeek.py",
    "Brussels": "pages/Brussels.py",
}

def render_left_panel(current_label: str):
    """
    Renders a persistent left panel (not Streamlit sidebar) with:
    - Map selector dropdown
    - A container for page-specific settings
    Returns:
      settings_container: where the page can render its own controls
      content_container: where the page should render its main content
    """
    nav_col, content_col = st.columns([1.2, 4.0], gap="large")

    with nav_col:
        st.markdown("### Map selector")

        options = list(PAGES.keys())
        try:
            default_index = options.index(current_label)
        except ValueError:
            default_index = 0

        selection = st.selectbox(
            "Choose an area",
            options,
            index=default_index,
            key="nav_map_selector",
        )

        if selection != current_label:
            st.session_state["_last_page"] = PAGES[selection]
            st.switch_page(PAGES[selection])

        st.markdown("---")
        st.markdown("### Settings")
        settings_container = st.container()

    with content_col:
        content_container = st.container()

    return settings_container, content_container
