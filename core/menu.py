import streamlit as st

PAGES = {
    "Ixelles-Etterbeek": "pages/1_Ixelles_Etterbeek.py",
    "Brussels": "pages/2_Brussels.py",
    "York": "pages/3_York.py",
}

def sidebar_dropdown(current_label: str) -> None:
    labels = list(PAGES.keys())
    if current_label not in labels:
        raise ValueError(f"Unknown page label: {current_label}")

    def _go():
        target = st.session_state["area_dropdown"]
        if target != current_label:
            st.switch_page(PAGES[target])

    st.sidebar.selectbox(
        "Area / map",
        labels,
        index=labels.index(current_label),
        key="area_dropdown",
        on_change=_go,
    )
