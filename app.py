import streamlit as st
from core.styles import inject_styles

st.set_page_config(
    page_title="Average Speed Estimator",
    layout="wide"
)

inject_styles()

PAGES = {
    "Ixelles-Etterbeek": "pages/Ixelles_Etterbeek.py",
    "Brussels": "pages/Brussels.py",
}

# ---------- Custom Left Panel (Not Sidebar) ----------
nav_col, content_col = st.columns([1, 4])

with nav_col:
    st.markdown("### Map selector")

    selection = st.selectbox(
        "Choose an area",
        ["-- Select a map --"] + list(PAGES.keys()),
        key="page_selector_main"
    )

with content_col:
    st.markdown(
        "<h1 style='text-align:center; color:#009688;'>Urban Area Average Speed Estimator</h1>",
        unsafe_allow_html=True
    )
    st.caption("Select a map from the left panel.")

if selection != "-- Select a map --":
    target = PAGES[selection]
    if st.session_state.get("_last_page") != target:
        st.session_state["_last_page"] = target
        st.switch_page(target)
