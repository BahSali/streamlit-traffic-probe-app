import time
import streamlit as st
from core.styles import inject_styles

# --- Sidebar recovery handler (MUST be before any other st.* calls) ---
BASE_PAGE_CONFIG = dict(
    page_title="Average Speed Estimator",
    layout="wide",
)

if "_sidebar_cycle" not in st.session_state:
    st.session_state["_sidebar_cycle"] = []

if st.session_state["_sidebar_cycle"]:
    next_state = st.session_state["_sidebar_cycle"].pop(0)
    st.set_page_config(**BASE_PAGE_CONFIG, initial_sidebar_state=next_state)
    if st.session_state["_sidebar_cycle"]:
        time.sleep(0.12)
        st.rerun()
else:
    st.set_page_config(**BASE_PAGE_CONFIG, initial_sidebar_state="expanded")

inject_styles()

PAGES = {
    "Ixelles-Etterbeek": "pages/Ixelles_Etterbeek.py",
    "Brussels": "pages/Brussels.py",
}

st.sidebar.markdown("### Map selector")

options = ["-- Select a map --"] + list(PAGES.keys())
selection = st.sidebar.selectbox(
    "Choose an area",
    options,
    index=0,
    key="page_selector",
)

# --- Recovery UI (visible even when sidebar is collapsed) ---
st.markdown(
    "<h1 style='text-align:center; color:#009688;'>Urban Area Average Speed Estimator</h1>",
    unsafe_allow_html=True
)
st.caption("Use the dropdown in the sidebar to select a map.")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Restore sidebar", use_container_width=True):
        # Force the frontend to toggle state: collapsed -> expanded
        st.session_state["_sidebar_cycle"] = ["collapsed", "expanded"]
        st.rerun()

if selection != "-- Select a map --":
    target = PAGES[selection]
    if st.session_state.get("_last_page") != target:
        st.session_state["_last_page"] = target
        st.switch_page(target)
