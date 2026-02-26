import streamlit as st
from core.styles import inject_styles

st.set_page_config(page_title="Average Speed Estimator", layout="wide")
inject_styles()

PAGES = {
    "Ixelles-Etterbeek": "pages/Ixelles_Etterbeek.py",
    "Brussels": "pages/Brussels.py",
    #"York": "pages/York.py",
}

st.sidebar.markdown("### Map selector")

options = ["-- Select a map --"] + list(PAGES.keys())
selection = st.sidebar.selectbox(
    "Choose an area",
    options,
    index=0,
    key="page_selector",
)

st.markdown(
    "<h1 style='text-align:center; color:#009688;'>Urban Area Average Speed Estimator</h1>",
    unsafe_allow_html=True
)
st.caption("Use the dropdown in the sidebar to select a map.")

if selection != "-- Select a map --":
    target = PAGES[selection]
    if st.session_state.get("_last_page") != target:
        st.session_state["_last_page"] = target
        st.switch_page(target)
