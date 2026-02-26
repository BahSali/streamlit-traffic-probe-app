import streamlit as st
from core.styles import inject_styles

st.set_page_config(page_title="Average Speed Estimator", layout="wide")
inject_styles()

st.markdown(
    "<h1 style='text-align:center; color:#009688;'>Urban Area Average Speed Estimator</h1>",
    unsafe_allow_html=True
)
st.caption("Select an area/city from the sidebar dropdown.")

PAGE_OPTIONS = {
    "Ixelles-Etterbeek": "pages/1_Ixelles_Etterbeek.py",
    "Brussels": "pages/2_Brussels.py",
    "York": "pages/3_York.py",
}

st.sidebar.markdown("### Navigation")
selected = st.sidebar.selectbox(
    "Select area or city",
    ["-- Select --"] + list(PAGE_OPTIONS.keys()),
    index=0,
)

if selected != "-- Select --":
    st.switch_page(PAGE_OPTIONS[selected])
else:
    st.info("Choose a city/area from the sidebar.")
