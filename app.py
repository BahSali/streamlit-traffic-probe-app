import streamlit as st
from core.styles import inject_styles
from core.nav_panel import render_left_panel

st.set_page_config(page_title="Average Speed Estimator", layout="wide")
inject_styles()

settings_box, content_box = render_left_panel("Home")

with settings_box:
    st.info("Select a map from the dropdown above.")

with content_box:
    st.markdown(
        "<h1 style='text-align:center; color:#009688;'>Urban Area Average Speed Estimator</h1>",
        unsafe_allow_html=True,
    )
    st.caption("This app uses a persistent left panel (not Streamlit sidebar).")
