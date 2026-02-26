import streamlit as st

from core.styles import inject_styles
from views.ixelles import render as render_ixelles
from views.brussels import render as render_brussels
from views.york import render as render_york

st.set_page_config(page_title="Average Speed Estimator", layout="wide")
inject_styles()

st.markdown(
    "<h1 style='text-align:center; color:#009688;'>Urban Area Average Speed Estimator</h1>",
    unsafe_allow_html=True
)

st.sidebar.markdown("### Area selection")

area = st.sidebar.selectbox(
    "Select city or area",
    ["Ixelles-Etterbeek", "Brussels", "York"],
    index=0
)

st.markdown("---")

if area == "Ixelles-Etterbeek":
    render_ixelles()
elif area == "Brussels":
    render_brussels()
elif area == "York":
    render_york()
