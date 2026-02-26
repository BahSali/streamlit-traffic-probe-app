import streamlit as st
from core.styles import inject_styles

st.set_page_config(page_title="Average Speed Estimator", layout="wide")
inject_styles()

st.markdown("<h1 style='text-align:center; color:#009688;'>Urban Area Average Speed Estimator</h1>", unsafe_allow_html=True)
st.caption("Open a map from the Pages menu (first time). After that, use the left panel navigation inside each map page.")    if st.session_state.get("_last_page") != target:
        st.session_state["_last_page"] = target
        st.switch_page(target)
