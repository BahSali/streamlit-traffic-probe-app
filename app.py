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
    st.caption(
        "This app estimates citywide traffic speed by using open GTFS bus trajectories as moving probes."
        "It extracts bus-based speed features, models spatial dependencies through an attention-based graph convolution, "
        "and captures temporal dynamics with temporal convolutions. The trained model then emulates overall traffic conditions "
        "by aligning bus-derived signals with reference speeds obtained from Google Routes API."
    )
    st.caption(
        "**Datasets / Study Areas:**\n"
        "- **iXcell subnetwork (Brussels)** — [paper link](YOUR_IXCELL_PAPER_URL)\n"
        "- **Brussels bus transportation network (code)** — [paper link](YOUR_NETWORK_PAPER_URL)"
    )
