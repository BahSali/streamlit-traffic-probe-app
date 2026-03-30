from __future__ import annotations

import pandas as pd
import streamlit as st


def render_brussels_results_visualisation(results_df: pd.DataFrame) -> None:
    st.markdown("### Visualisation")

    if results_df is None or results_df.empty:
        st.info("No results available for visualisation.")
        return

    st.markdown("#### Preview")
    st.dataframe(results_df, use_container_width=True)

    if "bus_speed" in results_df.columns:
        st.markdown("#### Bus speed distribution")
        st.bar_chart(results_df["bus_speed"].dropna())

    if "est_speed" in results_df.columns:
        st.markdown("#### Estimated speed distribution")
        st.bar_chart(results_df["est_speed"].dropna())

    if "google_speed" in results_df.columns:
        st.markdown("#### Google speed distribution")
        st.bar_chart(results_df["google_speed"].dropna())
