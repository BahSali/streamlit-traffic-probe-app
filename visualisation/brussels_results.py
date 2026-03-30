from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt


def _prepare_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()

    for col in ["bus_speed", "est_speed", "google_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "segment_name" not in df.columns:
        df["segment_name"] = df.get("segment_id", "").astype(str)

    return df


def render_brussels_results_visualisation(results_df: pd.DataFrame) -> None:
    st.markdown("### Visualisation")

    df = _prepare_results_df(results_df)
    if df.empty:
        st.info("No results available for visualisation.")
        return

    render_coverage_chart(df)
    render_estimation_vs_google_scatter(df)
    render_estimation_google_error_chart(df)
    render_speed_distribution_chart(df)
