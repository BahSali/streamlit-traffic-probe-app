from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt


def _prepare_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()

    rename_map = {
        "google_speed_kmh": "google_speed",
    }

    for old_col, new_col in rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})

    bus_live_candidates = [
        "bus_speed",
        "live_bus_speed",
        "live_speed",
        "stib_speed",
    ]
    bus_completed_candidates = [
        "completed_bus_speed",
        "final_completed_speed",
        "snapshot_speed",
        "observed_speed",
        "speed",
    ]

    if "bus_speed_live" not in df.columns:
        for col in bus_live_candidates:
            if col in df.columns:
                df["bus_speed_live"] = df[col]
                break

    if "bus_speed_completed" not in df.columns:
        for col in bus_completed_candidates:
            if col in df.columns:
                df["bus_speed_completed"] = df[col]
                break

    if "est_speed" not in df.columns:
        for col in ["estimated_speed", "prediction", "pred_speed"]:
            if col in df.columns:
                df["est_speed"] = df[col]
                break

    for col in ["bus_speed_live", "bus_speed_completed", "est_speed", "google_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "segment_id" not in df.columns:
        df["segment_id"] = ""

    if "segment_name" not in df.columns:
        df["segment_name"] = df["segment_id"].astype(str)

    if "bus_lines" not in df.columns:
        df["bus_lines"] = ""

    return df
    
def render_brussels_results_visualisation(results_df: pd.DataFrame) -> None:
    st.markdown("### Visualisation")

    df = _prepare_results_df(results_df)
    if df.empty:
        st.info("No results available for visualisation.")
        return

    render_summary_metrics(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Coverage",
            "Estimation vs Google",
            "Error Analysis",
            "Distribution",
            "Preview",
        ]
    )

    with tab1:
        render_coverage_chart(df)

    with tab2:
        render_estimation_vs_google_scatter(df)

    with tab3:
        render_estimation_google_error_chart(df)

    with tab4:
        render_speed_distribution_chart(df)

    with tab5:
        st.dataframe(df, use_container_width=True)


def render_summary_metrics(df: pd.DataFrame) -> None:
    google_overlap = 0
    mean_abs_error = None

    if {"est_speed", "google_speed"}.issubset(df.columns):
        overlap_df = df.dropna(subset=["est_speed", "google_speed"]).copy()
        google_overlap = len(overlap_df)
        if not overlap_df.empty:
            mean_abs_error = (overlap_df["est_speed"] - overlap_df["google_speed"]).abs().mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total rows", len(df))
    col2.metric("Bus available", int(df["bus_speed"].notna().sum()) if "bus_speed" in df.columns else 0)
    col3.metric("Google available", int(df["google_speed"].notna().sum()) if "google_speed" in df.columns else 0)
    col4.metric(
        "Mean |Est-Google|",
        f"{mean_abs_error:.2f} km/h" if mean_abs_error is not None else "N/A",
    )

    st.caption(f"Rows with both Estimation and Google: {google_overlap}")


def render_coverage_chart(df: pd.DataFrame) -> None:
    coverage_df = pd.DataFrame(
        {
            "source": ["Bus", "Estimation", "Google"],
            "count": [
                int(df["bus_speed"].notna().sum()) if "bus_speed" in df.columns else 0,
                int(df["est_speed"].notna().sum()) if "est_speed" in df.columns else 0,
                int(df["google_speed"].notna().sum()) if "google_speed" in df.columns else 0,
            ],
        }
    )

    chart = (
        alt.Chart(coverage_df)
        .mark_bar()
        .encode(
            x=alt.X("source:N", title="Source"),
            y=alt.Y("count:Q", title="Available rows"),
            tooltip=["source", "count"],
        )
        .properties(height=320, title="Coverage by source")
    )

    st.altair_chart(chart, use_container_width=True)


def render_estimation_vs_google_scatter(df: pd.DataFrame) -> None:
    required_cols = {"est_speed", "google_speed", "segment_id", "segment_name"}
    if not required_cols.issubset(df.columns):
        st.info("Required columns for Estimation vs Google scatter are not available.")
        return

    plot_df = df.dropna(subset=["est_speed", "google_speed"]).copy()
    if plot_df.empty:
        st.info("No overlapping Estimation and Google data available.")
        return

    min_val = min(plot_df["google_speed"].min(), plot_df["est_speed"].min())
    max_val = max(plot_df["google_speed"].max(), plot_df["est_speed"].max())

    diagonal_df = pd.DataFrame(
        {
            "x": [min_val, max_val],
            "y": [min_val, max_val],
        }
    )

    points = (
        alt.Chart(plot_df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("google_speed:Q", title="Google speed (km/h)"),
            y=alt.Y("est_speed:Q", title="Estimated speed (km/h)"),
            tooltip=[
                alt.Tooltip("segment_id:N", title="Segment ID"),
                alt.Tooltip("segment_name:N", title="Segment"),
                alt.Tooltip("google_speed:Q", title="Google", format=".2f"),
                alt.Tooltip("est_speed:Q", title="Estimation", format=".2f"),
            ],
        )
    )

    diagonal = (
        alt.Chart(diagonal_df)
        .mark_line()
        .encode(
            x="x:Q",
            y="y:Q",
        )
    )

    chart = (points + diagonal).properties(
        height=380,
        title="Estimation vs Google",
    )

    st.altair_chart(chart, use_container_width=True)


def render_estimation_google_error_chart(df: pd.DataFrame) -> None:
    required_cols = {"est_speed", "google_speed", "segment_id", "segment_name"}
    if not required_cols.issubset(df.columns):
        st.info("Required columns for error analysis are not available.")
        return

    plot_df = df.dropna(subset=["est_speed", "google_speed"]).copy()
    if plot_df.empty:
        st.info("No overlapping Estimation and Google data available for error analysis.")
        return

    plot_df["abs_error"] = (plot_df["est_speed"] - plot_df["google_speed"]).abs()
    plot_df["signed_error"] = plot_df["est_speed"] - plot_df["google_speed"]
    plot_df["label"] = plot_df["segment_name"].fillna(plot_df["segment_id"].astype(str))

    plot_df = plot_df.sort_values("abs_error", ascending=False).head(20)

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("abs_error:Q", title="Absolute error |Est - Google| (km/h)"),
            y=alt.Y("label:N", sort="-x", title="Segment"),
            tooltip=[
                alt.Tooltip("segment_id:N", title="Segment ID"),
                alt.Tooltip("segment_name:N", title="Segment"),
                alt.Tooltip("google_speed:Q", title="Google", format=".2f"),
                alt.Tooltip("est_speed:Q", title="Estimation", format=".2f"),
                alt.Tooltip("signed_error:Q", title="Signed error", format=".2f"),
                alt.Tooltip("abs_error:Q", title="Absolute error", format=".2f"),
            ],
        )
        .properties(height=520, title="Top Estimation vs Google errors")
    )

    st.altair_chart(chart, use_container_width=True)


def render_speed_distribution_chart(df: pd.DataFrame) -> None:
    parts = []

    for source_name, col in [
        ("Bus", "bus_speed"),
        ("Estimation", "est_speed"),
        ("Google", "google_speed"),
    ]:
        if col in df.columns:
            tmp = df[[col]].copy()
            tmp = tmp.rename(columns={col: "speed"})
            tmp["source"] = source_name
            parts.append(tmp)

    if not parts:
        st.info("No speed columns available for distribution chart.")
        return

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["speed"])

    if long_df.empty:
        st.info("No non-null speed values available for distribution chart.")
        return

    chart = (
        alt.Chart(long_df)
        .mark_boxplot()
        .encode(
            x=alt.X("source:N", title="Source"),
            y=alt.Y("speed:Q", title="Speed (km/h)"),
            tooltip=[alt.Tooltip("source:N", title="Source")],
        )
        .properties(height=380, title="Speed distribution by source")
    )

    st.altair_chart(chart, use_container_width=True)
