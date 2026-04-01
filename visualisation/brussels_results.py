from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt


def _prepare_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    df = results_df.copy().rename(
        columns={
            "final_speed_kmh": "bus_speed",
            "estimated_speed": "est_speed",
            "google_speed_kmh": "google_speed",
            "snapshot_time": "timestamp",
        }
    )

    for col in ["bus_speed", "est_speed", "google_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def render_brussels_results_visualisation(results_df: pd.DataFrame) -> None:
    df = _prepare_results_df(results_df)
    if df.empty:
        st.info("No results available for visualisation.")
        return

    render_summary_metrics(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Street Error Bar Plot",
            "Estimation vs Google",
            "Error Distribution",
            "Distribution",
            "Preview",
        ]
    )

    with tab1:
        render_estimation_google_error_by_street_chart(df)

    with tab2:
        render_estimation_vs_google_scatter(df)

    with tab3:
        render_estimation_google_signed_error_distribution(df)

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
            mean_abs_error = (
                overlap_df["est_speed"] - overlap_df["google_speed"]
            ).abs().mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total rows", len(df))
    col2.metric(
        "Bus available",
        int(df["bus_speed"].notna().sum()) if "bus_speed" in df.columns else 0,
    )
    col3.metric(
        "Google available",
        int(df["google_speed"].notna().sum()) if "google_speed" in df.columns else 0,
    )
    col4.metric(
        "Mean |Est-Google|",
        f"{mean_abs_error:.2f} km/h" if mean_abs_error is not None else "N/A",
    )

    st.caption(f"Rows with both Estimation and Google: {google_overlap}")


def render_estimation_google_error_by_street_chart(df: pd.DataFrame) -> None:
    required_cols = {"segment_name", "est_speed", "google_speed"}
    if not required_cols.issubset(df.columns):
        st.info("Required columns for street error chart are not available.")
        return

    plot_df = df.dropna(subset=["segment_name", "est_speed", "google_speed"]).copy()
    if plot_df.empty:
        st.info("No overlapping Estimation and Google data available for street error chart.")
        return

    plot_df["abs_error"] = (plot_df["est_speed"] - plot_df["google_speed"]).abs()

    street_error_df = (
        plot_df.groupby("segment_name", as_index=False)
        .agg(
            mean_abs_error=("abs_error", "mean"),
            row_count=("abs_error", "size"),
            mean_est_speed=("est_speed", "mean"),
            mean_google_speed=("google_speed", "mean"),
        )
        .sort_values("mean_abs_error", ascending=False)
        .head(20)
    )

    chart = (
        alt.Chart(street_error_df)
        .mark_bar()
        .encode(
            x=alt.X("mean_abs_error:Q", title="Mean absolute error |Est - Google| (km/h)"),
            y=alt.Y("segment_name:N", sort="-x", title="Street name"),
            tooltip=[
                alt.Tooltip("segment_name:N", title="Street"),
                alt.Tooltip("mean_abs_error:Q", title="Mean absolute error", format=".2f"),
                alt.Tooltip("mean_est_speed:Q", title="Mean estimation speed", format=".2f"),
                alt.Tooltip("mean_google_speed:Q", title="Mean Google speed", format=".2f"),
                alt.Tooltip("row_count:Q", title="Row count"),
            ],
        )
        .properties(
            height=520,
            title="Top street errors: Estimation vs Google",
        )
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


def render_estimation_google_signed_error_distribution(df: pd.DataFrame) -> None:
    required_cols = {"est_speed", "google_speed"}
    if not required_cols.issubset(df.columns):
        st.info("Required columns for signed error distribution are not available.")
        return

    plot_df = df.dropna(subset=["est_speed", "google_speed"]).copy()
    if plot_df.empty:
        st.info("No overlapping Estimation and Google data available for error distribution.")
        return

    plot_df["signed_error"] = plot_df["est_speed"] - plot_df["google_speed"]

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "signed_error:Q",
                bin=alt.Bin(maxbins=30),
                title="Signed error (Estimation - Google) [km/h]",
            ),
            y=alt.Y("count():Q", title="Number of rows"),
            tooltip=[
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(
            height=420,
            title="Distribution of signed errors",
        )
    )

    st.altair_chart(chart, use_container_width=True)

    mean_signed_error = plot_df["signed_error"].mean()
    median_signed_error = plot_df["signed_error"].median()
    mean_abs_error = plot_df["signed_error"].abs().mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean signed error", f"{mean_signed_error:.2f} km/h")
    c2.metric("Median signed error", f"{median_signed_error:.2f} km/h")
    c3.metric("Mean absolute error", f"{mean_abs_error:.2f} km/h")


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
