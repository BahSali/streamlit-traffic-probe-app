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
            "Street-Level Error",
            "Estimated vs Google",
            "Absolute Error Distribution",
            "Speed Distribution",
            "Data Preview",
        ]
    )

    with tab1:
        render_estimation_google_abs_error_by_street_chart(df)

    with tab2:
        render_estimation_vs_google_scatter(df)

    with tab3:
        render_estimation_google_absolute_error_distribution(df)

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
    col1.metric("Total Rows", len(df))
    col2.metric(
        "Rows with Bus Speed",
        int(df["bus_speed"].notna().sum()) if "bus_speed" in df.columns else 0,
    )
    col3.metric(
        "Rows with Google Speed",
        int(df["google_speed"].notna().sum()) if "google_speed" in df.columns else 0,
    )
    col4.metric(
        "Mean Absolute Error",
        f"{mean_abs_error:.2f} km/h" if mean_abs_error is not None else "N/A",
    )

    st.caption(f"Rows with both Estimated and Google speed values: {google_overlap}")


def render_estimation_google_abs_error_by_street_chart(df: pd.DataFrame) -> None:
    required_cols = {"segment_name", "est_speed", "google_speed"}
    if not required_cols.issubset(df.columns):
        st.info("Required columns for the street-level error chart are not available.")
        return

    plot_df = df.dropna(subset=["segment_name", "est_speed", "google_speed"]).copy()
    if plot_df.empty:
        st.info("No overlapping Estimated and Google speed data are available.")
        return

    plot_df["absolute_error"] = (plot_df["est_speed"] - plot_df["google_speed"]).abs()

    street_error_df = (
        plot_df.groupby("segment_name", as_index=False)
        .agg(
            mean_absolute_error=("absolute_error", "mean"),
            sample_count=("absolute_error", "size"),
            mean_estimated_speed=("est_speed", "mean"),
            mean_google_speed=("google_speed", "mean"),
        )
        .sort_values("mean_absolute_error", ascending=False)
        .head(20)
    )

    chart = (
        alt.Chart(street_error_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "mean_absolute_error:Q",
                title="Mean Absolute Error |Estimated Speed - Google Speed| (km/h)",
                scale=alt.Scale(domainMin=0),
            ),
            y=alt.Y(
                "segment_name:N",
                sort="-x",
                title="Street Name",
            ),
            tooltip=[
                alt.Tooltip("segment_name:N", title="Street Name"),
                alt.Tooltip(
                    "mean_absolute_error:Q",
                    title="Mean Absolute Error (km/h)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "mean_estimated_speed:Q",
                    title="Mean Estimated Speed (km/h)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "mean_google_speed:Q",
                    title="Mean Google Speed (km/h)",
                    format=".2f",
                ),
                alt.Tooltip("sample_count:Q", title="Number of Records"),
            ],
        )
        .properties(
            height=560,
            title="Top 20 Streets with the Highest Mean Absolute Error",
        )
    )

    st.altair_chart(chart, use_container_width=True)


def render_estimation_vs_google_scatter(df: pd.DataFrame) -> None:
    required_cols = {"est_speed", "google_speed", "segment_id", "segment_name"}
    if not required_cols.issubset(df.columns):
        st.info("Required columns for the Estimated vs Google scatter plot are not available.")
        return

    plot_df = df.dropna(subset=["est_speed", "google_speed"]).copy()
    if plot_df.empty:
        st.info("No overlapping Estimated and Google speed data are available.")
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
            x=alt.X("google_speed:Q", title="Google Speed (km/h)"),
            y=alt.Y("est_speed:Q", title="Estimated Speed (km/h)"),
            tooltip=[
                alt.Tooltip("segment_id:N", title="Segment ID"),
                alt.Tooltip("segment_name:N", title="Street Name"),
                alt.Tooltip("google_speed:Q", title="Google Speed (km/h)", format=".2f"),
                alt.Tooltip("est_speed:Q", title="Estimated Speed (km/h)", format=".2f"),
            ],
        )
    )

    diagonal = (
        alt.Chart(diagonal_df)
        .mark_line()
        .encode(
            x=alt.X("x:Q", title="Google Speed (km/h)"),
            y=alt.Y("y:Q", title="Estimated Speed (km/h)"),
        )
    )

    chart = (points + diagonal).properties(
        height=400,
        title="Estimated Speed versus Google Speed",
    )

    st.altair_chart(chart, use_container_width=True)


def render_estimation_google_absolute_error_distribution(df: pd.DataFrame) -> None:
    required_cols = {"est_speed", "google_speed"}
    if not required_cols.issubset(df.columns):
        st.info("Required columns for the absolute error distribution are not available.")
        return

    plot_df = df.dropna(subset=["est_speed", "google_speed"]).copy()
    if plot_df.empty:
        st.info("No overlapping Estimated and Google speed data are available.")
        return

    plot_df["absolute_error"] = (plot_df["est_speed"] - plot_df["google_speed"]).abs()

    max_error = float(plot_df["absolute_error"].max())
    max_bin = max(1, int(max_error) + 1)

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "absolute_error:Q",
                bin=alt.Bin(step=1, extent=[0, max_bin]),
                title="Absolute Error |Estimated Speed - Google Speed| (km/h)",
            ),
            y=alt.Y(
                "count():Q",
                title="Distribution",
                axis=alt.Axis(format="d", tickMinStep=1),
            ),
            tooltip=[
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(
            height=420,
            title="Distribution of Absolute Error",
        )
    )

    st.altair_chart(chart, use_container_width=True)

    mean_abs_error = plot_df["absolute_error"].mean()
    median_abs_error = plot_df["absolute_error"].median()
    max_abs_error = plot_df["absolute_error"].max()

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Absolute Error", f"{mean_abs_error:.2f} km/h")
    c2.metric("Median Absolute Error", f"{median_abs_error:.2f} km/h")
    c3.metric("Maximum Absolute Error", f"{max_abs_error:.2f} km/h")

def render_speed_distribution_chart(df: pd.DataFrame) -> None:
    parts = []

    for source_name, col in [
        ("Bus", "bus_speed"),
        ("Estimated", "est_speed"),
        ("Google", "google_speed"),
    ]:
        if col in df.columns:
            tmp = df[[col]].copy()
            tmp = tmp.rename(columns={col: "speed"})
            tmp["source"] = source_name
            parts.append(tmp)

    if not parts:
        st.info("No speed columns are available for the distribution chart.")
        return

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["speed"])

    if long_df.empty:
        st.info("No non-null speed values are available for the distribution chart.")
        return

    chart = (
        alt.Chart(long_df)
        .mark_boxplot()
        .encode(
            x=alt.X("source:N", title="Data Source"),
            y=alt.Y("speed:Q", title="Speed (km/h)"),
            tooltip=[alt.Tooltip("source:N", title="Data Source")],
        )
        .properties(
            height=380,
            title="Speed Distribution by Data Source",
        )
    )

    st.altair_chart(chart, use_container_width=True)
