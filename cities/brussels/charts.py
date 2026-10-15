"""Brussels "Performance Analysis": compares STIB, the Primary Model, TabPFN and Google.

The data and statistics come from comparison.py; this file only draws them.
Google is the reference when the run has enough of it (see
comparison.MIN_REFERENCE_ROWS): error metrics and error charts. Otherwise the
sources are only compared with each other, and nothing says which is more
accurate. TabPFN appears only when its prediction is shown on the maps.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from cities.brussels import comparison as cmp

# One fixed colour per source, so it never changes when TabPFN is switched off.
SOURCE_COLORS = {"stib": "#1baf7a", "primary": "#2a78d6", "tabpfn": "#eb6834", "google": "#6b6a66"}
NEUTRAL_COLOR = "#5b6b7a"


def render_brussels_results_visualisation(results_df: pd.DataFrame, tabpfn_speeds: dict[str, float] | None = None) -> None:
    """results_df: the RUN's enriched snapshot; tabpfn_speeds: TabPFN speeds shown on the maps, or None."""
    frame = cmp.build_frame(results_df, tabpfn_speeds)
    if frame.empty:
        st.info("No results available for visualisation.")
        return

    errors = cmp.reference_errors(frame)
    reference = cmp.has_reference(frame)
    render_summary_metrics(frame, errors, reference)

    if reference:
        panels = [
            ("Street-Level Error", lambda: render_street_error_chart(errors)),
            ("Estimated vs Google", lambda: render_source_scatter(_reference_pairs(errors))),
            ("Absolute Error Distribution", lambda: render_error_distribution(errors)),
            ("Speed Distribution", lambda: render_speed_distribution_chart(frame)),
        ]
    else:
        panels = [
            ("Street-Level Disagreement", lambda: render_street_spread_chart(frame)),
            ("Source Comparison", lambda: render_source_comparison(frame)),
            ("Speed Distribution", lambda: render_speed_distribution_chart(frame)),
        ]
    for tab, (_, render) in zip(st.tabs([title for title, _ in panels]), panels):
        with tab:
            render()

    with st.expander("Data Preview"):
        st.dataframe(frame.rename(columns={key: cmp.source_label(key) for key in cmp.SOURCES}), width="stretch")


def render_summary_metrics(frame: pd.DataFrame, errors: pd.DataFrame, reference: bool) -> None:
    total = len(frame)
    google_rows = int(frame["google"].notna().sum())

    summary = cmp.error_summary(errors) if reference else pd.DataFrame()
    columns = st.columns(2 + max(len(summary), 1))
    columns[0].metric("Total Segments", f"{total:,}")
    columns[1].metric("Segments with Google Speed", f"{google_rows:,} ({100 * google_rows / total:.1f}%)")
    if reference:
        for column, (key, row) in zip(columns[2:], summary.iterrows()):
            column.metric(f"MAE {cmp.source_label(key)}", f"{row['MAE']:.2f} km/h", help=f"{int(row['n']):,} segments with Google")
        models = [key for key in cmp.MODELS if key in summary.index]
        samples = []
        if models:
            names = " and ".join(cmp.source_label(key) for key in models)
            samples.append(f"{names}: {int(summary.loc[models[0], 'n']):,} segments (same for all)")
        if "stib" in summary.index:
            samples.append(f"STIB: {int(summary.loc['stib', 'n']):,} segments (its own sample)")
        st.caption(
            f"Errors use only the segments with Google speeds, not the full network. Samples — {'; '.join(samples)}. "
            "Results on different samples are not directly comparable."
        )
    else:
        columns[2].metric("Mean Absolute Error", "N/A")
        st.caption(
            f"Fewer than {cmp.MIN_REFERENCE_ROWS} Google segments in this run, so no error is computed. "
            "The charts show how the available sources differ; this does not show which is more accurate."
        )


def _color(keys: list[str], field: str = "label", labels: list[str] | None = None, legend: bool = True) -> alt.Color:
    """Source colours; labels are the values of field, in the order of keys."""
    labels = labels or [cmp.source_label(key) for key in keys]
    return alt.Color(
        f"{field}:N",
        scale=alt.Scale(domain=labels, range=[SOURCE_COLORS[key] for key in keys]),
        legend=alt.Legend(title="Source") if legend else None,
    )


def render_street_error_chart(errors: pd.DataFrame) -> None:
    """Mean absolute error vs Google per street (segment_name groups the segments of a street)."""
    street = cmp.street_errors(errors)
    if street.empty:
        st.info("No model estimates overlap with Google speeds.")
        return
    street["label"] = street["source"].map(cmp.source_label)
    order = street.groupby("street")["mae"].mean().sort_values(ascending=False).index.tolist()
    keys = [key for key in cmp.MODELS if key in street["source"].values]

    chart = (
        alt.Chart(street)
        .mark_bar()
        .encode(
            x=alt.X("mae:Q", title="Mean Absolute Error vs Google (km/h)", scale=alt.Scale(domainMin=0)),
            y=alt.Y("street:N", sort=order, title="Street"),
            yOffset=alt.YOffset("label:N", sort=[cmp.source_label(key) for key in keys]),
            color=_color(keys),
            tooltip=[
                alt.Tooltip("street:N", title="Street"),
                alt.Tooltip("label:N", title="Source"),
                alt.Tooltip("mae:Q", title="Mean Absolute Error (km/h)", format=".2f"),
                alt.Tooltip("segments:Q", title="Segments"),
            ],
        )
        .properties(
            height=max(280, 22 * len(order) * len(keys)),
            title=f"Top {len(order)} streets by mean model error (segments with Google speeds only)",
        )
    )
    st.altair_chart(chart, width="stretch")


def render_street_spread_chart(frame: pd.DataFrame) -> None:
    """No Google: the streets where the available sources differ most."""
    street = cmp.street_spread(frame)
    if street.empty:
        st.info("At least two sources with overlapping segments are needed to compare streets.")
        return
    chart = (
        alt.Chart(street)
        .mark_bar(color=NEUTRAL_COLOR)
        .encode(
            x=alt.X("spread:Q", title="Mean spread between sources (km/h)", scale=alt.Scale(domainMin=0)),
            y=alt.Y("street:N", sort="-x", title="Street", axis=alt.Axis(labelOverlap=False)),
            tooltip=[
                alt.Tooltip("street:N", title="Street"),
                alt.Tooltip("spread:Q", title="Mean spread (km/h)", format=".2f"),
                alt.Tooltip("segments:Q", title="Segments"),
            ],
        )
        .properties(
            height=28 * len(street),
            title=f"Top {len(street)} streets where the sources differ most (highest - lowest speed)",
        )
    )
    st.altair_chart(chart, width="stretch")


def _reference_pairs(errors: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Google (x) against each source (y), each on its own sample."""
    return {
        ("google", key): sample[["segment_id", "segment_name"]].assign(x=sample["google"], y=sample["speed"])
        for key, sample in errors.groupby("source")
    }


def render_source_scatter(pairs: dict[tuple[str, str], pd.DataFrame]) -> None:
    """One scatter per pair of sources on shared axes with a y = x line."""
    pairs = {keys: pair for keys, pair in pairs.items() if not pair.empty}
    if not pairs:
        st.info("No overlapping speeds are available to compare.")
        return
    low = max(0.0, min(min(pair["x"].min(), pair["y"].min()) for pair in pairs.values()))
    high = max(max(pair["x"].max(), pair["y"].max()) for pair in pairs.values())
    axis = alt.Scale(domain=[low, high], nice=True)

    for column, ((x_key, y_key), pair) in zip(st.columns(len(pairs)), pairs.items()):
        x_title, y_title = f"{cmp.source_label(x_key)} Speed (km/h)", f"{cmp.source_label(y_key)} Speed (km/h)"
        points = (
            alt.Chart(pair)
            .mark_circle(size=60, opacity=0.7, color=SOURCE_COLORS[y_key])
            .encode(
                x=alt.X("x:Q", title=x_title, scale=axis),
                y=alt.Y("y:Q", title=y_title, scale=axis),
                tooltip=[
                    alt.Tooltip("segment_id:N", title="Segment ID"),
                    alt.Tooltip("segment_name:N", title="Street Name"),
                    alt.Tooltip("x:Q", title=x_title, format=".2f"),
                    alt.Tooltip("y:Q", title=y_title, format=".2f"),
                ],
            )
        )
        diagonal = (
            alt.Chart(pd.DataFrame({"v": [low, high]}))
            .mark_line(color="#9aa0a6", strokeDash=[4, 4])
            .encode(x=alt.X("v:Q", title=x_title), y=alt.Y("v:Q", title=y_title))
        )
        with column:
            st.altair_chart(
                (points + diagonal).properties(height=340, title=f"{cmp.source_label(y_key)} vs {cmp.source_label(x_key)} (n = {len(pair):,})"),
                width="stretch",
            )


def render_source_comparison(frame: pd.DataFrame) -> None:
    """No Google: pairwise scatter of the available sources and how far apart they are."""
    pairs = cmp.source_pairs(frame)
    if not pairs:
        st.info("At least two sources with overlapping segments are needed for a comparison.")
        return
    render_source_scatter(pairs)
    st.caption("Differences between sources, not errors: without Google reference neither source is known to be right.")
    st.dataframe(cmp.difference_summary(pairs).round(2), width="stretch")


def render_error_distribution(errors: pd.DataFrame) -> None:
    """Share of each source's segments per 1 km/h bin of absolute error vs Google (one panel per source), and the error statistics."""
    keys = [key for key in cmp.SOURCES if key in errors["source"].values]
    plot_df = errors.assign(label=errors["source"].map(cmp.source_label))
    max_bin = max(1, int(plot_df["abs_error"].max()) + 1)

    chart = (
        alt.Chart(plot_df)
        .transform_joinaggregate(total="count()", groupby=["label"])
        .transform_calculate(share="1 / datum.total")
        .mark_bar()
        .encode(
            x=alt.X(
                "abs_error:Q",
                bin=alt.Bin(step=1, extent=[0, max_bin]),
                title="Absolute Error |Speed - Google Speed| (km/h)",
            ),
            y=alt.Y("sum(share):Q", axis=alt.Axis(format="%"), title="Share of the source's segments"),
            color=_color(keys, legend=False),
            column=alt.Column("label:N", title=None, sort=[cmp.source_label(key) for key in keys]),
            tooltip=[alt.Tooltip("label:N", title="Source"), alt.Tooltip("count():Q", title="Segments")],
        )
        .properties(height=320, width=280)
        .properties(title="Distribution of Absolute Error vs Google")
    )
    st.altair_chart(chart, width="stretch")

    summary = cmp.error_summary(errors)
    summary.index = [cmp.source_label(key) for key in summary.index]
    st.dataframe(
        summary.rename(columns={"n": "Segments", "MAE": "MAE (km/h)", "Median AE": "Median AE (km/h)", "RMSE": "RMSE (km/h)", "Bias": "Bias (km/h)"}).round(2),
        width="stretch",
    )
    st.caption("Bias is the mean of (source - Google). STIB and the models are evaluated on different segment samples (Segments).")


def render_speed_distribution_chart(frame: pd.DataFrame) -> None:
    """Boxplot of each source's speeds over all the segments it covers (n in the label)."""
    keys = cmp.available_sources(frame)
    long_df = pd.concat(
        [
            frame[[key]].dropna().set_axis(["speed"], axis=1).assign(source=f"{cmp.source_label(key)} (n = {frame[key].notna().sum():,})")
            for key in keys
        ],
        ignore_index=True,
    )
    labels = long_df["source"].drop_duplicates().tolist()
    chart = (
        alt.Chart(long_df)
        .mark_boxplot()
        .encode(
            x=alt.X("source:N", title="Data Source", sort=labels, axis=alt.Axis(labelAngle=0, labelLimit=220)),
            y=alt.Y("speed:Q", title="Speed (km/h)"),
            color=_color(keys, field="source", labels=labels, legend=False),
        )
        .properties(height=380, title="Speed Distribution by Data Source")
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Each source covers the segments it has values for; Google usually covers only the selected ones.")
