"""Brussels "Performance Analysis": compares STIB, the Traffic Estimation Model, TabPFN and Google.

The data and statistics come from comparison.py; this file only draws them.
Sources always appear in the order STIB, Traffic Estimation Model, TabPFN,
Google, each in its own fixed colour. Google is the reference when the run has
enough of it (see comparison.MIN_REFERENCE_ROWS): error metrics and error
charts. Otherwise the sources are only compared with each other, and nothing
says which is more accurate. TabPFN appears only when its prediction is shown
on the maps.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from cities.brussels import comparison as cmp

# One fixed colour per source, so it never changes when TabPFN is switched off.
SOURCE_COLORS = {"stib": "#1baf7a", "primary": "#2a78d6", "tabpfn": "#eb6834", "google": "#6b6a66"}
NEUTRAL_COLOR = "#5b6b7a"
HIGHEST, LOWEST = "Highest error", "Lowest error"


def render_brussels_results_visualisation(
    results_df: pd.DataFrame, tabpfn_speeds: dict[str, float] | None = None, key: str = ""
) -> None:
    """results_df: the RUN's enriched snapshot; tabpfn_speeds: TabPFN speeds shown on the maps, or None.

    key: unique per RUN result (a RUN draws the previous result first), for the widgets' ids.
    """
    frame = cmp.build_frame(results_df, tabpfn_speeds)
    if frame.empty:
        st.info("No results available for visualisation.")
        return

    errors = cmp.reference_errors(frame)
    reference = cmp.has_reference(frame)
    render_summary_metrics(frame, errors, reference, key)

    if reference:
        panels = [
            ("Street-Level Error", lambda: render_street_error_chart(errors, key)),
            ("Estimated vs Google", lambda: render_source_scatter(_reference_pairs(errors))),
            ("Absolute Error Distribution", lambda: render_error_distribution(errors)),
            ("Speed Distribution", lambda: render_speed_distribution_chart(frame)),
        ]
    else:
        panels = [
            ("Largest Speed Differences", lambda: render_street_spread_chart(frame)),
            ("Source Comparison", lambda: render_source_comparison(frame)),
            ("Speed Distribution", lambda: render_speed_distribution_chart(frame)),
        ]
    for tab, (_, render) in zip(st.tabs([title for title, _ in panels]), panels):
        with tab:
            render()

    with st.expander("Data Preview"):
        st.dataframe(frame.rename(columns={key: cmp.source_label(key) for key in cmp.SOURCES}), width="stretch")


def render_summary_metrics(frame: pd.DataFrame, errors: pd.DataFrame, reference: bool, key: str) -> None:
    """Coverage, then one MAE per source; sample sizes are in the tooltips. (Typography: core.styles.)"""
    total = len(frame)
    google_rows = int(frame["google"].notna().sum())
    with st.container(key=f"bru_summary_{key}"):
        columns = st.columns(3)
        columns[0].metric("Segments", f"{total:,}")
        columns[1].metric(
            "With Google speed",
            f"{google_rows:,} ({100 * google_rows / total:.0f}%)",
            help="Errors are computed only on these segments, not on the full network.",
        )
        if not reference:
            st.caption(
                f"Fewer than {cmp.MIN_REFERENCE_ROWS} Google segments: the sources are only compared with each other, "
                "which does not show which one is more accurate."
            )
            return
        st.caption("Mean absolute error vs Google (km/h), on segments with Google speed")
        summary = cmp.error_summary(errors)
        for column, (source, row) in zip(st.columns(3), summary.iterrows()):
            sample = (
                "its own sample, not directly comparable with the models"
                if source == "stib"
                else "same segments for all models"
            )
            column.metric(cmp.source_label(source), f"{row['MAE']:.2f}", help=f"{int(row['n']):,} segments ({sample})")


def _title(text: str, subtitle: str | list[str] | None = None) -> alt.TitleParams:
    return alt.TitleParams(text, subtitle=subtitle, anchor="start", fontSize=14, subtitleFontSize=12, subtitleColor="#6b7280")


def _show(chart: alt.TopLevelMixin) -> None:
    """Draw a chart with the same fonts and no frame everywhere."""
    chart = chart.configure_axis(labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
    chart = chart.configure_legend(labelFontSize=11, titleFontSize=12).configure_view(stroke=None)
    st.altair_chart(chart, width="stretch")


def _color(keys: list[str], field: str = "label", labels: list[str] | None = None, legend: bool = True) -> alt.Color:
    """Source colours; labels are the values of field, in the order of keys."""
    labels = labels or [cmp.source_label(key) for key in keys]
    return alt.Color(
        f"{field}:N",
        scale=alt.Scale(domain=labels, range=[SOURCE_COLORS[key] for key in keys]),
        legend=alt.Legend(title=None, orient="top", direction="horizontal") if legend else None,
    )


def _choice(name: str, options: list[str], key: str, label: str) -> str:
    """A selector whose choice survives a new RUN: its widget id changes with every result (key),
    so the choice is remembered in session state under name."""
    saved = st.session_state.get(name)
    default = saved if saved in options else options[0]
    widget = f"{name}_{key}"

    def remember() -> None:
        if st.session_state[widget]:  # clicking the selected option again clears it: keep the old choice
            st.session_state[name] = st.session_state[widget]

    return st.segmented_control(label, options, default=default, key=widget, on_change=remember) or default


def render_street_error_chart(errors: pd.DataFrame, key: str) -> None:
    """Mean absolute error vs Google per street (segment_name groups the segments of a street).

    The 15 streets with the highest (or lowest) error of the chosen model, with every available source
    beside it. The ranking model is chosen here when there are two (TabPFN shown), else it is the
    Traffic Estimation Model.
    """
    models = [k for k in cmp.MODELS if k in errors["source"].values]
    if not models:
        st.info("No model estimates overlap with Google speeds.")
        return
    left, right, _ = st.columns([5, 4, 1])
    with left:
        rank_label = (
            _choice("bru_street_rank", [cmp.source_label(k) for k in models], key, "Ranked by")
            if len(models) > 1
            else cmp.source_label(models[0])
        )
    with right:
        view = _choice("bru_street_view", [HIGHEST, LOWEST], key, "Show")
    rank_by = next(k for k in models if cmp.source_label(k) == rank_label)
    highest = view == HIGHEST

    street = cmp.street_errors(errors, rank_by, highest)
    street["label"] = street["source"].map(cmp.source_label)
    street["street_label"] = street["street"] + " (" + street["rank_segments"].astype(str) + ")"
    order = (
        street.drop_duplicates("street_label")
        .sort_values("rank_mae", ascending=not highest)["street_label"]
        .tolist()
    )
    keys = [k for k in cmp.SOURCES if k in street["source"].values]

    chart = (
        alt.Chart(street)
        .mark_bar()
        .encode(
            x=alt.X("mae:Q", title="Mean absolute error vs Google (km/h)", scale=alt.Scale(domainMin=0)),
            y=alt.Y("street_label:N", sort=order, title=None),
            yOffset=alt.YOffset("label:N", sort=[cmp.source_label(k) for k in keys]),
            color=_color(keys),
            tooltip=[
                alt.Tooltip("street:N", title="Street"),
                alt.Tooltip("label:N", title="Source"),
                alt.Tooltip("mae:Q", title="Mean absolute error (km/h)", format=".2f"),
                alt.Tooltip("segments:Q", title="Segments in this street"),
            ],
        )
        .properties(
            height=max(300, 13 * len(order) * len(keys)),
            title=_title(
                f"{len(order)} streets with the {'highest' if highest else 'lowest'} error, ranked by {rank_label}",
                "Error averaged over each street's segments with Google speed; their number is in brackets",
            ),
        )
    )
    _show(chart)


def render_street_spread_chart(frame: pd.DataFrame) -> None:
    """No Google: the streets where the available sources differ most."""
    street = cmp.street_spread(frame)
    if street.empty:
        st.info("At least two sources with overlapping segments are needed to compare streets.")
        return
    street["street_label"] = street["street"] + " (" + street["segments"].astype(str) + ")"
    chart = (
        alt.Chart(street)
        .mark_bar(color=NEUTRAL_COLOR)
        .encode(
            x=alt.X("spread:Q", title="Mean speed difference between sources (km/h)", scale=alt.Scale(domainMin=0)),
            y=alt.Y("street_label:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
            tooltip=[
                alt.Tooltip("street:N", title="Street"),
                alt.Tooltip("spread:Q", title="Mean difference (km/h)", format=".2f"),
                alt.Tooltip("segments:Q", title="Segments"),
            ],
        )
        .properties(
            height=28 * len(street),
            title=_title(
                f"{len(street)} streets with the largest speed differences between sources",
                "Highest minus lowest source speed, averaged over the street's segments; their number is in brackets",
            ),
        )
    )
    _show(chart)


def _reference_pairs(errors: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Google (x) against each source (y), each on its own sample."""
    return {
        ("google", key): sample[["segment_id", "segment_name"]].assign(x=sample["google"], y=sample["speed"])
        for key, sample in errors.groupby("source", sort=False)
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
        x_title, y_title = f"{cmp.source_label(x_key)} (km/h)", f"{cmp.source_label(y_key)} (km/h)"
        points = (
            alt.Chart(pair)
            .mark_circle(size=50, opacity=0.7, color=SOURCE_COLORS[y_key])
            .encode(
                x=alt.X("x:Q", title=x_title, scale=axis),
                y=alt.Y("y:Q", title=y_title, scale=axis),
                tooltip=[
                    alt.Tooltip("segment_id:N", title="Segment ID"),
                    alt.Tooltip("segment_name:N", title="Street"),
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
            _show(
                (points + diagonal).properties(
                    height=320,
                    title=_title(cmp.source_label(y_key), [f"vs {cmp.source_label(x_key)}", f"{len(pair):,} segments"]),
                )
            )


def render_source_comparison(frame: pd.DataFrame) -> None:
    """No Google: pairwise scatter of the available sources and how far apart they are."""
    pairs = cmp.source_pairs(frame)
    if not pairs:
        st.info("At least two sources with overlapping segments are needed for a comparison.")
        return
    render_source_scatter(pairs)
    summary = cmp.difference_summary(pairs).rename(
        columns={"n": "Segments", "Mean |difference|": "Mean |difference| (km/h)", "Mean difference": "Mean difference (km/h)"}
    )
    st.dataframe(summary.round(2), width="stretch")
    st.caption("Differences between sources, not errors: without Google neither is known to be right. Mean difference = first minus second.")


def render_error_distribution(errors: pd.DataFrame) -> None:
    """Share of each source's segments per 1 km/h bin of absolute error vs Google, and the error statistics."""
    keys = [key for key in cmp.SOURCES if key in errors["source"].values]
    bins = errors["abs_error"].floordiv(1).astype(int)
    shares = errors.assign(bin=bins).groupby(["source", "bin"]).size() / errors.groupby("source").size()
    y_scale = alt.Scale(domain=[0, shares.max()], nice=True)  # shared by the panels
    max_bin = max(1, int(errors["abs_error"].max()) + 1)

    for column, key in zip(st.columns(len(keys)), keys):
        sample = errors[errors["source"] == key]
        chart = (
            alt.Chart(sample)
            .transform_joinaggregate(total="count()")
            .transform_calculate(share="1 / datum.total")
            .mark_bar(color=SOURCE_COLORS[key])
            .encode(
                x=alt.X("abs_error:Q", bin=alt.Bin(step=1, extent=[0, max_bin]), title="Absolute error vs Google (km/h)"),
                y=alt.Y("sum(share):Q", scale=y_scale, axis=alt.Axis(format="%"), title="Share of segments"),
                tooltip=[alt.Tooltip("count():Q", title="Segments")],
            )
            .properties(height=300, title=_title(cmp.source_label(key), f"{len(sample):,} segments"))
        )
        with column:
            _show(chart)

    summary = cmp.error_summary(errors)
    summary.index = [cmp.source_label(key) for key in summary.index]
    st.dataframe(
        summary.rename(
            columns={"n": "Segments", "MAE": "MAE (km/h)", "Median AE": "Median AE (km/h)", "RMSE": "RMSE (km/h)", "Bias": "Bias (km/h)"}
        ).round(2),
        width="stretch",
    )
    st.caption("Bias = mean of (source − Google). STIB is evaluated on its own segments, the models on the same ones.")


def render_speed_distribution_chart(frame: pd.DataFrame) -> None:
    """Boxplot of each source's speeds over all the segments it covers (n under the name)."""
    keys = cmp.available_sources(frame)
    long_df = pd.concat(
        [
            frame[[key]].dropna().set_axis(["speed"], axis=1).assign(source=f"{cmp.source_label(key)}|{frame[key].notna().sum():,} segments")
            for key in keys
        ],
        ignore_index=True,
    )
    labels = long_df["source"].drop_duplicates().tolist()
    chart = (
        alt.Chart(long_df)
        .mark_boxplot()
        .encode(
            x=alt.X("source:N", title=None, sort=labels, axis=alt.Axis(labelAngle=0, labelLimit=0, labelExpr="split(datum.label, '|')")),
            y=alt.Y("speed:Q", title="Speed (km/h)"),
            color=_color(keys, field="source", labels=labels, legend=False),
        )
        .properties(
            height=380,
            title=_title("Speed distribution by source", "Each source over all the segments it covers; Google only covers the selected ones"),
        )
    )
    _show(chart)
