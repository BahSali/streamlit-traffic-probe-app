"""Brussels "Travel Time" strips, and the "Route-Level Error" strips that reuse them.

Travel time of road segments and connected sections, per source.

Data and rules: travel_time.py. One strip per source (Google when it has values,
Traffic Estimation Model, TabPFN when shown): each block is a segment, as wide
as its length and coloured by speed (the maps' scale); the total travel time is
at the end of the strip, in seconds and minutes.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from cities.brussels import comparison as cmp
from cities.brussels import travel_time as tt
from cities.brussels.charts import _show, _title
from core.colors import LEGEND_RANGES, get_speed_color, speed_color_or

# Absolute speed error (km/h): upper limit, label and colour of each class, green (small) to dark red (large).
ERROR_CLASSES = [
    (2, "0–2", "#00B050"), (4, "2–4", "#9ACD32"), (6, "4–6", "#FFFF00"),
    (10, "6–10", "#FFA500"), (15, "10–15", "#FF0000"), (float("inf"), "15+", "#8B0000"),
]

NO_VALUE_COLOR = "#e5e7eb"
SELECTION = "selection"


def render_travel_time(results_df: pd.DataFrame, tabpfn_speeds: dict[str, float] | None, selection: tuple, key: str) -> None:
    """results_df: the RUN's enriched snapshot; tabpfn_speeds: shown TabPFN speeds or None;
    selection: the RUN's (segment names, bus ids); key: unique per RUN result (widget ids)."""
    segments = tt.segment_table()
    codes = ([SELECTION] if selection and any(selection) else []) + tt.bus_lines(segments)
    scope = _select("bru_travel_scope", "Segments", codes, lambda code: "Current selection" if code == SELECTION else f"Bus line {code}", key)
    scoped = tt.scope_segments(segments, selection, None if scope == SELECTION else scope)
    if scoped.empty:
        st.info("No segments selected.")
        return
    if len(scoped) > tt.MAX_SEGMENTS:
        st.info(f"{len(scoped):,} segments are too many to draw. Select fewer segments, or a single bus line.")
        return

    lines = [scope] if scope != SELECTION else sorted(selection[1], key=lambda line: (len(line), line))
    ordered, routes = tt.group_segments(scoped, lines)
    table = tt.build_table(ordered, routes, results_df, tabpfn_speeds)
    sources = table.attrs["sources"]
    if not sources:
        st.info("No speeds are available for these segments.")
        return
    totals = tt.totals(table)
    sections = len(routes)
    shared = tt.shared_segments(table)

    st.latex(r"t = 3.6 \cdot \frac{L}{v} \qquad T = \sum t")
    st.caption(
        "t: segment travel time (s) · L: segment length (m) · v: speed (km/h) · T: total (s). "
        "For Google, t is its reported duration (s). Minutes = seconds / 60."
    )
    basis = table[table["in_basis"]]
    covered = f"{len(basis)} of {_count(len(table), 'segment')} ({100 * basis['length_m'].sum() / table['length_m'].sum():.0f}% of length)"
    parts = any(route["parts"] > 1 for route in routes)
    kind = "1 connected section" if sections == 1 and not parts else f"{sections} section{'s' * (sections > 1)}" + (" (some with disconnected parts)" if parts else "")
    st.caption(
        f"{_count(len(table), 'segment')} · {table['length_m'].sum() / 1000:.1f} km · {kind}. "
        f"Totals use the {covered} that have a value from every source here; other blocks are faded."
    )
    if not len(basis):
        st.info("No segment has a value from every source, so no total is shown.")
    _strips(table, sources, totals, sections)
    st.markdown(_legend("Speed (km/h)", [(get_speed_color(low), label) for low, label in LEGEND_RANGES], "no value"), unsafe_allow_html=True)
    if sections > 1 or parts:
        st.caption(
            "Σ is the sum over sections, not the time of one trip."
            + (f" {shared} shared segment{'s' * (shared > 1)} drawn once, under the first route that uses {'them' if shared > 1 else 'it'}; "
               "each route's row includes them and Σ counts them once." if shared else "")
        )
        _section_table(table, sources, totals, shared)
    st.download_button(
        "Download segment travel times",
        data=tt.export_table(table).to_csv(index=False).encode("utf-8"),
        file_name="travel_time_segments.csv",
        mime="text/csv",
        key=f"bru_travel_csv_{key}",
    )


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" + ("" if n == 1 else "s")


def _select(name: str, label: str, codes: list[str], format_func, key: str) -> str:
    """A selectbox whose choice survives a new RUN: its widget id changes with every result (key),
    so the choice is remembered in session state under name."""
    saved = st.session_state.get(name)
    widget = f"{name}_{key}"

    def remember() -> None:
        st.session_state[name] = st.session_state[widget]

    return st.selectbox(
        label, codes, index=codes.index(saved if saved in codes else codes[0]), format_func=format_func, key=widget, on_change=remember
    )


def _strips(table: pd.DataFrame, sources: list[str], totals: dict[str, float], sections: int) -> None:
    """One strip per source: a block per segment (width = length, colour = speed), total at the end."""
    labels = [cmp.source_label(k) for k in sources]
    blocks = pd.concat(
        [
            pd.DataFrame(
                {
                    "source": cmp.source_label(k),
                    "x0": table["x0"],
                    "x1": table["x1"],
                    "color": table[f"speed_{k}"].map(lambda v: speed_color_or(v, NO_VALUE_COLOR)),
                    "in_basis": table["in_basis"],
                    "street": table["segment_name"],
                    "route": table["route"],
                    "bus_lines": table["bus_lines"],
                    "length": table["length_m"].round().astype(int),
                    "speed": table[f"speed_{k}"].map(_speed_text),
                    "time": table[f"time_{k}"].map(tt.format_time),
                }
            )
            for k in sources
        ],
        ignore_index=True,
    )
    prefix = "Σ " if sections > 1 else ""
    ends = pd.DataFrame({"source": labels, "label": [prefix + tt.format_time(totals.get(k, float("nan"))) for k in sources]})
    tooltip = [
        alt.Tooltip("street:N", title="Street"),
        alt.Tooltip("route:N", title="Section"),
        alt.Tooltip("bus_lines:N", title="Bus lines"),
        alt.Tooltip("length:Q", title="Length (m)"),
        alt.Tooltip("speed:N", title="Speed"),
        alt.Tooltip("time:N", title="Travel time"),
    ]
    title = _title("Travel time per segment", "Block width = segment length · colour = speed · number at the end = total travel time")
    _draw_strips(blocks, table, labels, tooltip, title, ends)


def _speed_text(speed: float, missing: str = "no value") -> str:
    return missing if pd.isna(speed) else f"{speed:.1f} km/h"


def _draw_strips(blocks: pd.DataFrame, table: pd.DataFrame, labels: list[str], tooltip: list, title, ends: pd.DataFrame | None = None) -> None:
    """Blocks (source, x0, x1, color, in_basis + tooltip fields) as one strip per source label.

    ends: optional text (source, label) at the end of each strip; faded blocks are those with in_basis False.
    """
    x_end = float(table["x1"].max())
    y = alt.Y("source:N", sort=labels, title=None, axis=alt.Axis(ticks=False, domain=False, labelFontSize=12, labelLimit=170), scale=alt.Scale(paddingInner=0.15))
    layers = (
        alt.Chart(blocks)
        .mark_rect(stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "x0:Q",
                title="Distance along the segments (km; gaps between sections are not to scale)",
                scale=alt.Scale(domain=[0, x_end * (1.25 if ends is not None else 1.02)], nice=False),
                axis=alt.Axis(grid=False),
            ),
            x2="x1:Q",
            y=y,
            color=alt.Color("color:N", scale=None),
            opacity=alt.condition("datum.in_basis", alt.value(1), alt.value(0.3)),
            tooltip=tooltip,
        )
    )
    if ends is not None:
        layers += (
            alt.Chart(ends.assign(x=x_end))
            .mark_text(align="left", dx=6, fontSize=12, fontWeight="bold", color="#1F2937")
            .encode(x=alt.X("x:Q"), y=alt.Y("source:N", sort=labels), text="label:N")
        )
    if table["section"].nunique() > 1:  # section numbers above the first strip
        centers = table.groupby("section").agg(x0=("x0", "min"), x1=("x1", "max")).reset_index()
        layers += (
            alt.Chart(centers.assign(x=(centers["x0"] + centers["x1"]) / 2))
            .mark_text(baseline="bottom", fontSize=11, color="#6b7280", clip=False)
            .encode(x="x:Q", y=alt.value(-2), text="section:N")
        )
    _show(layers.properties(height=alt.Step(34), title=title))


def _legend(title: str, items: list[tuple[str, str]], grey: str) -> str:
    """One small legend line: (colour, label) swatches, then what grey means."""
    swatches = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:12px;white-space:nowrap;'>"
        f"<span style='width:14px;height:10px;border-radius:2px;background:{color};'></span>{label}</span>"
        for color, label in items
    )
    return f"<div style='font-size:0.8rem;color:#6b7280;'>{title}: {swatches}<span>grey: {grey}</span></div>"


def _section_table(table: pd.DataFrame, sources: list[str], totals: dict[str, float], shared: int) -> None:
    """Each section's own total, then the total over the unique segments."""
    summary = tt.route_summary(table)
    label = "Σ all segments, shared ones counted once (not one trip)" if shared else "Σ sum of sections (not one trip)"
    total_row = {"Section": label, "Segments": len(table), "Length (km)": table["length_m"].sum() / 1000}
    total_row.update({k: totals[k] for k in sources})
    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)
    for k in sources:  # seconds first, then minutes
        summary[k] = summary[k].map(tt.format_time)
    st.dataframe(
        summary.rename(columns={k: cmp.source_label(k) for k in sources}).round({"Length (km)": 1}).astype({"Segments": int}),
        width="stretch",
        hide_index=True,
    )


def _error_color(error: float) -> str:
    return NO_VALUE_COLOR if pd.isna(error) else next(color for limit, _, color in ERROR_CLASSES if error < limit)


def render_route_error(results_df: pd.DataFrame, tabpfn_speeds: dict[str, float] | None, key: str) -> None:
    """One bus line with Google data: its whole route, coloured by |model - Google| where Google has a value."""
    segments = tt.segment_table()
    lines = tt.lines_with_google(segments, results_df)
    if not lines:
        st.caption("No Google speeds in this RUN, so no route error can be shown.")
        return
    line = _select("bru_error_line", "Bus line (with Google data)", lines, lambda code: f"Bus line {code}", key)
    ordered, routes = tt.group_segments(tt.scope_segments(segments, ((), ()), line), [line])
    table = tt.build_table(ordered, routes, results_df, tabpfn_speeds)
    models = [k for k in ("primary", "tabpfn") if f"speed_{k}" in table.columns and table[f"speed_{k}"].notna().any()]

    covered = table["speed_google"].notna()
    st.caption(
        f"Google covers {covered.sum()} of {_count(len(table), 'segment')} on bus line {line} "
        f"({100 * table.loc[covered, 'length_m'].sum() / table['length_m'].sum():.0f}% of route length). "
        "The other segments stay grey: without Google there is no error."
    )
    errors = {k: tt.speed_error(table, k) for k in models}
    blocks = pd.concat(
        [
            pd.DataFrame(
                {
                    "source": cmp.source_label(k),
                    "x0": table["x0"],
                    "x1": table["x1"],
                    "color": errors[k].map(_error_color),
                    "in_basis": True,
                    "street": table["segment_name"],
                    "google": table["speed_google"].map(lambda v: _speed_text(v, "no Google data")),
                    "model": table[f"speed_{k}"].map(_speed_text),
                    "error": errors[k].map(lambda v: "—" if pd.isna(v) else f"{v:.1f} km/h"),
                }
            )
            for k in models
        ],
        ignore_index=True,
    )
    tooltip = [
        alt.Tooltip("street:N", title="Street"),
        alt.Tooltip("google:N", title="Google speed"),
        alt.Tooltip("model:N", title="Model speed"),
        alt.Tooltip("error:N", title="Absolute error"),
    ]
    title = _title("Absolute speed error vs Google", "Block width = segment length · colour = |model − Google| speed · grey = no Google data")
    _draw_strips(blocks, table, [cmp.source_label(k) for k in models], tooltip, title)
    st.markdown(_legend("Absolute error (km/h)", [(color, label) for _, label, color in ERROR_CLASSES], "no Google data"), unsafe_allow_html=True)
