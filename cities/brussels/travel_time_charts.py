"""Brussels "Travel Time": the travel time of road segments and connected sections, per source.

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

NO_VALUE_COLOR = "#e5e7eb"
SELECTION = "selection"


def render_travel_time(results_df: pd.DataFrame, tabpfn_speeds: dict[str, float] | None, selection: tuple, key: str) -> None:
    """results_df: the RUN's enriched snapshot; tabpfn_speeds: shown TabPFN speeds or None;
    selection: the RUN's (segment names, bus ids); key: unique per RUN result (widget ids)."""
    segments = tt.segment_table()
    scope = _choose_scope(tt.bus_lines(segments), bool(selection and any(selection)), key)
    scoped = tt.scope_segments(segments, selection, None if scope == SELECTION else scope)
    if scoped.empty:
        st.info("No segments selected.")
        return
    if len(scoped) > tt.MAX_SEGMENTS:
        st.info(f"{len(scoped):,} segments are too many to draw. Select fewer segments, or a single bus line.")
        return

    table = tt.build_table(tt.order_sections(scoped), results_df, tabpfn_speeds)
    sources = table.attrs["sources"]
    if not sources:
        st.info("No speeds are available for these segments.")
        return
    totals = tt.totals(table)
    sections = int(table["section"].max())

    st.latex(r"t = 3.6 \cdot \frac{L}{v} \qquad T = \sum t")
    st.caption(
        "t: segment travel time (s) · L: segment length (m) · v: speed (km/h) · T: total (s). "
        "For Google, t is its reported duration (s). Minutes = seconds / 60."
    )
    basis = table[table["in_basis"]]
    covered = f"{len(basis)} of {_count(len(table), 'segment')} ({100 * basis['length_m'].sum() / table['length_m'].sum():.0f}% of length)"
    kind = "1 connected section" if sections == 1 else f"{sections} separate sections"
    st.caption(
        f"{_count(len(table), 'segment')} · {table['length_m'].sum() / 1000:.1f} km · {kind}. "
        f"Totals use the {covered} that have a value from every source here; other blocks are faded."
    )
    if not len(basis):
        st.info("No segment has a value from every source, so no total is shown.")
    _strips(table, sources, totals, sections)
    st.markdown(_speed_legend(), unsafe_allow_html=True)
    if sections > 1:
        st.caption("Σ is the sum of the separate sections' times, not the time of one trip.")
        only_lines = {scope} if scope != SELECTION else set(selection[1])
        _section_table(table, sources, totals, only_lines)
    st.download_button(
        "Download segment travel times",
        data=tt.export_table(table).to_csv(index=False).encode("utf-8"),
        file_name="travel_time_segments.csv",
        mime="text/csv",
        key=f"bru_travel_csv_{key}",
    )


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" + ("" if n == 1 else "s")


def _choose_scope(lines: list[str], has_selection: bool, key: str) -> str:
    """Selection or one bus line; the choice survives a new RUN (the widget id changes with every result)."""
    codes = ([SELECTION] if has_selection else []) + lines
    saved = st.session_state.get("bru_travel_scope")
    default = saved if saved in codes else codes[0]
    widget = f"bru_travel_scope_{key}"

    def remember() -> None:
        st.session_state["bru_travel_scope"] = st.session_state[widget]

    return st.selectbox(
        "Segments",
        codes,
        index=codes.index(default),
        format_func=lambda code: "Current selection" if code == SELECTION else f"Bus line {code}",
        key=widget,
        on_change=remember,
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
                    "section": table["section"],
                    "length": table["length_m"].round().astype(int),
                    "speed": table[f"speed_{k}"].map(lambda v: "no value" if pd.isna(v) else f"{v:.1f} km/h"),
                    "time": table[f"time_{k}"].map(tt.format_time),
                }
            )
            for k in sources
        ],
        ignore_index=True,
    )
    x_end = float(table["x1"].max())
    y = alt.Y("source:N", sort=labels, title=None, axis=alt.Axis(ticks=False, domain=False, labelFontSize=12, labelLimit=170), scale=alt.Scale(paddingInner=0.15))
    rects = (
        alt.Chart(blocks)
        .mark_rect(stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "x0:Q",
                title="Distance along the segments (km; gaps between sections are not to scale)",
                scale=alt.Scale(domain=[0, x_end * 1.25], nice=False),
                axis=alt.Axis(grid=False),
            ),
            x2="x1:Q",
            y=y,
            color=alt.Color("color:N", scale=None),
            opacity=alt.condition("datum.in_basis", alt.value(1), alt.value(0.3)),
            tooltip=[
                alt.Tooltip("street:N", title="Street"),
                alt.Tooltip("section:Q", title="Section"),
                alt.Tooltip("length:Q", title="Length (m)"),
                alt.Tooltip("speed:N", title="Speed"),
                alt.Tooltip("time:N", title="Travel time"),
            ],
        )
    )
    prefix = "Σ " if sections > 1 else ""
    ends = pd.DataFrame({"source": labels, "x": x_end, "label": [prefix + tt.format_time(totals.get(k, float("nan"))) for k in sources]})
    text = (
        alt.Chart(ends)
        .mark_text(align="left", dx=6, fontSize=12, fontWeight="bold", color="#1F2937")
        .encode(x=alt.X("x:Q"), y=alt.Y("source:N", sort=labels), text="label:N")
    )
    layers = rects + text
    if sections > 1:  # section numbers above the first strip
        centers = table.groupby("section").agg(x0=("x0", "min"), x1=("x1", "max")).reset_index()
        numbers = (
            alt.Chart(centers.assign(x=(centers["x0"] + centers["x1"]) / 2))
            .mark_text(baseline="bottom", fontSize=11, color="#6b7280", clip=False)
            .encode(x="x:Q", y=alt.value(-2), text="section:N")
        )
        layers = layers + numbers
    _show(layers.properties(height=alt.Step(34), title=_title("Travel time per segment", "Block width = segment length · colour = speed · number at the end = total travel time")))


def _speed_legend() -> str:
    """The maps' speed scale, as one small line."""
    items = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:12px;white-space:nowrap;'>"
        f"<span style='width:14px;height:10px;border-radius:2px;background:{get_speed_color(low)};'></span>{label}</span>"
        for low, label in LEGEND_RANGES
    )
    return f"<div style='font-size:0.8rem;color:#6b7280;'>Speed (km/h): {items}<span>grey: no value</span></div>"


def _section_table(table: pd.DataFrame, sources: list[str], totals: dict[str, float], only_lines: set[str]) -> None:
    """Each section's own total, then their sum."""
    summary = tt.section_summary(table, only_lines)
    total_row = {"Section": "Σ sum of sections (not one trip)", "Segments": len(table), "Length (km)": table["length_m"].sum() / 1000}
    total_row.update({k: totals[k] for k in sources})
    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)
    for k in sources:  # seconds first, then minutes
        summary[k] = summary[k].map(tt.format_time)
    st.dataframe(
        summary.rename(columns={k: cmp.source_label(k) for k in sources}).round({"Length (km)": 1}).astype({"Segments": int}),
        width="stretch",
        hide_index=True,
    )
