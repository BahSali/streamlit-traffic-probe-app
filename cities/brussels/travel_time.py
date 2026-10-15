"""Brussels travel time: per road segment, per connected section, and per source.

No Streamlit and no drawing here (travel_time_charts.py draws it). Everything
comes from the current RUN: the speeds of the Traffic Estimation Model, TabPFN
(when shown) and Google, and Google's reported durations. No new requests.

Travel time of a segment:
- Traffic Estimation Model, TabPFN: t = 3.6 * L / v  (seconds; L in metres, v in km/h);
- Google: its reported duration in seconds.
A section's time is the sum of its segments' times.

A section is a connected, ordered run of segments of one direction (chained
on the segments' node ids). Scattered segments are separate sections, and a
total over several sections is a sum, never one trip. Each segment id is
counted once, however many bus lines serve it.

Totals use one common basis: the segments where every shown source has a
value, so the sources are compared on the same segments.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import pandas as pd

from cities.brussels.map_data import get_selected_mask, load_brussels_map

METRIC_CRS = 31370  # Belgian Lambert 72: lengths in metres
SOURCES = ("primary", "tabpfn", "google")  # display order (same as comparison.SOURCES)
TOP_CONTRIBUTORS = 8
MAX_SEGMENTS = 200  # more than this is too many blocks to draw


@lru_cache(maxsize=1)
def segment_table() -> pd.DataFrame:
    """The map's segments (static): id, name, bus lines, direction, node ids, map order and length in metres."""
    gdf = load_brussels_map()
    return pd.DataFrame(
        {
            "segment_id": gdf["id"].astype(str).str.strip(),
            "segment_name": gdf["segment_name"],
            "bus_line_list": gdf["bus_line_list"],
            "direction": gdf["direction"],
            "start_id": gdf["start_id"],
            "end_id": gdf["end_id"],
            "fid": gdf["map_fid"],
            "length_m": gdf.to_crs(METRIC_CRS).length.to_numpy(),
        }
    ).reset_index(drop=True)


def bus_lines(segments: pd.DataFrame) -> list[str]:
    return sorted({line for lines in segments["bus_line_list"] for line in lines}, key=lambda line: (len(line), line))


def scope_segments(segments: pd.DataFrame, selection: tuple, line: str | None = None) -> pd.DataFrame:
    """Unique segments of one bus line, or of the RUN's selection (segment names, bus ids)."""
    if line is not None:
        mask = segments["bus_line_list"].apply(lambda lines: line in lines)
    else:
        mask = get_selected_mask(segments, *selection)
    return segments[mask]


def order_sections(segments: pd.DataFrame) -> pd.DataFrame:
    """Segments in section order, with a 1-based `section` column.

    Two segments are chained when one ends at the node where the other starts, in the same
    direction, and neither has another way on or in; so every section is a simple ordered path.
    Sections follow the map order of their first segment.
    """
    segments = segments.sort_values("fid").reset_index(drop=True)
    starts, ends = defaultdict(list), defaultdict(list)
    for i, row in segments.iterrows():
        starts[(row["direction"], row["start_id"])].append(i)
        ends[(row["direction"], row["end_id"])].append(i)
    follows = {}
    for i, row in segments.iterrows():
        candidates = starts[(row["direction"], row["end_id"])]
        if len(candidates) == 1 and candidates[0] != i and len(ends[(row["direction"], segments.at[candidates[0], "start_id"])]) == 1:
            follows[i] = candidates[0]

    seen: set[int] = set()

    def walk(i: int | None) -> list[int]:
        chain = []
        while i is not None and i not in seen:
            seen.add(i)
            chain.append(i)
            i = follows.get(i)
        return chain

    has_previous = set(follows.values())
    chains = [walk(i) for i in range(len(segments)) if i not in has_previous]
    chains += [walk(i) for i in range(len(segments)) if i not in seen]  # closed loops
    chains = [chain for chain in chains if chain]
    chains.sort(key=lambda chain: segments.loc[chain, "fid"].min())
    ordered = segments.iloc[[i for chain in chains for i in chain]].reset_index(drop=True)
    ordered["section"] = [number for number, chain in enumerate(chains, 1) for _ in chain]
    return ordered


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.Series(float("nan"), index=df.index)


def build_table(ordered: pd.DataFrame, snapshot_df: pd.DataFrame, tabpfn_speeds: dict[str, float] | None) -> pd.DataFrame:
    """Section-ordered segments with speed_<source> and time_<source> columns, the common basis and x positions.

    snapshot_df: the RUN's enriched snapshot (estimated speed, Google speed and duration by segment id).
    tabpfn_speeds: TabPFN speeds shown on the maps by segment id, or None.
    """
    run = pd.DataFrame(
        {
            "segment_id": snapshot_df["segment_id"].astype(str).str.strip(),
            "speed_primary": _numeric(snapshot_df, "estimated_speed"),
            "speed_google": _numeric(snapshot_df, "google_speed_kmh"),
            "time_google": _numeric(snapshot_df, "google_duration_seconds"),
        }
    ).drop_duplicates("segment_id")
    table = ordered.merge(run, on="segment_id", how="left")
    if tabpfn_speeds is not None:
        table["speed_tabpfn"] = pd.to_numeric(table["segment_id"].map(tabpfn_speeds), errors="coerce")
    for key in ("primary", "tabpfn"):
        if f"speed_{key}" in table.columns:
            table[f"speed_{key}"] = table[f"speed_{key}"].where(table[f"speed_{key}"] > 0)
            table[f"time_{key}"] = 3.6 * table["length_m"] / table[f"speed_{key}"]

    table.attrs["sources"] = sources = [k for k in SOURCES if f"time_{k}" in table.columns and table[f"time_{k}"].notna().any()]
    table["in_basis"] = table[[f"time_{k}" for k in sources]].notna().all(axis=1) if sources else False

    # x positions in km, with a gap between sections.
    length_km = table["length_m"] / 1000
    gap = 0.02 * length_km.sum()
    x, previous, x0, x1 = 0.0, None, [], []
    for section, length in zip(table["section"], length_km):
        if previous is not None and section != previous:
            x += gap
        x0.append(x)
        x += length
        x1.append(x)
        previous = section
    return table.assign(x0=x0, x1=x1)


def totals(table: pd.DataFrame) -> dict[str, float]:
    """Travel time in seconds per source over the common basis."""
    basis = table[table["in_basis"]]
    return {key: float(basis[f"time_{key}"].sum()) for key in table.attrs["sources"]}


def section_summary(table: pd.DataFrame, only_lines: set[str] | None = None) -> pd.DataFrame:
    """One row per section: name, segments, length (km) and the time in seconds per source over the basis.

    only_lines: name a section after these bus lines only (a segment can be served by many); None: all.
    """
    rows = []
    for section, part in table.groupby("section", sort=True):
        served = {line for lines in part["bus_line_list"] for line in lines}
        lines = ", ".join(sorted(served & only_lines if only_lines and served & only_lines else served, key=lambda s: (len(s), s)))
        name = part["segment_name"].iloc[0] if len(part) == 1 else f"Line {lines} · direction {part['direction'].iloc[0]}"
        row = {"Section": f"{section}. {name}", "Segments": len(part), "Length (km)": part["length_m"].sum() / 1000}
        basis = part[part["in_basis"]]
        for key in table.attrs["sources"]:
            row[key] = basis[f"time_{key}"].sum() if len(basis) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def contributions(table: pd.DataFrame, top: int = TOP_CONTRIBUTORS) -> list[dict]:
    """Which segments make up the travel-time difference of each model, over the common basis.

    Reference: Google when shown, else the Traffic Estimation Model (then only TabPFN is compared).
    Each entry: model, reference, bars (label, delta seconds, segments), total difference, n segments.
    """
    sources = table.attrs["sources"]
    reference = "google" if "google" in sources else "primary"
    basis = table[table["in_basis"]]
    out = []
    for model in ("primary", "tabpfn"):
        if model not in sources or model == reference or len(basis) < 2:
            continue
        delta = (basis[f"time_{model}"] - basis[f"time_{reference}"]).rename("delta")
        ranked = basis.assign(delta=delta).reindex(delta.abs().sort_values(ascending=False).index)
        bars = pd.DataFrame(
            {
                "label": ranked["segment_name"].head(top) + " · " + ranked["segment_id"].head(top),
                "delta": ranked["delta"].head(top),
                "segments": 1,
            }
        ).reset_index(drop=True)
        rest = ranked.iloc[top:]
        if len(rest):
            bars.loc[len(bars)] = [f"All other segments ({len(rest)})", rest["delta"].sum(), len(rest)]
        out.append({"model": model, "reference": reference, "bars": bars.reset_index(drop=True), "total": float(delta.sum()), "n": len(basis)})
    return out
