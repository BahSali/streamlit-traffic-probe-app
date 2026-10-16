"""Brussels travel time: per road segment, per connected section, and per source.

No Streamlit and no drawing here (travel_time_charts.py draws it). Everything
comes from the current RUN: the speeds of the Traffic Estimation Model, TabPFN
(when shown) and Google, and Google's reported durations. No new requests.

Travel time of a segment:
- Traffic Estimation Model, TabPFN: t = 3.6 * L / v  (seconds; L in metres, v in km/h);
- Google: its reported duration in seconds.
A section's time is the sum of its segments' times. Times are shown as seconds first, then minutes.

A section is either a route (one selected bus line in one direction, ordered by
chaining the segments' node ids; a line can be cut by the map's edge into
disconnected parts) or, for segments picked by name, a connected run of
segments. A total over several sections is a sum, never one trip. Segments
shared by several selected lines are drawn once (under the first route that
uses them) and counted once in the total; each route's own row includes them.

Totals use one common basis: the segments where every shown source has a
value, so the sources are compared on the same segments.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import pandas as pd

from cities.brussels import comparison as cmp
from cities.brussels.map_data import get_selected_mask, load_brussels_map

METRIC_CRS = 31370  # Belgian Lambert 72: lengths in metres
SOURCES = ("google", "primary", "tabpfn")  # display order: Google where STIB is in the other charts
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


def group_segments(segments: pd.DataFrame, lines: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """The scoped segments in strip order, and the sections they belong to.

    lines: the selected bus lines. Each (line, direction) that has segments is a route, taken in line then
    direction order. A segment shared by several routes is placed once, under the first one. Segments not on a
    selected line (picked by name) form connected sections of their own.

    Returns (segments with section, part, route; routes: [{name, ids, parts}] with all of a route's ids,
    shared ones included).
    """
    routes, placed, groups = [], set(), []
    for line in lines:
        on_line = segments[segments["bus_line_list"].apply(lambda served: line in served)]
        for direction, sub in on_line.groupby("direction"):
            ordered = order_sections(sub)
            name = f"Line {line} · direction {direction}"
            routes.append({"name": name, "ids": ordered["segment_id"].tolist(), "parts": int(ordered["section"].max())})
            fresh = ordered[~ordered["segment_id"].isin(placed)]
            placed.update(fresh["segment_id"])
            groups.append(fresh.assign(part=fresh["section"], section=len(routes), route=name))
    rest = segments[~segments["segment_id"].isin(placed)]
    if len(rest):
        for _, piece in order_sections(rest).groupby("section"):
            name = piece["segment_name"].iloc[0] if len(piece) == 1 else f"{piece['segment_name'].iloc[0]} … ({len(piece)} segments)"
            routes.append({"name": name, "ids": piece["segment_id"].tolist(), "parts": 1})
            groups.append(piece.assign(part=1, section=len(routes), route=name))
    return pd.concat(groups, ignore_index=True), routes


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.Series(float("nan"), index=df.index)


def build_table(
    ordered: pd.DataFrame, routes: list[dict], snapshot_df: pd.DataFrame, tabpfn_speeds: dict[str, float] | None
) -> pd.DataFrame:
    """Segments in strip order (from group_segments, with its routes) with speed_<source> and time_<source> columns, the common basis and x positions.

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
    table["bus_lines"] = table["bus_line_list"].map(lambda lines: ", ".join(sorted(lines, key=lambda line: (len(line), line))))
    if tabpfn_speeds is not None:
        table["speed_tabpfn"] = pd.to_numeric(table["segment_id"].map(tabpfn_speeds), errors="coerce")
    for key in ("primary", "tabpfn"):
        if f"speed_{key}" in table.columns:
            table[f"speed_{key}"] = table[f"speed_{key}"].where(table[f"speed_{key}"] > 0)
            table[f"time_{key}"] = 3.6 * table["length_m"] / table[f"speed_{key}"]

    table.attrs["sources"] = sources = [k for k in SOURCES if f"time_{k}" in table.columns and table[f"time_{k}"].notna().any()]
    table["in_basis"] = table[[f"time_{k}" for k in sources]].notna().all(axis=1) if sources else False

    # x positions in km, with a gap between sections and between disconnected parts of a route.
    length_km = table["length_m"] / 1000
    gap = 0.02 * length_km.sum()
    x, previous, x0, x1 = 0.0, None, [], []
    for section, part, length in zip(table["section"], table["part"], length_km):
        if previous is not None and (section, part) != previous:
            x += gap
        x0.append(x)
        x += length
        x1.append(x)
        previous = (section, part)
    table = table.assign(x0=x0, x1=x1)
    table.attrs["sources"], table.attrs["routes"] = sources, routes
    return table


def totals(table: pd.DataFrame) -> dict[str, float]:
    """Travel time in seconds per source over the common basis."""
    basis = table[table["in_basis"]]
    return {key: float(basis[f"time_{key}"].sum()) for key in table.attrs["sources"]}


def shared_segments(table: pd.DataFrame) -> int:
    """How many segment places the routes have beyond the unique segments (shared stretches)."""
    return sum(len(route["ids"]) for route in table.attrs["routes"]) - len(table)


def route_summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per section: name, segments, length (km) and the time in seconds per source over the basis.

    A route's row includes its shared segments, so rows can overlap; the unique total is totals().
    """
    by_id = table.set_index("segment_id")
    rows = []
    for number, route in enumerate(table.attrs["routes"], 1):
        part = by_id.loc[route["ids"]]
        suffix = f" · {route['parts']} disconnected parts" if route["parts"] > 1 else ""
        row = {"Section": f"{number}. {route['name']}{suffix}", "Segments": len(part), "Length (km)": part["length_m"].sum() / 1000}
        basis = part[part["in_basis"]]
        for key in table.attrs["sources"]:
            row[key] = basis[f"time_{key}"].sum() if len(basis) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def format_time(seconds: float) -> str:
    """Seconds first, minutes second: `125 s (2.08 min)`."""
    return "no value" if pd.isna(seconds) else f"{seconds:,.0f} s ({seconds / 60:.2f} min)"


def export_table(table: pd.DataFrame) -> pd.DataFrame:
    """The segment-level data behind the strips, for the CSV download (available sources only)."""
    out = pd.DataFrame(
        {
            "segment_id": table["segment_id"],
            "street_name": table["segment_name"],
            "section": table["section"],
            "route": table["route"],
            "bus_lines": table["bus_lines"],
            "length_m": table["length_m"].round(1),
        }
    )
    for key in table.attrs["sources"]:
        name = cmp.source_label(key).lower().replace(" ", "_")
        out[f"{name}_speed_kmh"] = table[f"speed_{key}"].round(2)
        out[f"{name}_time_s"] = table[f"time_{key}"].round(1)
    out["used_in_totals"] = table["in_basis"]
    return out
