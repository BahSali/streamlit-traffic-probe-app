"""Brussels chart data: the speeds of every source side by side, and the numbers the charts show.

No Streamlit and no drawing here, so the rules for which sources and which
segments are compared live in one place (drawing is charts.py).

Sources: STIB (observed probe speed, not ground truth), the Traffic Estimation Model,
TabPFN (only when its prediction is shown on the maps) and Google. Google is
the reference when there is enough of it (MIN_REFERENCE_ROWS segments).

Samples (each statistic says how many segments it uses):
- the models (Traffic Estimation Model, TabPFN) are compared with Google on the segments where
  all of them and Google have a value, so they are directly comparable;
- STIB is compared with Google on its own segments (a different sample).
Without Google the sources are only compared with each other, pair by pair.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from core import config

# Display order of the sources, and the ones that are models.
SOURCES = ("stib", "primary", "tabpfn", "google")
MODELS = ("primary", "tabpfn")
# Fewer Google segments than this: no error statistics or Google-reference charts.
MIN_REFERENCE_ROWS = 10
TOP_STREETS = 15


def source_label(key: str) -> str:
    return {
        "stib": "STIB",
        "primary": "Traffic Estimation Model",
        "tabpfn": config.FOUNDATION_MODEL_NAME,
        "google": "Google",
    }[key]


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def build_frame(snapshot_df: pd.DataFrame, tabpfn_speeds: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per segment: segment_id, segment_name and a speed column per source.

    snapshot_df: the RUN's enriched STIB snapshot (the table of the download).
    tabpfn_speeds: the TabPFN speeds shown on the maps by segment id, or None
    (map off / no prediction yet), which leaves the column out.
    """
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame()
    ids = snapshot_df["segment_id"].astype(str).str.strip()
    # STIB as the maps show it: the live speed (final_speed_kmh also holds interpolated history).
    stib_column = "live_speed_kmh" if "live_speed_kmh" in snapshot_df.columns else "final_speed_kmh"
    frame = pd.DataFrame(
        {
            "segment_id": ids,
            "segment_name": snapshot_df.get("segment_name", pd.Series("N/A", index=snapshot_df.index)),
            "stib": _numeric(snapshot_df, stib_column),
            "primary": _numeric(snapshot_df, "estimated_speed"),
        }
    )
    if tabpfn_speeds is not None:
        frame["tabpfn"] = pd.to_numeric(ids.map(tabpfn_speeds), errors="coerce")
    frame["google"] = _numeric(snapshot_df, "google_speed_kmh")
    return frame


def available_sources(frame: pd.DataFrame) -> list[str]:
    """Sources with at least one value, in display order (Google included)."""
    return [key for key in SOURCES if key in frame.columns and frame[key].notna().any()]


def reference_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Long table of speed - Google per source: source, segment_id, segment_name, google, speed, error, abs_error.

    The models share one sample (all of them and Google present); STIB has its own.
    """
    available = available_sources(frame)
    with_google = frame[frame["google"].notna()]
    models = [key for key in MODELS if key in available]
    model_sample = with_google.dropna(subset=models) if models else None
    # Built in SOURCES order, so every chart lists the sources the same way.
    samples = {}
    for key in SOURCES:
        if key in models:
            samples[key] = model_sample
        elif key == "stib" and key in available:
            samples[key] = with_google.dropna(subset=["stib"])
    parts = [
        sample[["segment_id", "segment_name", "google"]].assign(source=key, speed=sample[key])
        for key, sample in samples.items()
    ]
    if not parts:
        return pd.DataFrame(columns=["source", "segment_id", "segment_name", "google", "speed", "error", "abs_error"])
    errors = pd.concat(parts, ignore_index=True)
    errors["error"] = errors["speed"] - errors["google"]
    errors["abs_error"] = errors["error"].abs()
    return errors


def has_reference(frame: pd.DataFrame) -> bool:
    return bool(frame["google"].notna().sum() >= MIN_REFERENCE_ROWS and not reference_errors(frame).empty)


def error_summary(errors: pd.DataFrame) -> pd.DataFrame:
    """Per source (display order): n, MAE, Median AE, RMSE, Bias (mean of speed - Google)."""
    grouped = errors.groupby("source")
    summary = pd.DataFrame(
        {
            "n": grouped.size(),
            "MAE": grouped["abs_error"].mean(),
            "Median AE": grouped["abs_error"].median(),
            "RMSE": grouped["error"].apply(lambda e: float(np.sqrt((e**2).mean()))),
            "Bias": grouped["error"].mean(),
        }
    )
    return summary.reindex([key for key in SOURCES if key in summary.index])


def source_pairs(frame: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Segments where two sources have a value, as x/y tables: STIB, Traffic Estimation Model, TabPFN taken two at a time."""
    keys = [key for key in available_sources(frame) if key != "google"]
    pairs = {}
    for x_key, y_key in combinations(keys, 2):
        both = frame.dropna(subset=[x_key, y_key])
        pairs[(x_key, y_key)] = both[["segment_id", "segment_name"]].assign(x=both[x_key], y=both[y_key])
    return pairs


def difference_summary(pairs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Per pair: n, mean |y - x| and mean (y - x). Agreement between sources, not accuracy."""
    rows = {
        f"{source_label(y_key)} vs {source_label(x_key)}": {
            "n": len(pair),
            "Mean |difference|": (pair["y"] - pair["x"]).abs().mean(),
            "Mean difference": (pair["y"] - pair["x"]).mean(),
        }
        for (x_key, y_key), pair in pairs.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def street_errors(errors: pd.DataFrame, rank_by: str, highest: bool = True) -> pd.DataFrame:
    """MAE vs Google per street (segment_name) and source, for the TOP_STREETS streets with the highest
    (or lowest) MAE of the rank_by source (a model key).

    Columns: street, source, mae, segments (of that source in the street), rank_mae, rank_segments (of rank_by),
    in SOURCES order.
    """
    street = (
        errors.groupby(["segment_name", "source"], sort=False)["abs_error"]
        .agg(mae="mean", segments="size")
        .reset_index()
        .rename(columns={"segment_name": "street"})
    )
    ranked = street[street["source"] == rank_by].set_index("street")
    chosen = ranked["mae"].nlargest(TOP_STREETS) if highest else ranked["mae"].nsmallest(TOP_STREETS)
    street = street[street["street"].isin(chosen.index)]
    return street.assign(
        rank_mae=street["street"].map(ranked["mae"]), rank_segments=street["street"].map(ranked["segments"])
    )


def street_spread(frame: pd.DataFrame) -> pd.DataFrame:
    """For the TOP_STREETS streets where the available sources disagree most: street, spread, segments.

    spread = highest - lowest speed among the available non-Google sources of a
    segment (needs two), averaged over the street.
    """
    keys = [key for key in available_sources(frame) if key != "google"]
    rows = frame[keys].dropna(thresh=2)
    spread = (rows.max(axis=1) - rows.min(axis=1)).rename("spread")
    street = (
        frame.loc[rows.index, ["segment_name"]]
        .join(spread)
        .groupby("segment_name")["spread"]
        .agg(spread="mean", segments="size")
        .reset_index()
        .rename(columns={"segment_name": "street"})
    )
    return street.nlargest(TOP_STREETS, "spread")
