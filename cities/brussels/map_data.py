"""Brussels road-segment map: loading, segment metadata and filter selection."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from core.config import DATA_DIR
from core.data_sources import load_gpkg

MAP_PATH = str(DATA_DIR / "Brussels_map_6km.gpkg")


@st.cache_data(show_spinner=False)
def parse_bus_lines(value) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    tokens = re.findall(r"[A-Za-z0-9]+", text)
    return [token.strip() for token in tokens if token.strip()]


@st.cache_data(show_spinner=False)
def load_brussels_map(path: str = MAP_PATH):
    gdf = load_gpkg(path).copy()

    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf.reset_index(drop=True)
    gdf["map_fid"] = gdf.index.astype(int)

    for column in ["start_name", "end_name", "bus_lines"]:
        if column not in gdf.columns:
            gdf[column] = ""

    gdf["segment_name"] = (
        gdf["start_name"].fillna("").astype(str).str.strip()
        + " - "
        + gdf["end_name"].fillna("").astype(str).str.strip()
    )
    gdf["segment_name"] = gdf["segment_name"].replace(" - ", "").fillna("N/A")

    gdf["bus_lines_display"] = gdf["bus_lines"].fillna("").astype(str)
    gdf["bus_line_list"] = gdf["bus_lines"].apply(parse_bus_lines)

    return gdf


def build_segment_metadata_df(gdf: pd.DataFrame) -> pd.DataFrame:
    metadata_df = gdf.copy()

    if "id" not in metadata_df.columns:
        raise KeyError("Brussels map gdf must contain an 'id' column.")

    metadata_df["segment_id"] = metadata_df["id"].astype(str).str.strip()

    for column in ["segment_name", "bus_lines"]:
        if column not in metadata_df.columns:
            metadata_df[column] = ""

    metadata_df = (
        metadata_df[["segment_id", "segment_name", "bus_lines"]]
        .drop_duplicates(subset=["segment_id"])
        .reset_index(drop=True)
    )

    return metadata_df


def attach_segment_metadata(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    source_id_col: str = "segment_id",
) -> pd.DataFrame:
    if df.empty or source_id_col not in df.columns:
        return df.copy()

    result = df.copy()
    result[source_id_col] = result[source_id_col].astype(str).str.strip()

    metadata_for_merge = metadata_df.rename(columns={"segment_id": source_id_col})

    for col in ["segment_name", "bus_lines"]:
        if col in result.columns:
            result = result.drop(columns=[col])

    result = result.merge(
        metadata_for_merge,
        on=source_id_col,
        how="left",
        validate="many_to_one",
    )

    return result


@st.cache_data(show_spinner=False)
def get_filter_options():
    gdf = load_brussels_map()

    segment_options = sorted(
        [
            value
            for value in gdf["segment_name"].dropna().astype(str).unique().tolist()
            if value.strip()
        ]
    )

    bus_id_options = sorted(
        {
            bus_id
            for bus_list in gdf["bus_line_list"].tolist()
            for bus_id in bus_list
            if str(bus_id).strip()
        },
        key=lambda value: (len(str(value)), str(value)),
    )

    return segment_options, bus_id_options


def get_selected_mask(
    gdf: pd.DataFrame,
    selected_segment_names: list[str],
    selected_bus_ids: list[str],
) -> pd.Series:
    selected_segment_names = {
        str(value).strip()
        for value in (selected_segment_names or [])
        if str(value).strip()
    }
    selected_bus_ids = {
        str(value).strip()
        for value in (selected_bus_ids or [])
        if str(value).strip()
    }

    mask = pd.Series(False, index=gdf.index)

    if selected_segment_names:
        mask = mask | gdf["segment_name"].isin(selected_segment_names)

    if selected_bus_ids:
        mask = mask | gdf["bus_line_list"].apply(
            lambda bus_list: any(bus_id in selected_bus_ids for bus_id in bus_list)
        )

    return mask

