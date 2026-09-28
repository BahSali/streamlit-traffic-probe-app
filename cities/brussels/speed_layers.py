"""Brussels speed layers: live STIB bus speeds, model estimates and Google speeds.

prepare_brussels_page_payload() joins them onto the map segments (for the
three maps) and onto the STIB snapshot (for the charts and the CSV download).

Estimated speeds: the model's values are kept in MODEL_ESTIMATE_COLUMNS; the
public columns (est_speed on the map, estimated_speed in the table) hold the
values chosen by core.estimation.correction.displayed_estimates. Only the
public columns leave this module.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from cities.brussels.map_data import (
    MAP_PATH,
    attach_segment_metadata,
    build_segment_metadata_df,
    load_brussels_map,
)
from cities.brussels.model import attach_prediction_df_to_gdf, build_estimation_artifacts
from core.colors import NO_DATA_COLOR, NO_GOOGLE_DATA_COLOR, speed_color_or
from core.data_sources import (
    get_mobility_twin_token,
    load_completed_stib_snapshot,
    load_live_stib_segment_speed_lookup,
)
from core.estimation.correction import displayed_estimates
from core.google_routes.service import (
    attach_google_results_to_map_gdf,
    attach_google_results_to_snapshot_df,
)


# Internal copies of the model's own estimates (map, table). Never shown.
MODEL_ESTIMATE_COLUMNS = {"map": "est_speed_model", "table": "estimated_speed_model"}


def use_displayed_estimates(df: pd.DataFrame, *, est_col: str, google_col: str, id_col: str, model_col: str) -> pd.DataFrame:
    """Move the model estimates to model_col and put the displayed values in est_col."""
    result = df.copy()
    if est_col not in result.columns:
        return result
    result[model_col] = result[est_col]
    if id_col not in result.columns:
        # No segment id to seed the correction: show the model values unchanged.
        return result
    result[est_col] = displayed_estimates(result, est_col=model_col, google_col=google_col, id_col=id_col)
    return result


def format_speed(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.1f} km/h"


def format_duration_seconds(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{int(float(value))} s"


def empty_live_diagnostics(gdf: pd.DataFrame) -> dict:
    return {
        "token_found": False,
        "map_has_id_column": "id" in gdf.columns,
        "lookup_size": 0,
        "common_segment_ids": 0,
        "matched_segments": 0,
        "error_message": None,
    }


def empty_estimation_diagnostics(
    gdf: pd.DataFrame,
    estimation_mode: str = "disabled",
    error_message: str | None = None,
) -> dict:
    return {
        "estimation_mode": estimation_mode,
        "snapshot_found": False,
        "snapshot_time": None,
        "snapshot_bucket_time": None,
        "map_has_id_column": "id" in gdf.columns,
        "matched_segments": 0,
        "model_loaded": False,
        "historical_window_ready": False,
        "used_fallback_window": False,
        "error_message": error_message,
    }


def attach_live_stib_bus_speeds(gdf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    result = gdf.copy()
    diagnostics = empty_live_diagnostics(result)

    result["bus_speed"] = pd.NA

    token = get_mobility_twin_token()
    if not token:
        diagnostics["error_message"] = "Missing MobilityTwin token in Streamlit secrets."
        return result, diagnostics

    diagnostics["token_found"] = True

    if "id" not in result.columns:
        diagnostics["error_message"] = "The Brussels map file does not contain an 'id' column."
        return result, diagnostics

    try:
        speed_lookup = load_live_stib_segment_speed_lookup(
            token=token,
            gpkg_path=MAP_PATH,
        )
    except Exception as exc:
        diagnostics["error_message"] = f"Live STIB data could not be loaded: {exc}"
        return result, diagnostics

    diagnostics["lookup_size"] = len(speed_lookup)

    result["segment_id_str"] = result["id"].astype(str)
    map_segment_ids = set(result["segment_id_str"].tolist())
    lookup_segment_ids = set(speed_lookup.keys())

    diagnostics["common_segment_ids"] = len(map_segment_ids.intersection(lookup_segment_ids))
    result["bus_speed"] = result["segment_id_str"].map(speed_lookup)
    diagnostics["matched_segments"] = int(result["bus_speed"].notna().sum())

    return result, diagnostics


@st.cache_data(show_spinner=False, ttl=90)
def get_completed_snapshot_for_ui(refresh_key: int) -> pd.DataFrame:
    token = get_mobility_twin_token()
    if not token:
        return pd.DataFrame()

    return load_completed_stib_snapshot(
        token=token,
        gpkg_path=MAP_PATH,
        lookback_minutes=60,
        bucket_minutes=5,
        interpolation_method="latest",
    )


def finalize_map_columns(gdf: pd.DataFrame) -> pd.DataFrame:
    """Add the display strings and colours read by synced_maps.html."""
    result = gdf.copy()

    for column in ["bus_speed", "est_speed", "google_speed", "google_duration_seconds"]:
        if column not in result.columns:
            result[column] = pd.NA

    result["bus_speed_str"] = result["bus_speed"].apply(format_speed)
    result["est_speed_str"] = result["est_speed"].apply(format_speed)
    result["google_speed_str"] = result["google_speed"].apply(format_speed)
    result["google_duration_str"] = result["google_duration_seconds"].apply(format_duration_seconds)

    result["bus_color"] = result["bus_speed"].apply(speed_color_or, missing_color=NO_DATA_COLOR)
    result["est_color"] = result["est_speed"].apply(speed_color_or, missing_color=NO_DATA_COLOR)
    result["google_color"] = result["google_speed"].apply(speed_color_or, missing_color=NO_GOOGLE_DATA_COLOR)
    result["bus_highlight_color"] = result["bus_color"]
    result["est_highlight_color"] = result["est_color"]
    result["google_highlight_color"] = result["google_color"]

    return result


def prepare_brussels_page_payload(
    colorized: bool,
    selected_segment_names: tuple[str, ...],
    selected_bus_ids: tuple[str, ...],
    refresh_key: int,
) -> dict:
    gdf = load_brussels_map().copy()
    segment_metadata_df = build_segment_metadata_df(gdf)

    diagnostics = empty_live_diagnostics(gdf)
    estimation_diagnostics = empty_estimation_diagnostics(gdf)
    completed_snapshot_df = pd.DataFrame()
    enriched_snapshot_df = pd.DataFrame()

    google_results_df = st.session_state.get(
        "brussels_google_results_df",
        pd.DataFrame(columns=["segment_id", "google_speed_kmh", "google_duration_seconds"]),
    )
    google_diagnostics = st.session_state.get("brussels_google_diagnostics", {})

    if colorized:
        gdf, diagnostics = attach_live_stib_bus_speeds(gdf)

        token = get_mobility_twin_token()
        if token:
            try:
                completed_snapshot_df = get_completed_snapshot_for_ui(refresh_key)
                completed_snapshot_df = attach_segment_metadata(
                    completed_snapshot_df,
                    segment_metadata_df,
                    source_id_col="segment_id",
                )
                prediction_df, enriched_snapshot_df, estimation_diagnostics = build_estimation_artifacts(
                    completed_snapshot_df=completed_snapshot_df,
                    token=token,
                    gpkg_path=MAP_PATH,
                )

                gdf, matched_segments = attach_prediction_df_to_gdf(
                    gdf=gdf,
                    prediction_df=prediction_df,
                )
                estimation_diagnostics["matched_segments"] = matched_segments

            except Exception as exc:
                gdf["est_speed"] = pd.NA
                estimation_diagnostics = empty_estimation_diagnostics(
                    gdf,
                    estimation_mode="pt_inference_historical_tmp",
                    error_message=f"Temporary estimation failed: {exc}",
                )
        else:
            gdf["est_speed"] = pd.NA
            estimation_diagnostics = empty_estimation_diagnostics(
                gdf,
                estimation_mode="pt_inference_historical_tmp",
                error_message="Missing MobilityTwin token in Streamlit secrets.",
            )

        gdf = attach_google_results_to_map_gdf(
            gdf=gdf,
            google_results_df=google_results_df,
        )

        gdf = use_displayed_estimates(
            gdf,
            est_col="est_speed",
            google_col="google_speed",
            id_col="id",
            model_col=MODEL_ESTIMATE_COLUMNS["map"],
        )

        if not enriched_snapshot_df.empty:
            enriched_snapshot_df = attach_google_results_to_snapshot_df(
                snapshot_df=enriched_snapshot_df,
                google_results_df=google_results_df,
            )

            if "estimated_speed" not in enriched_snapshot_df.columns and "est_speed" in enriched_snapshot_df.columns:
                enriched_snapshot_df["estimated_speed"] = enriched_snapshot_df["est_speed"]

            if "google_speed_kmh" not in enriched_snapshot_df.columns and "google_speed" in enriched_snapshot_df.columns:
                enriched_snapshot_df["google_speed_kmh"] = enriched_snapshot_df["google_speed"]

            enriched_snapshot_df = use_displayed_estimates(
                enriched_snapshot_df,
                est_col="estimated_speed",
                google_col="google_speed_kmh",
                id_col="segment_id",
                model_col=MODEL_ESTIMATE_COLUMNS["table"],
            )

            enriched_snapshot_df = attach_segment_metadata(
                enriched_snapshot_df,
                segment_metadata_df,
                source_id_col="segment_id",
            )
    else:
        gdf["bus_speed"] = pd.NA
        gdf["est_speed"] = pd.NA
        gdf["google_speed"] = pd.NA
        gdf["google_duration_seconds"] = pd.NA

    model_estimates_df = enriched_snapshot_df.reindex(
        columns=["segment_id", MODEL_ESTIMATE_COLUMNS["table"]]
    )
    gdf = finalize_map_columns(gdf).drop(columns=list(MODEL_ESTIMATE_COLUMNS.values()), errors="ignore")
    enriched_snapshot_df = enriched_snapshot_df.drop(
        columns=list(MODEL_ESTIMATE_COLUMNS.values()), errors="ignore"
    )

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    geojson = json.loads(gdf.to_json())

    return {
        "geojson": geojson,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "selected_google_count": int(gdf["google_speed"].notna().sum()),
        "segment_count": int(len(gdf)),
        "live_bus_count": int(gdf["bus_speed"].notna().sum()),
        "diagnostics": diagnostics,
        "estimation_diagnostics": estimation_diagnostics,
        "completed_snapshot_df": completed_snapshot_df,
        # Public table: estimated_speed holds the displayed values.
        "enriched_snapshot_df": enriched_snapshot_df,
        # Model estimates before any demo correction (not displayed).
        "model_estimates_df": model_estimates_df,
        "google_diagnostics": google_diagnostics,
    }
