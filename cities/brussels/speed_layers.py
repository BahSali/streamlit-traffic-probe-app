"""Brussels speed layers: live STIB bus speeds, model estimates and Google speeds.

build_run_payload() joins them onto the map segments (for the three maps) and
onto the STIB snapshot (for the charts and the CSV download). It reports its
stages through on_stage so the page can show progress. build_idle_payload() is
the uncoloured network shown before the first RUN.

Estimated speeds: the model's values are kept in MODEL_ESTIMATE_COLUMNS; the
public columns (est_speed on the map, estimated_speed in the table) hold the
values chosen by core.estimation.correction.displayed_estimates. Only the
public columns leave this module.

Foundation-model estimates (optional fourth map): only when
core.config.SHOW_FOUNDATION_MODEL_MAP is True, foundation_model_speed (from
foundation_model.py) and its FOUNDATION_MODEL_MAP_PROPERTIES are added to the
map segments. They are not part of the table, charts or CSV download.
"""
from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pandas as pd
import shapely
import streamlit as st

from cities.brussels.map_data import (
    MAP_PATH,
    attach_segment_metadata,
    build_segment_metadata_df,
    load_brussels_map,
)
from cities.brussels.foundation_model import (
    UNAVAILABLE_MESSAGE,
    FoundationUnavailable,
    predict_foundation_model_speeds,
)
from core import config
from core.colors import NO_DATA_COLOR, NO_GOOGLE_DATA_COLOR, speed_color_or
from core.data_sources import (
    get_mobility_twin_token,
    load_completed_stib_snapshot,
    load_live_stib_segment_speed_lookup,
)
from core.estimation.correction import displayed_estimates
from core.timing import timed
from core.google_routes.service import (
    attach_google_results_to_map_gdf,
    attach_google_results_to_snapshot_df,
)

# Feature properties sent to the browser: what synced_maps.html reads, plus
# the segment id and numeric speeds for inspection.
MAP_PROPERTIES = [
    "id",
    "segment_name",
    "bus_lines_display",
    "bus_speed",
    "est_speed",
    "google_speed",
    "bus_speed_str",
    "est_speed_str",
    "google_speed_str",
    "google_duration_str",
    "bus_color",
    "est_color",
    "google_color",
    "bus_highlight_color",
    "est_highlight_color",
    "google_highlight_color",
]
# Added to the feature properties only when the foundation-model map is shown.
FOUNDATION_MODEL_MAP_PROPERTIES = [
    "foundation_model_speed",
    "foundation_model_speed_str",
    "foundation_model_color",
    "foundation_model_highlight_color",
]
COORDINATE_DECIMALS = 6  # ~0.1 m

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

    # Only present when the foundation-model map is shown; same formatting and
    # colour scale as the estimated-speed map.
    if "foundation_model_speed" in result.columns:
        result["foundation_model_speed_str"] = result["foundation_model_speed"].apply(format_speed)
        result["foundation_model_color"] = result["foundation_model_speed"].apply(
            speed_color_or, missing_color=NO_DATA_COLOR
        )
        result["foundation_model_highlight_color"] = result["foundation_model_color"]

    return result


@st.cache_resource(show_spinner=False)
def map_geometry_json(path: str = MAP_PATH) -> tuple[str, ...]:
    """GeoJSON geometry of each map segment, serialized once per process."""
    geometries = load_brussels_map(path).geometry.to_numpy()
    rounded = shapely.transform(geometries, lambda coords: np.round(coords, COORDINATE_DECIMALS))
    return tuple(shapely.to_geojson(rounded))


def _json_value(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_geojson_text(gdf: pd.DataFrame) -> str:
    """FeatureCollection text for synced_maps.html (static geometry + current speeds)."""
    geometry_json = map_geometry_json()
    properties = MAP_PROPERTIES
    if "foundation_model_speed" in gdf.columns:
        properties = MAP_PROPERTIES + FOUNDATION_MODEL_MAP_PROPERTIES
    records = gdf[properties].astype(object).to_numpy().tolist()
    features = [
        '{"type": "Feature", "properties": '
        + json.dumps(dict(zip(properties, map(_json_value, record))))
        + ', "geometry": '
        + geometry_json[map_fid]
        + "}"
        for record, map_fid in zip(records, gdf["map_fid"].astype(int))
    ]
    return '{"type": "FeatureCollection", "features": [' + ", ".join(features) + "]}"


def _finish_payload(gdf: pd.DataFrame, **parts) -> dict:
    with timed("brussels.map_columns"):
        gdf = finalize_map_columns(gdf).drop(columns=list(MODEL_ESTIMATE_COLUMNS.values()), errors="ignore")

    with timed("brussels.geojson_serialize", features=len(gdf)) as log:
        geojson_text = build_geojson_text(gdf)
        log["bytes"] = len(geojson_text)

    minx, miny, maxx, maxy = gdf.total_bounds

    return {
        "geojson_text": geojson_text,
        "center_lat": (miny + maxy) / 2,
        "center_lon": (minx + maxx) / 2,
        "selected_google_count": int(gdf["google_speed"].notna().sum()),
        "segment_count": int(len(gdf)),
        "live_bus_count": int(gdf["bus_speed"].notna().sum()),
        **parts,
    }


@st.cache_data(show_spinner=False)
def build_idle_payload(include_foundation_model: bool = False) -> dict:
    """The uncoloured network shown before the first RUN and after Reset.

    include_foundation_model: also carry the (empty) foundation-model fields,
    for the fourth map.
    """
    gdf = load_brussels_map().copy()
    for column in ["bus_speed", "est_speed", "google_speed", "google_duration_seconds"]:
        gdf[column] = pd.NA
    if include_foundation_model:
        gdf["foundation_model_speed"] = pd.NA
    return _finish_payload(
        gdf,
        diagnostics=empty_live_diagnostics(gdf),
        estimation_diagnostics=empty_estimation_diagnostics(gdf),
        completed_snapshot_df=pd.DataFrame(),
        enriched_snapshot_df=pd.DataFrame(),
        model_estimates_df=pd.DataFrame(),
        foundation_shown=include_foundation_model,
    )


def build_run_payload(
    google_results_df: pd.DataFrame,
    refresh_key: int,
    on_stage: Callable[[str], None] = lambda stage: None,
    include_foundation_model: bool | None = None,
) -> dict:
    """Everything the page shows after a RUN.

    include_foundation_model: compute and show the foundation-model map; None
    uses core.config.SHOW_FOUNDATION_MODEL_MAP.

    Stages reported through on_stage: "bus_data" (MobilityTwin live and
    recent STIB data), "estimating" (model), "updating_maps".
    """
    with timed("brussels.load_map"):
        gdf = load_brussels_map().copy()
    segment_metadata_df = build_segment_metadata_df(gdf)

    estimation_diagnostics = empty_estimation_diagnostics(gdf)
    completed_snapshot_df = pd.DataFrame()
    enriched_snapshot_df = pd.DataFrame()
    prediction_df = None

    on_stage("bus_data")
    with timed("brussels.live_bus_speeds"):
        gdf, diagnostics = attach_live_stib_bus_speeds(gdf)

    token = get_mobility_twin_token()
    if token:
        try:
            with timed("brussels.completed_snapshot"):
                completed_snapshot_df = get_completed_snapshot_for_ui(refresh_key)
            completed_snapshot_df = attach_segment_metadata(
                completed_snapshot_df,
                segment_metadata_df,
                source_id_col="segment_id",
            )

            on_stage("estimating")
            # Imported here so opening the page does not load PyTorch.
            with timed("brussels.import_model"):
                from cities.brussels.model import attach_prediction_df_to_gdf, build_estimation_artifacts

            with timed("brussels.model_estimation"):
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

    if include_foundation_model is None:
        include_foundation_model = config.SHOW_FOUNDATION_MODEL_MAP
    foundation_warning = None
    if include_foundation_model:
        with timed("brussels.foundation_model"):
            # A TabPFN failure leaves the foundation map empty; nothing else is affected.
            try:
                gdf["foundation_model_speed"] = predict_foundation_model_speeds(gdf, completed_snapshot_df, token)
            except Exception as exc:
                gdf["foundation_model_speed"] = np.nan
                reason = str(exc) if isinstance(exc, FoundationUnavailable) else type(exc).__name__
                foundation_warning = f"{UNAVAILABLE_MESSAGE} ({reason})"

    on_stage("updating_maps")
    model_estimates_df = enriched_snapshot_df.reindex(
        columns=["segment_id", MODEL_ESTIMATE_COLUMNS["table"]]
    )
    return _finish_payload(
        gdf,
        diagnostics=diagnostics,
        estimation_diagnostics=estimation_diagnostics,
        completed_snapshot_df=completed_snapshot_df,
        # Public table: estimated_speed holds the displayed values.
        enriched_snapshot_df=enriched_snapshot_df.drop(
            columns=list(MODEL_ESTIMATE_COLUMNS.values()), errors="ignore"
        ),
        # Model estimates before any demo correction (not displayed).
        model_estimates_df=model_estimates_df,
        foundation_warning=foundation_warning,
        foundation_shown=include_foundation_model,
    )
