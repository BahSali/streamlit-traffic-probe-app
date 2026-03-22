from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
import pandas as pd
import requests


LIVE_SPEED_URL = "https://api.mobilitytwin.brussels/stib/aggregated-speed"
BRUSSELS_TIMEZONE = "Europe/Brussels"


def auth_get_json(url: str, token: str, timeout: int = 60) -> dict | list:
    """
    Send an authenticated GET request and return the parsed JSON payload.
    """
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def flatten_json_payload(payload: dict | list) -> pd.DataFrame:
    """
    Flatten the live endpoint JSON payload into a DataFrame.
    """
    if isinstance(payload, list):
        return pd.json_normalize(payload)

    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            return pd.json_normalize(payload["results"])

        if "data" in payload and isinstance(payload["data"], list):
            return pd.json_normalize(payload["data"])

        if "features" in payload and isinstance(payload["features"], list):
            rows: list[dict] = []

            for feature in payload["features"]:
                row: dict = {}
                if isinstance(feature, dict):
                    row.update(feature.get("properties", {}))
                    if "geometry" in feature:
                        row["geometry"] = feature["geometry"]
                    if "id" in feature:
                        row["feature_id"] = feature["id"]
                rows.append(row)

            return pd.json_normalize(rows)

    raise ValueError("Unsupported JSON payload shape from live speed endpoint.")


def find_first_existing_column(
    columns: Iterable[str],
    candidates: list[str],
    field_name: str,
) -> str:
    """
    Return the first matching column name from a candidate list.
    """
    column_set = set(columns)

    for candidate in candidates:
        if candidate in column_set:
            return candidate

    raise ValueError(
        f"Could not detect the '{field_name}' column. "
        f"Available columns: {sorted(columns)}"
    )


def standardize_live_speed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the live speed endpoint column names.

    Expected logical fields:
    - lineId
    - pointId
    - directionId
    - speed

    Since no timestamp is returned, the request time is used as snapshot time.
    """
    line_col = find_first_existing_column(
        df.columns,
        candidates=["lineId", "line_id", "line", "route_id"],
        field_name="lineId",
    )

    point_col = find_first_existing_column(
        df.columns,
        candidates=[
            "pointId",
            "point_id",
            "stopId",
            "stop_id",
            "stop",
            "stop_code",
        ],
        field_name="pointId/stopId",
    )

    direction_col = find_first_existing_column(
        df.columns,
        candidates=["directionId", "direction_id", "direction"],
        field_name="directionId",
    )

    speed_col = find_first_existing_column(
        df.columns,
        candidates=["speed", "speed_kmh", "avg_speed", "average_speed"],
        field_name="speed",
    )

    standardized = df.copy()
    standardized = standardized.rename(
        columns={
            line_col: "lineId",
            point_col: "pointId",
            direction_col: "directionId",
            speed_col: "speed_kmh",
        }
    )

    standardized["lineId"] = standardized["lineId"].astype(str).str.strip()
    standardized["pointId"] = standardized["pointId"].astype(str).str.strip()
    standardized["directionId"] = standardized["directionId"].astype(str).str.strip()
    standardized["speed_kmh"] = pd.to_numeric(standardized["speed_kmh"], errors="coerce")

    snapshot_time = pd.Timestamp.now(tz=BRUSSELS_TIMEZONE).tz_localize(None)
    standardized["local_date"] = snapshot_time

    standardized = standardized.dropna(
        subset=["lineId", "pointId", "directionId", "speed_kmh"]
    )

    return standardized


def load_segment_metadata_from_gpkg(gpkg_path: str) -> pd.DataFrame:
    """
    Load segment metadata directly from the GPKG file.

    Expected columns:
    - id
    - start_id
    - bus_lines
    """
    gdf = gpd.read_file(gpkg_path)

    required_columns = {"id", "start_id", "bus_lines"}
    missing_columns = required_columns - set(gdf.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in GPKG file: {sorted(missing_columns)}"
        )

    segments = gdf[["id", "start_id", "bus_lines"]].copy()

    segments["bus_lines"] = segments["bus_lines"].astype(str).str.split(",")
    segments = segments.explode("bus_lines")

    segments["bus_lines"] = segments["bus_lines"].astype(str).str.strip()
    segments["start_id"] = segments["start_id"].astype(str).str.strip()

    segments = segments.rename(
        columns={
            "id": "segment_id",
            "start_id": "pointId",
            "bus_lines": "lineId",
        }
    )

    return segments[["segment_id", "pointId", "lineId"]].drop_duplicates()


def build_segment_speed_snapshot(
    point_speeds: pd.DataFrame,
    segment_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map live point-level speeds to segments and compute average speed per segment.

    All segments are returned.
    Segments without data keep avg_speed_kmh as NaN.
    """
    point_speeds = point_speeds.copy()
    point_speeds["lineId"] = point_speeds["lineId"].astype(str).str.strip()
    point_speeds["pointId"] = point_speeds["pointId"].astype(str).str.strip()

    merged = pd.merge(
        point_speeds,
        segment_metadata,
        on=["lineId", "pointId"],
        how="right",
    )

    snapshot_time = (
        point_speeds["local_date"].iloc[0] if not point_speeds.empty else pd.NaT
    )

    segment_snapshot = (
        merged.groupby("segment_id", as_index=False)
        .agg(
            avg_speed_kmh=("speed_kmh", "mean"),
            sample_count=("speed_kmh", lambda s: s.notna().sum()),
        )
        .sort_values("segment_id")
        .reset_index(drop=True)
    )

    segment_snapshot["snapshot_time"] = snapshot_time

    segment_snapshot = segment_snapshot[
        ["snapshot_time", "segment_id", "avg_speed_kmh", "sample_count"]
    ]

    return segment_snapshot


def fetch_live_segment_speeds(
    token: str,
    gpkg_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch live STIB speed data and aggregate it by segment.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - segment_snapshot
        - live_point_speeds
    """
    payload = auth_get_json(LIVE_SPEED_URL, token)
    raw_df = flatten_json_payload(payload)
    standardized_df = standardize_live_speed_columns(raw_df)

    segment_metadata = load_segment_metadata_from_gpkg(gpkg_path)
    segment_snapshot = build_segment_speed_snapshot(
        standardized_df,
        segment_metadata,
    )

    return segment_snapshot, standardized_df


def build_segment_speed_lookup(segment_snapshot: pd.DataFrame) -> dict[str, float]:
    """
    Convert the segment snapshot into a segment_id -> avg_speed_kmh dictionary.
    """
    valid_rows = segment_snapshot.dropna(subset=["avg_speed_kmh"]).copy()
    valid_rows["segment_id"] = valid_rows["segment_id"].astype(str).str.strip()

    return dict(zip(valid_rows["segment_id"], valid_rows["avg_speed_kmh"]))
