from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
import pandas as pd
import requests


LIVE_SPEED_URL = "https://api.mobilitytwin.brussels/stib/aggregated-speed"


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
    Standardize column names from the live speed endpoint.

    Expected logical fields:
    - lineId
    - pointId
    - directionId
    - speed_kmh

    Since the endpoint may not provide a timestamp, the request time is used
    as the snapshot time.
    """
    line_col = find_first_existing_column(
        df.columns,
        candidates=["lineId", "line_id", "line", "route_id"],
        field_name="lineId",
    )

    point_col = find_first_existing_column(
        df.columns,
        candidates=["pointId", "point_id", "stopId", "stop_id", "stop", "stop_code"],
        field_name="pointId",
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

    standardized = df.rename(
        columns={
            line_col: "lineId",
            point_col: "pointId",
            direction_col: "directionId",
            speed_col: "speed_kmh",
        }
    ).copy()

    standardized["lineId"] = standardized["lineId"].astype(str).str.strip()
    standardized["pointId"] = standardized["pointId"].astype(str).str.strip()
    standardized["directionId"] = standardized["directionId"].astype(str).str.strip()
    standardized["speed_kmh"] = pd.to_numeric(standardized["speed_kmh"], errors="coerce")

    snapshot_time = pd.Timestamp.now(tz="Europe/Brussels").tz_localize(None)
    standardized["snapshot_time"] = snapshot_time

    standardized = standardized.dropna(
        subset=["lineId", "pointId", "directionId", "speed_kmh"]
    )

    return standardized


def load_segment_mapping_from_gpkg(gpkg_path: str) -> pd.DataFrame:
    """
    Load segment mapping from the GPKG file.

    Required columns:
    - id
    - start_id
    - bus_lines

    Returned columns:
    - segment_id
    - lineId
    - pointId
    """
    gdf = gpd.read_file(gpkg_path)

    required_columns = {"id", "start_id", "bus_lines"}
    missing_columns = required_columns - set(gdf.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in GPKG file: {sorted(missing_columns)}"
        )

    mapping = gdf[["id", "start_id", "bus_lines"]].copy()
    mapping["bus_lines"] = mapping["bus_lines"].astype(str).str.split(",")
    mapping = mapping.explode("bus_lines")

    mapping["bus_lines"] = mapping["bus_lines"].astype(str).str.strip()
    mapping["start_id"] = mapping["start_id"].astype(str).str.strip()

    mapping = mapping.rename(
        columns={
            "id": "segment_id",
            "start_id": "pointId",
            "bus_lines": "lineId",
        }
    )

    return mapping[["segment_id", "lineId", "pointId"]].drop_duplicates()


def aggregate_live_speeds_by_segment(
    point_speeds: pd.DataFrame,
    segment_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map point-level live speeds to segments and compute one average speed per segment.

    All segments are returned.
    Segments with no matching live data keep avg_speed_kmh as NaN.
    """
    merged = pd.merge(
        segment_mapping,
        point_speeds[["lineId", "pointId", "directionId", "speed_kmh", "snapshot_time"]],
        on=["lineId", "pointId"],
        how="left",
    )

    snapshot_time = (
        point_speeds["snapshot_time"].iloc[0] if not point_speeds.empty else pd.NaT
    )

    segment_snapshot = (
        merged.groupby("segment_id", as_index=False)
        .agg(
            avg_speed_kmh=("speed_kmh", "mean"),
            sample_count=("speed_kmh", lambda values: values.notna().sum()),
        )
        .sort_values("segment_id")
        .reset_index(drop=True)
    )

    segment_snapshot["snapshot_time"] = snapshot_time

    return segment_snapshot[["snapshot_time", "segment_id", "avg_speed_kmh", "sample_count"]]


def fetch_live_segment_speeds(
    *,
    token: str,
    gpkg_path: str,
) -> pd.DataFrame:
    """
    Fetch live STIB speeds and aggregate them to segment level.

    Returns
    -------
    pd.DataFrame
        Columns:
        - snapshot_time
        - segment_id
        - avg_speed_kmh
        - sample_count
    """
    payload = auth_get_json(LIVE_SPEED_URL, token=token)
    raw_df = flatten_json_payload(payload)
    point_speeds = standardize_live_speed_columns(raw_df)
    segment_mapping = load_segment_mapping_from_gpkg(gpkg_path)
    return aggregate_live_speeds_by_segment(point_speeds, segment_mapping)


def build_segment_speed_lookup(segment_snapshot: pd.DataFrame) -> dict[str, float]:
    """
    Convert the segment snapshot into a lookup dictionary.

    Returns
    -------
    dict[str, float]
        Mapping from segment_id to avg_speed_kmh.
        Missing speeds remain NaN in the DataFrame, but are excluded here.
    """
    valid_rows = segment_snapshot.dropna(subset=["avg_speed_kmh"]).copy()
    valid_rows["segment_id"] = valid_rows["segment_id"].astype(str)
    return dict(zip(valid_rows["segment_id"], valid_rows["avg_speed_kmh"]))
