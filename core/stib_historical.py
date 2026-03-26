from __future__ import annotations

from io import BytesIO
from datetime import datetime, timedelta, timezone

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


API_URL = "https://api.mobilitytwin.brussels/parquetized"
COMPONENT_NAME = "stib_vehicle_distance_parquetize"
BRUSSELS_TIMEZONE = "Europe/Brussels"


def auth_request(url: str, token: str, timeout: int = 60) -> dict:
    """
    Send an authenticated GET request and return the JSON response.
    """
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def download_parquets(url_list: list[str], timeout: int = 120) -> pa.Table:
    """
    Download parquet files and concatenate them into a single Arrow table.
    """
    combined_table: pa.Table | None = None

    for url in url_list:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        table = pq.read_table(BytesIO(response.content))

        if combined_table is None:
            combined_table = table
        else:
            combined_table = pa.concat_tables([combined_table, table])

    if combined_table is None:
        raise RuntimeError("No parquet data could be downloaded.")

    return combined_table


def build_parquetized_url(start_dt: datetime, end_dt: datetime) -> str:
    """
    Build the parquetized endpoint URL for the given UTC datetime range.
    """
    return (
        f"{API_URL}"
        f"?start_timestamp={start_dt.timestamp()}"
        f"&end_timestamp={end_dt.timestamp()}"
        f"&component={COMPONENT_NAME}"
    )


def load_segment_metadata_from_gpkg(gpkg_path: str) -> pd.DataFrame:
    """
    Load segment metadata from the GPKG file.

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


def fetch_historical_point_speeds(
    token: str,
    start_dt: datetime,
    end_dt: datetime,
    bucket_minutes: int = 5,
) -> pd.DataFrame:
    """
    Fetch historical STIB point-level speeds for the requested time range.

    Returns
    -------
    pd.DataFrame
        Columns:
        - bucket_time
        - lineId
        - pointId
        - directionId
        - avg_speed_kmh
    """
    if start_dt.tzinfo is None or end_dt.tzinfo is None:
        raise ValueError("start_dt and end_dt must be timezone-aware UTC datetimes.")

    url = build_parquetized_url(start_dt, end_dt)
    payload = auth_request(url, token)
    parquet_urls = payload.get("results", [])

    if not parquet_urls:
        return pd.DataFrame(
            columns=["bucket_time", "lineId", "pointId", "directionId", "avg_speed_kmh"]
        )

    arrow_table = download_parquets(parquet_urls)

    con = duckdb.connect()
    con.register("combined_data", arrow_table)

    query = f"""
        WITH entries AS (
            SELECT
                CAST(lineId AS VARCHAR) AS lineId,
                CAST(pointId AS VARCHAR) AS pointId,
                CAST(directionId AS VARCHAR) AS directionId,
                CAST(distanceFromPoint AS DOUBLE) AS distanceFromPoint,
                (date AT TIME ZONE 'UTC' AT TIME ZONE '{BRUSSELS_TIMEZONE}')::timestamp AS local_date
            FROM combined_data
            WHERE lineId IS NOT NULL
              AND pointId IS NOT NULL
              AND directionId IS NOT NULL
              AND distanceFromPoint IS NOT NULL
              AND date IS NOT NULL
        ),
        filtered AS (
            SELECT
                *,
                COUNT(*) OVER (
                    PARTITION BY lineId, directionId, pointId, local_date
                ) AS row_count
            FROM entries
        ),
        deltas AS (
            SELECT
                lineId,
                pointId,
                directionId,
                local_date,
                distanceFromPoint,
                distanceFromPoint - LAG(distanceFromPoint) OVER (
                    PARTITION BY lineId, directionId, pointId
                    ORDER BY local_date
                ) AS distance_delta,
                EXTRACT(
                    EPOCH FROM (
                        local_date - LAG(local_date) OVER (
                            PARTITION BY lineId, directionId, pointId
                            ORDER BY local_date
                        )
                    )
                ) AS time_delta_seconds
            FROM filtered
            WHERE row_count = 1
        ),
        valid_speeds AS (
            SELECT
                lineId,
                pointId,
                directionId,
                local_date,
                (distance_delta / time_delta_seconds) * 3.6 AS speed_kmh
            FROM deltas
            WHERE time_delta_seconds > 0
              AND time_delta_seconds <= 120
              AND distance_delta >= 0
              AND distance_delta < 1000
        ),
        bucketed AS (
            SELECT
                lineId,
                pointId,
                directionId,
                time_bucket(INTERVAL '{bucket_minutes} minutes', local_date) AS bucket_time,
                AVG(speed_kmh) AS avg_speed_kmh
            FROM valid_speeds
            GROUP BY 1, 2, 3, 4
        )
        SELECT
            bucket_time,
            lineId,
            pointId,
            directionId,
            ROUND(avg_speed_kmh, 2) AS avg_speed_kmh
        FROM bucketed
        ORDER BY bucket_time, lineId, directionId, pointId
    """

    return con.execute(query).df()


def map_historical_point_speeds_to_segments(
    point_speed_df: pd.DataFrame,
    gpkg_path: str,
) -> pd.DataFrame:
    """
    Map historical point-level speeds to segment-level speeds.

    Returns
    -------
    pd.DataFrame
        Columns:
        - bucket_time
        - segment_id
        - avg_speed_kmh
        - sample_count
    """
    if point_speed_df.empty:
        return pd.DataFrame(
            columns=["bucket_time", "segment_id", "avg_speed_kmh", "sample_count"]
        )

    segment_metadata = load_segment_metadata_from_gpkg(gpkg_path)

    working_df = point_speed_df.copy()
    working_df["lineId"] = working_df["lineId"].astype(str).str.strip()
    working_df["pointId"] = working_df["pointId"].astype(str).str.strip()

    merged = pd.merge(
        working_df,
        segment_metadata,
        on=["lineId", "pointId"],
        how="inner",
    )

    segment_df = (
        merged.groupby(["bucket_time", "segment_id"], as_index=False)
        .agg(
            avg_speed_kmh=("avg_speed_kmh", "mean"),
            sample_count=("avg_speed_kmh", lambda values: values.notna().sum()),
        )
        .sort_values(["bucket_time", "segment_id"])
        .reset_index(drop=True)
    )

    return segment_df


def fetch_historical_segment_speeds(
    token: str,
    gpkg_path: str,
    lookback_minutes: int = 60,
    bucket_minutes: int = 5,
) -> pd.DataFrame:
    """
    Fetch historical STIB segment speeds for the recent lookback window.

    The default configuration returns the last 60 minutes with 5-minute buckets.
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(minutes=lookback_minutes)

    point_df = fetch_historical_point_speeds(
        token=token,
        start_dt=start_dt,
        end_dt=end_dt,
        bucket_minutes=bucket_minutes,
    )

    return map_historical_point_speeds_to_segments(
        point_speed_df=point_df,
        gpkg_path=gpkg_path,
    )
