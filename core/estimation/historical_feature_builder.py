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
    start_dt: datetime,
    end_dt: datetime,
    bucket_minutes: int = 5,
) -> pd.DataFrame:
    """
    Fetch historical segment-level speeds for an arbitrary time range.
    """
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


def build_segment_time_matrix(
    segment_speed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a time x segment matrix from historical segment-level speeds.

    Output shape:
    - rows: bucket_time
    - columns: segment_id
    - values: avg_speed_kmh
    """
    if segment_speed_df.empty:
        return pd.DataFrame()

    working_df = segment_speed_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str).str.strip()
    working_df["bucket_time"] = pd.to_datetime(working_df["bucket_time"], errors="coerce")

    matrix_df = (
        working_df.pivot_table(
            index="bucket_time",
            columns="segment_id",
            values="avg_speed_kmh",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    matrix_df.columns.name = None
    return matrix_df


def floor_to_bucket(timestamp: pd.Timestamp, bucket_minutes: int) -> pd.Timestamp:
    """
    Floor a timestamp to the nearest lower bucket boundary.
    """
    timestamp = pd.Timestamp(timestamp)
    return timestamp.floor(f"{bucket_minutes}min")


def build_required_feature_timestamps(
    snapshot_time: pd.Timestamp,
    bucket_minutes: int = 15,
) -> dict[str, pd.Timestamp]:
    """
    Build the exact timestamps required by the estimator feature set.

    Features:
    - recent_1 .. recent_4
    - daily_1
    - weekly_1 .. weekly_3

    Notes
    -----
    All timestamps are aligned to the requested bucket size.
    """
    aligned_time = floor_to_bucket(snapshot_time, bucket_minutes)

    return {
        "recent_1": aligned_time - timedelta(minutes=15),
        "recent_2": aligned_time - timedelta(minutes=30),
        "recent_3": aligned_time - timedelta(minutes=45),
        "recent_4": aligned_time - timedelta(minutes=60),
        "daily_1": aligned_time - timedelta(days=1),
        "weekly_1": aligned_time - timedelta(weeks=1),
        "weekly_2": aligned_time - timedelta(weeks=2),
        "weekly_3": aligned_time - timedelta(weeks=3),
    }


def build_required_historical_window(
    snapshot_time: pd.Timestamp,
    bucket_minutes: int = 15,
) -> tuple[datetime, datetime]:
    """
    Build the minimum historical UTC window needed to cover all estimator lags.

    The earliest required timestamp is 3 weeks before the aligned snapshot time.
    """
    aligned_time = floor_to_bucket(snapshot_time, bucket_minutes)
    earliest_needed = aligned_time - timedelta(weeks=3)
    latest_needed = aligned_time

    start_dt = earliest_needed.tz_localize(BRUSSELS_TIMEZONE).tz_convert("UTC").to_pydatetime()
    end_dt = latest_needed.tz_localize(BRUSSELS_TIMEZONE).tz_convert("UTC").to_pydatetime()

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    return start_dt, end_dt


def fetch_estimation_historical_matrix(
    token: str,
    gpkg_path: str,
    snapshot_time: pd.Timestamp,
    bucket_minutes: int = 15,
) -> pd.DataFrame:
    """
    Fetch the historical matrix needed by the estimator.

    Output shape:
    - rows: timestamps
    - columns: segment_id
    - values: avg_speed_kmh
    """
    start_dt, end_dt = build_required_historical_window(
        snapshot_time=snapshot_time,
        bucket_minutes=bucket_minutes,
    )

    segment_df = fetch_historical_segment_speeds(
        token=token,
        gpkg_path=gpkg_path,
        start_dt=start_dt,
        end_dt=end_dt,
        bucket_minutes=bucket_minutes,
    )

    return build_segment_time_matrix(segment_df)


def build_estimation_feature_frame(
    historical_matrix_df: pd.DataFrame,
    completed_snapshot_df: pd.DataFrame,
    snapshot_time: pd.Timestamp,
    bucket_minutes: int = 15,
) -> pd.DataFrame:
    """
    Build the estimator feature frame for the current snapshot.

    Output shape:
    - rows: segment_id
    - columns:
        segment_id
        current_speed_kmh
        recent_1
        recent_2
        recent_3
        recent_4
        daily_1
        weekly_1
        weekly_2
        weekly_3

    Notes
    -----
    The current value comes from completed_snapshot_df.
    Historical lag values come from the historical matrix.
    """
    required_times = build_required_feature_timestamps(
        snapshot_time=snapshot_time,
        bucket_minutes=bucket_minutes,
    )

    completed_df = completed_snapshot_df.copy()
    completed_df["segment_id"] = completed_df["segment_id"].astype(str).str.strip()

    if "final_speed_kmh" not in completed_df.columns:
        raise ValueError(
            "completed_snapshot_df must contain the 'final_speed_kmh' column."
        )

    result = completed_df[["segment_id", "final_speed_kmh"]].rename(
        columns={"final_speed_kmh": "current_speed_kmh"}
    )

    historical_matrix = historical_matrix_df.copy()
    if not historical_matrix.empty:
        historical_matrix.index = pd.to_datetime(historical_matrix.index, errors="coerce")
        historical_matrix.columns = historical_matrix.columns.astype(str)

    for feature_name, feature_time in required_times.items():
        if historical_matrix.empty or feature_time not in historical_matrix.index:
            result[feature_name] = pd.NA
            continue

        feature_series = historical_matrix.loc[feature_time]
        feature_df = feature_series.rename(feature_name).reset_index()
        feature_df.columns = ["segment_id", feature_name]
        feature_df["segment_id"] = feature_df["segment_id"].astype(str).str.strip()

        result = result.merge(feature_df, on="segment_id", how="left")

    ordered_columns = [
        "segment_id",
        "current_speed_kmh",
        "recent_1",
        "recent_2",
        "recent_3",
        "recent_4",
        "daily_1",
        "weekly_1",
        "weekly_2",
        "weekly_3",
    ]

    for column in ordered_columns[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result[ordered_columns].copy()


def prepare_estimation_features(
    token: str,
    gpkg_path: str,
    completed_snapshot_df: pd.DataFrame,
    snapshot_time: pd.Timestamp | None = None,
    bucket_minutes: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare all historical features required by the speed estimator.

    Parameters
    ----------
    token : str
        MobilityTwin API token.
    gpkg_path : str
        Path to the segment GPKG file.
    completed_snapshot_df : pd.DataFrame
        Completed current-time STIB snapshot.
    snapshot_time : pd.Timestamp | None
        The reference timestamp for feature extraction.
        If None, it is inferred from completed_snapshot_df["snapshot_time"].
    bucket_minutes : int
        Historical bucket size used for lag extraction.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - feature_frame_df
        - historical_matrix_df
    """
    if completed_snapshot_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if snapshot_time is None:
        if "snapshot_time" not in completed_snapshot_df.columns:
            raise ValueError(
                "snapshot_time was not provided and completed_snapshot_df does not "
                "contain a 'snapshot_time' column."
            )

        snapshot_time = pd.to_datetime(
            completed_snapshot_df["snapshot_time"].iloc[0],
            errors="coerce",
        )

    if pd.isna(snapshot_time):
        raise ValueError("Could not determine a valid snapshot_time.")

    historical_matrix_df = fetch_estimation_historical_matrix(
        token=token,
        gpkg_path=gpkg_path,
        snapshot_time=pd.Timestamp(snapshot_time),
        bucket_minutes=bucket_minutes,
    )

    feature_frame_df = build_estimation_feature_frame(
        historical_matrix_df=historical_matrix_df,
        completed_snapshot_df=completed_snapshot_df,
        snapshot_time=pd.Timestamp(snapshot_time),
        bucket_minutes=bucket_minutes,
    )

    return feature_frame_df, historical_matrix_df
