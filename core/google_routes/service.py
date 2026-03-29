from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


GOOGLE_ROUTES_SECRET_KEY = "GOOGLE_ROUTES_API_KEY"
GOOGLE_ROUTES_MONTHLY_LIMIT = 5000
GOOGLE_ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_ROUTES_TIMEOUT_SECONDS = 20

BRUSSELS_TIMEZONE = "Europe/Brussels"

MAX_GAP_METERS = 15.0
MAX_TURN_DEG = 30.0
MAX_GROUP_SIZE = 17

USAGE_DIR = Path("data/google_routes_usage")
USAGE_CSV_PATH = USAGE_DIR / "google_routes_usage.csv"

SESSION = requests.Session()


def get_google_routes_api_key() -> str | None:
    """
    Read the Google Routes API key from Streamlit secrets.
    """
    return st.secrets.get(GOOGLE_ROUTES_SECRET_KEY)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distance in meters between two latitude/longitude points.
    """
    radius_m = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius_m * c


def segment_vector(segment_row: pd.Series) -> tuple[float, float]:
    """
    Build a direction vector from start point to end point in degree space.
    """
    return (
        float(segment_row["end_lon"]) - float(segment_row["start_lon"]),
        float(segment_row["end_lat"]) - float(segment_row["start_lat"]),
    )


def angle_between(
    vector_1: tuple[float, float],
    vector_2: tuple[float, float],
) -> float:
    """
    Angle between two vectors in degrees.
    """
    x1, y1 = vector_1
    x2, y2 = vector_2

    dot_product = x1 * x2 + y1 * y2
    norm_1 = math.hypot(x1, y1)
    norm_2 = math.hypot(x2, y2)

    if norm_1 == 0 or norm_2 == 0:
        return 0.0

    cosine_angle = max(min(dot_product / (norm_1 * norm_2), 1.0), -1.0)
    return math.degrees(math.acos(cosine_angle))


def ensure_google_usage_storage() -> None:
    """
    Create the usage storage directory and CSV file if they do not exist.
    """
    USAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not USAGE_CSV_PATH.exists():
        with open(USAGE_CSV_PATH, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                [
                    "request_timestamp_utc",
                    "month_key",
                    "request_count",
                    "group_count",
                    "selected_segment_count",
                    "success_count",
                    "failure_count",
                ]
            )


def get_month_key_utc(dt: datetime | None = None) -> str:
    """
    Convert a datetime to a UTC month key such as '2026-03'.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m")


def read_google_usage_log() -> pd.DataFrame:
    """
    Read the persistent Google usage log.
    """
    ensure_google_usage_storage()

    try:
        usage_df = pd.read_csv(USAGE_CSV_PATH)
    except Exception:
        usage_df = pd.DataFrame(
            columns=[
                "request_timestamp_utc",
                "month_key",
                "request_count",
                "group_count",
                "selected_segment_count",
                "success_count",
                "failure_count",
            ]
        )

    if usage_df.empty:
        return usage_df

    numeric_columns = [
        "request_count",
        "group_count",
        "selected_segment_count",
        "success_count",
        "failure_count",
    ]
    for column in numeric_columns:
        usage_df[column] = pd.to_numeric(usage_df[column], errors="coerce").fillna(0).astype(int)

    usage_df["month_key"] = usage_df["month_key"].astype(str)

    return usage_df


def get_monthly_google_request_count(month_key: str | None = None) -> int:
    """
    Return the total number of Google Routes requests already used this month.
    """
    if month_key is None:
        month_key = get_month_key_utc()

    usage_df = read_google_usage_log()
    if usage_df.empty:
        return 0

    month_df = usage_df.loc[usage_df["month_key"] == month_key]
    if month_df.empty:
        return 0

    return int(month_df["request_count"].sum())


def can_consume_google_requests(
    planned_request_count: int,
    monthly_limit: int = GOOGLE_ROUTES_MONTHLY_LIMIT,
) -> tuple[bool, int, int]:
    """
    Check whether a planned batch can be executed without crossing the monthly limit.
    """
    month_key = get_month_key_utc()
    used_request_count = get_monthly_google_request_count(month_key=month_key)
    remaining_request_count = monthly_limit - used_request_count

    return planned_request_count <= remaining_request_count, used_request_count, remaining_request_count


def append_google_usage_log_row(
    request_count: int,
    group_count: int,
    selected_segment_count: int,
    success_count: int,
    failure_count: int,
) -> None:
    """
    Append one usage row after a Google batch has actually been sent.
    """
    ensure_google_usage_storage()

    now_utc = datetime.now(timezone.utc)
    month_key = get_month_key_utc(now_utc)

    with open(USAGE_CSV_PATH, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                now_utc.isoformat(),
                month_key,
                int(request_count),
                int(group_count),
                int(selected_segment_count),
                int(success_count),
                int(failure_count),
            ]
        )


def extract_google_ready_segments(selected_gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the columns required for Google Routes requests.

    The selected subset is expected to already represent the filtered segments only.
    """
    required_columns = {
        "id",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "map_fid",
    }
    missing_columns = required_columns - set(selected_gdf.columns)

    if missing_columns:
        raise ValueError(
            f"Selected Brussels segments are missing required columns: {sorted(missing_columns)}"
        )

    result = selected_gdf.copy()

    result["segment_id"] = result["id"].astype(str).str.strip()
    result["start_lat"] = pd.to_numeric(result["start_lat"], errors="coerce")
    result["start_lon"] = pd.to_numeric(result["start_lon"], errors="coerce")
    result["end_lat"] = pd.to_numeric(result["end_lat"], errors="coerce")
    result["end_lon"] = pd.to_numeric(result["end_lon"], errors="coerce")
    result["map_fid"] = pd.to_numeric(result["map_fid"], errors="coerce").astype("Int64")

    result = result.dropna(
        subset=["segment_id", "start_lat", "start_lon", "end_lat", "end_lon", "map_fid"]
    ).copy()

    result = result.sort_values(["map_fid", "segment_id"]).reset_index(drop=True)

    return result[
        [
            "segment_id",
            "map_fid",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
        ]
    ]


def build_groups_for_selected_segments(
    selected_segments_df: pd.DataFrame,
    max_gap_meters: float = MAX_GAP_METERS,
    max_turn_deg: float = MAX_TURN_DEG,
    max_group_size: int = MAX_GROUP_SIZE,
) -> list[pd.DataFrame]:
    """
    Build adjacency groups using only the selected subset.

    Important:
    - No global network-wide grouping is performed.
    - Only the filtered segments are inspected.
    - Segments are grouped only if they are close and directionally compatible.
    """
    if selected_segments_df.empty:
        return []

    groups: list[list[int]] = []
    current_group_indices: list[int] = [0]

    for row_index in range(1, len(selected_segments_df)):
        previous_row = selected_segments_df.iloc[current_group_indices[-1]]
        current_row = selected_segments_df.iloc[row_index]

        gap_meters = haversine_m(
            float(previous_row["end_lat"]),
            float(previous_row["end_lon"]),
            float(current_row["start_lat"]),
            float(current_row["start_lon"]),
        )

        previous_vector = segment_vector(previous_row)
        current_vector = segment_vector(current_row)
        turn_angle_deg = angle_between(previous_vector, current_vector)

        is_adjacent = (
            gap_meters <= max_gap_meters
            and turn_angle_deg <= max_turn_deg
            and len(current_group_indices) < max_group_size
        )

        if is_adjacent:
            current_group_indices.append(row_index)
        else:
            groups.append(current_group_indices)
            current_group_indices = [row_index]

    groups.append(current_group_indices)

    return [selected_segments_df.iloc[group_indices].reset_index(drop=True) for group_indices in groups]


def parse_duration_to_seconds(duration_value: str | None) -> int | None:
    """
    Google duration values usually look like '123s'.
    """
    if not isinstance(duration_value, str):
        return None

    if not duration_value.endswith("s"):
        return None

    try:
        return int(duration_value[:-1])
    except Exception:
        return None


def build_google_route_request_body(group_df: pd.DataFrame) -> dict[str, Any]:
    """
    Build one Google Routes request body for one adjacency group.
    """
    first_row = group_df.iloc[0]
    last_row = group_df.iloc[-1]

    origin = {
        "location": {
            "latLng": {
                "latitude": float(first_row["start_lat"]),
                "longitude": float(first_row["start_lon"]),
            }
        }
    }

    destination = {
        "location": {
            "latLng": {
                "latitude": float(last_row["end_lat"]),
                "longitude": float(last_row["end_lon"]),
            }
        }
    }

    intermediates: list[dict[str, Any]] = []

    if len(group_df) >= 2:
        for row_index in range(1, len(group_df)):
            row = group_df.iloc[row_index]
            intermediates.append(
                {
                    "location": {
                        "latLng": {
                            "latitude": float(row["start_lat"]),
                            "longitude": float(row["start_lon"]),
                        }
                    }
                }
            )

    body: dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
    }

    if intermediates:
        body["intermediates"] = intermediates

    return body


def post_google_route_request(
    api_key: str,
    body: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Send one Google Routes API request.
    """
    headers = {
        "X-Goog-FieldMask": "routes.legs.distanceMeters,routes.legs.duration"
    }

    try:
        response = SESSION.post(
            f"{GOOGLE_ROUTES_API_URL}?key={api_key}",
            headers=headers,
            json=body,
            timeout=GOOGLE_ROUTES_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as exc:
        return None, f"Google Routes request failed: {exc}"

    try:
        response_json = response.json()
    except Exception as exc:
        return None, f"Google Routes JSON parsing failed: {exc}"

    if "error" in response_json:
        return None, f"Google Routes API returned an error: {response_json['error']}"

    return response_json, None


def convert_group_route_to_segment_results(
    group_df: pd.DataFrame,
    route_json: dict[str, Any] | None,
) -> pd.DataFrame:
    """
    Convert one route response into per-segment speed rows.
    """
    result_rows: list[dict[str, Any]] = []

    if not route_json or "routes" not in route_json or not route_json["routes"]:
        for _, row in group_df.iterrows():
            result_rows.append(
                {
                    "segment_id": str(row["segment_id"]).strip(),
                    "google_speed_kmh": pd.NA,
                    "google_duration_seconds": pd.NA,
                }
            )
        return pd.DataFrame(result_rows)

    route = route_json["routes"][0]
    legs = route.get("legs", [])

    for row_index in range(len(group_df)):
        row = group_df.iloc[row_index]

        if row_index >= len(legs):
            result_rows.append(
                {
                    "segment_id": str(row["segment_id"]).strip(),
                    "google_speed_kmh": pd.NA,
                    "google_duration_seconds": pd.NA,
                }
            )
            continue

        leg = legs[row_index]
        distance_meters = leg.get("distanceMeters")
        duration_seconds = parse_duration_to_seconds(leg.get("duration"))

        speed_kmh = pd.NA

        if distance_meters is not None and duration_seconds is not None:
            if duration_seconds == 0 and float(distance_meters) < 20:
                duration_seconds = 1

            if duration_seconds > 0:
                speed_kmh = (float(distance_meters) / 1000.0) / (float(duration_seconds) / 3600.0)

        result_rows.append(
            {
                "segment_id": str(row["segment_id"]).strip(),
                "google_speed_kmh": speed_kmh,
                "google_duration_seconds": duration_seconds if duration_seconds is not None else pd.NA,
            }
        )

    return pd.DataFrame(result_rows)


def build_empty_google_fetch_result(
    message: str | None = None,
) -> dict[str, Any]:
    """
    Standard empty result payload for the Brussels page.
    """
    return {
        "google_results_df": pd.DataFrame(
            columns=["segment_id", "google_speed_kmh", "google_duration_seconds"]
        ),
        "diagnostics": {
            "selected_segment_count": 0,
            "group_count": 0,
            "request_count_planned": 0,
            "request_count_sent": 0,
            "success_count": 0,
            "failure_count": 0,
            "usage_month_key": get_month_key_utc(),
            "usage_monthly_limit": GOOGLE_ROUTES_MONTHLY_LIMIT,
            "usage_used_before_run": get_monthly_google_request_count(),
            "usage_remaining_before_run": GOOGLE_ROUTES_MONTHLY_LIMIT - get_monthly_google_request_count(),
            "was_requested": False,
            "error_message": message,
            "info_message": message,
        },
    }


def fetch_google_speeds_for_selected_segments(
    selected_gdf: pd.DataFrame,
    monthly_limit: int = GOOGLE_ROUTES_MONTHLY_LIMIT,
) -> dict[str, Any]:
    """
    Fetch live Google Routes speeds for the selected Brussels segments only.

    Rules enforced:
    - no selection -> no request
    - only filtered subset is grouped
    - persistent monthly request counting
    - hard limit check before execution
    """
    if selected_gdf.empty:
        return build_empty_google_fetch_result(
            message="No segments selected for Google Routes. No Google request was sent."
        )

    api_key = get_google_routes_api_key()
    if not api_key:
        return build_empty_google_fetch_result(
            message="Missing Google Routes API key in Streamlit secrets."
        )

    selected_segments_df = extract_google_ready_segments(selected_gdf)

    if selected_segments_df.empty:
        return build_empty_google_fetch_result(
            message="Selected segments do not contain usable Google Routes coordinates."
        )

    groups = build_groups_for_selected_segments(selected_segments_df)
    planned_request_count = len(groups)

    if planned_request_count == 0:
        return build_empty_google_fetch_result(
            message="No Google Routes groups could be built from the selected segments."
        )

    can_execute, used_before_run, remaining_before_run = can_consume_google_requests(
        planned_request_count=planned_request_count,
        monthly_limit=monthly_limit,
    )

    if not can_execute:
        return {
            "google_results_df": pd.DataFrame(
                columns=["segment_id", "google_speed_kmh", "google_duration_seconds"]
            ),
            "diagnostics": {
                "selected_segment_count": int(len(selected_segments_df)),
                "group_count": int(len(groups)),
                "request_count_planned": int(planned_request_count),
                "request_count_sent": 0,
                "success_count": 0,
                "failure_count": 0,
                "usage_month_key": get_month_key_utc(),
                "usage_monthly_limit": int(monthly_limit),
                "usage_used_before_run": int(used_before_run),
                "usage_remaining_before_run": int(remaining_before_run),
                "was_requested": False,
                "error_message": (
                    "Google Routes monthly request limit would be exceeded. "
                    "No Google request was sent."
                ),
                "info_message": None,
            },
        }

    result_frames: list[pd.DataFrame] = []
    success_count = 0
    failure_count = 0

    for group_df in groups:
        body = build_google_route_request_body(group_df)
        route_json, error_message = post_google_route_request(
            api_key=api_key,
            body=body,
        )

        if error_message:
            failure_count += 1
        else:
            success_count += 1

        group_result_df = convert_group_route_to_segment_results(
            group_df=group_df,
            route_json=route_json,
        )
        result_frames.append(group_result_df)

    google_results_df = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame(
        columns=["segment_id", "google_speed_kmh", "google_duration_seconds"]
    )

    if not google_results_df.empty:
        google_results_df["segment_id"] = google_results_df["segment_id"].astype(str).str.strip()
        google_results_df["google_speed_kmh"] = pd.to_numeric(
            google_results_df["google_speed_kmh"],
            errors="coerce",
        )
        google_results_df["google_duration_seconds"] = pd.to_numeric(
            google_results_df["google_duration_seconds"],
            errors="coerce",
        )

    append_google_usage_log_row(
        request_count=planned_request_count,
        group_count=len(groups),
        selected_segment_count=len(selected_segments_df),
        success_count=success_count,
        failure_count=failure_count,
    )

    diagnostics = {
        "selected_segment_count": int(len(selected_segments_df)),
        "group_count": int(len(groups)),
        "request_count_planned": int(planned_request_count),
        "request_count_sent": int(planned_request_count),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
        "usage_month_key": get_month_key_utc(),
        "usage_monthly_limit": int(monthly_limit),
        "usage_used_before_run": int(used_before_run),
        "usage_remaining_before_run": int(remaining_before_run),
        "was_requested": True,
        "error_message": None,
        "info_message": None,
    }

    return {
        "google_results_df": google_results_df,
        "diagnostics": diagnostics,
    }


def attach_google_results_to_map_gdf(
    gdf: pd.DataFrame,
    google_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach Google speed results to the Brussels map GeoDataFrame.
    """
    result = gdf.copy()

    if "id" not in result.columns:
        result["google_speed"] = pd.NA
        result["google_duration_seconds"] = pd.NA
        return result

    if google_results_df.empty:
        result["google_speed"] = pd.NA
        result["google_duration_seconds"] = pd.NA
        return result

    working_google_df = google_results_df.copy()
    working_google_df["segment_id"] = working_google_df["segment_id"].astype(str).str.strip()

    result["segment_id_str"] = result["id"].astype(str).str.strip()

    speed_lookup = dict(
        zip(working_google_df["segment_id"], working_google_df["google_speed_kmh"])
    )
    duration_lookup = dict(
        zip(working_google_df["segment_id"], working_google_df["google_duration_seconds"])
    )

    result["google_speed"] = result["segment_id_str"].map(speed_lookup)
    result["google_duration_seconds"] = result["segment_id_str"].map(duration_lookup)

    return result


def attach_google_results_to_snapshot_df(
    snapshot_df: pd.DataFrame,
    google_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Google speeds into the results dataframe.
    """
    result = snapshot_df.copy()

    if result.empty:
        result["google_speed_kmh"] = pd.NA
        result["google_duration_seconds"] = pd.NA
        return result

    if "segment_id" not in result.columns:
        raise ValueError("snapshot_df must contain 'segment_id'.")

    if google_results_df.empty:
        result["google_speed_kmh"] = pd.NA
        result["google_duration_seconds"] = pd.NA
        return result

    working_google_df = google_results_df.copy()
    working_google_df["segment_id"] = working_google_df["segment_id"].astype(str).str.strip()

    result["segment_id"] = result["segment_id"].astype(str).str.strip()

    merged_df = result.merge(
        working_google_df,
        on="segment_id",
        how="left",
    )

    return merged_df
