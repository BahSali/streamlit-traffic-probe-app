from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


GOOGLE_ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_ROUTES_TIMEOUT_SECONDS = 20
GOOGLE_ROUTES_MONTHLY_LIMIT = 5000

MAX_GAP_METERS = 15.0
MAX_TURN_DEG = 30.0
MAX_GROUP_SIZE = 17

USAGE_FILE_PATH = Path("یعب")

SESSION = requests.Session()


def get_google_routes_api_key() -> str | None:
    """
    Read the Google Maps / Routes API key from Streamlit secrets.

    Supports both names to avoid config mismatch:
    - GOOGLE_MAPS_API_KEY
    - GOOGLE_ROUTES_API_KEY
    """
    return (
        st.secrets.get("GOOGLE_MAPS_API_KEY")
        or st.secrets.get("GOOGLE_ROUTES_API_KEY")
    )


def get_current_month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def ensure_usage_file_exists() -> None:
    USAGE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not USAGE_FILE_PATH.exists():
        USAGE_FILE_PATH.write_text(
            json.dumps(
                {
                    "month_key": get_current_month_key(),
                    "request_count": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def read_usage_state() -> dict[str, Any]:
    """
    Read the persistent monthly Google request counter.

    If the saved month is not the current month, reset automatically.
    """
    ensure_usage_file_exists()

    try:
        state = json.loads(USAGE_FILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        state = {
            "month_key": get_current_month_key(),
            "request_count": 0,
        }

    current_month_key = get_current_month_key()

    if state.get("month_key") != current_month_key:
        state = {
            "month_key": current_month_key,
            "request_count": 0,
        }
        write_usage_state(state)

    request_count = int(state.get("request_count", 0))

    return {
        "month_key": current_month_key,
        "request_count": request_count,
    }


def write_usage_state(state: dict[str, Any]) -> None:
    ensure_usage_file_exists()
    USAGE_FILE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_monthly_google_request_count() -> int:
    return int(read_usage_state()["request_count"])


def increment_monthly_google_request_count(increment: int) -> dict[str, Any]:
    """
    Increase the monthly request counter by the number of requests actually sent.
    """
    state = read_usage_state()
    state["request_count"] = int(state.get("request_count", 0)) + int(increment)
    write_usage_state(state)
    return state


def can_send_google_requests(
    planned_request_count: int,
    monthly_limit: int = GOOGLE_ROUTES_MONTHLY_LIMIT,
) -> tuple[bool, int, int]:
    used_count = get_monthly_google_request_count()
    remaining_count = int(monthly_limit) - int(used_count)
    return planned_request_count <= remaining_count, used_count, remaining_count


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def angle_between(
    vector_1: tuple[float, float],
    vector_2: tuple[float, float],
) -> float:
    x1, y1 = vector_1
    x2, y2 = vector_2

    dot_product = x1 * x2 + y1 * y2
    norm_1 = math.hypot(x1, y1)
    norm_2 = math.hypot(x2, y2)

    if norm_1 == 0 or norm_2 == 0:
        return 0.0

    cosine_angle = max(min(dot_product / (norm_1 * norm_2), 1.0), -1.0)
    return math.degrees(math.acos(cosine_angle))


def extract_google_ready_segments(selected_gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the selected subset. No whole-network processing is done here.
    """
    required_columns = {
        "id",
        "map_fid",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
    }
    missing_columns = required_columns - set(selected_gdf.columns)

    if missing_columns:
        raise ValueError(
            f"Selected segments are missing required Google columns: {sorted(missing_columns)}"
        )

    result = selected_gdf.copy()
    result["segment_id"] = result["id"].astype(str).str.strip()
    result["map_fid"] = pd.to_numeric(result["map_fid"], errors="coerce")
    result["start_lat"] = pd.to_numeric(result["start_lat"], errors="coerce")
    result["start_lon"] = pd.to_numeric(result["start_lon"], errors="coerce")
    result["end_lat"] = pd.to_numeric(result["end_lat"], errors="coerce")
    result["end_lon"] = pd.to_numeric(result["end_lon"], errors="coerce")

    result = result.dropna(
        subset=["segment_id", "map_fid", "start_lat", "start_lon", "end_lat", "end_lon"]
    ).copy()

    result = result.sort_values("map_fid").reset_index(drop=True)

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
    Build adjacency groups using only the filtered subset.
    """
    if selected_segments_df.empty:
        return []

    def build_vector(row: pd.Series) -> tuple[float, float]:
        return (
            float(row["end_lon"]) - float(row["start_lon"]),
            float(row["end_lat"]) - float(row["start_lat"]),
        )

    groups: list[list[int]] = []
    current_group = [0]

    for row_index in range(1, len(selected_segments_df)):
        previous_row = selected_segments_df.iloc[current_group[-1]]
        current_row = selected_segments_df.iloc[row_index]

        gap_meters = haversine_m(
            float(previous_row["end_lat"]),
            float(previous_row["end_lon"]),
            float(current_row["start_lat"]),
            float(current_row["start_lon"]),
        )

        turn_deg = angle_between(
            build_vector(previous_row),
            build_vector(current_row),
        )

        is_adjacent = (
            gap_meters <= max_gap_meters
            and turn_deg <= max_turn_deg
            and len(current_group) < max_group_size
        )

        if is_adjacent:
            current_group.append(row_index)
        else:
            groups.append(current_group)
            current_group = [row_index]

    groups.append(current_group)

    return [
        selected_segments_df.iloc[group_indices].reset_index(drop=True)
        for group_indices in groups
    ]


def parse_duration_to_seconds(duration_value: str | None) -> int | None:
    if not isinstance(duration_value, str):
        return None

    if not duration_value.endswith("s"):
        return None

    try:
        return int(duration_value[:-1])
    except Exception:
        return None


def build_google_request_body(group_df: pd.DataFrame) -> dict[str, Any]:
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

    intermediates = []
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


def send_google_route_request(
    api_key: str,
    body: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
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
        payload = response.json()
    except Exception as exc:
        return None, f"Google Routes JSON parsing failed: {exc}"

    if "error" in payload:
        return None, f"Google Routes API error: {payload['error']}"

    return payload, None


def convert_group_response_to_rows(
    group_df: pd.DataFrame,
    response_payload: dict[str, Any] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not response_payload or "routes" not in response_payload or not response_payload["routes"]:
        for _, row in group_df.iterrows():
            rows.append(
                {
                    "segment_id": str(row["segment_id"]).strip(),
                    "google_speed_kmh": pd.NA,
                    "google_duration_seconds": pd.NA,
                }
            )
        return pd.DataFrame(rows)

    route = response_payload["routes"][0]
    legs = route.get("legs", [])

    for row_index in range(len(group_df)):
        row = group_df.iloc[row_index]

        if row_index >= len(legs):
            rows.append(
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

        google_speed_kmh = pd.NA

        if distance_meters is not None and duration_seconds is not None:
            if duration_seconds == 0 and float(distance_meters) < 20:
                duration_seconds = 1

            if duration_seconds > 0:
                google_speed_kmh = (
                    (float(distance_meters) / 1000.0)
                    / (float(duration_seconds) / 3600.0)
                )

        rows.append(
            {
                "segment_id": str(row["segment_id"]).strip(),
                "google_speed_kmh": google_speed_kmh,
                "google_duration_seconds": duration_seconds if duration_seconds is not None else pd.NA,
            }
        )

    return pd.DataFrame(rows)


def build_empty_google_result(message: str | None = None) -> dict[str, Any]:
    used_before_run = get_monthly_google_request_count()
    remaining_before_run = GOOGLE_ROUTES_MONTHLY_LIMIT - used_before_run

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
            "usage_month_key": get_current_month_key(),
            "usage_monthly_limit": GOOGLE_ROUTES_MONTHLY_LIMIT,
            "usage_used_before_run": used_before_run,
            "usage_remaining_before_run": remaining_before_run,
            "usage_used_after_run": used_before_run,
            "usage_remaining_after_run": remaining_before_run,
            "was_requested": False,
            "error_message": None,
            "info_message": message,
        },
    }


def fetch_google_speeds_for_selected_segments(
    selected_gdf: pd.DataFrame,
    monthly_limit: int = GOOGLE_ROUTES_MONTHLY_LIMIT,
) -> dict[str, Any]:
    """
    Google fetch for the selected subset only.
    """
    if selected_gdf.empty:
        return build_empty_google_result(
            "No segments selected for Google Routes. No Google request was sent."
        )

    api_key = get_google_routes_api_key()
    if not api_key:
        result = build_empty_google_result()
        result["diagnostics"]["error_message"] = "Missing Google Maps / Routes API key in Streamlit secrets."
        result["diagnostics"]["info_message"] = None
        return result

    try:
        selected_segments_df = extract_google_ready_segments(selected_gdf)
    except Exception as exc:
        result = build_empty_google_result()
        result["diagnostics"]["error_message"] = f"Google segment preparation failed: {exc}"
        result["diagnostics"]["info_message"] = None
        return result

    if selected_segments_df.empty:
        return build_empty_google_result(
            "Selected segments do not have usable start/end coordinates for Google Routes."
        )

    groups = build_groups_for_selected_segments(selected_segments_df)
    planned_request_count = len(groups)

    used_before_run = get_monthly_google_request_count()
    remaining_before_run = int(monthly_limit) - int(used_before_run)

    if planned_request_count == 0:
        return build_empty_google_result(
            "No Google Routes groups could be built from the selected subset."
        )

    is_allowed, _, _ = can_send_google_requests(
        planned_request_count=planned_request_count,
        monthly_limit=monthly_limit,
    )

    if not is_allowed:
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
                "usage_month_key": get_current_month_key(),
                "usage_monthly_limit": int(monthly_limit),
                "usage_used_before_run": int(used_before_run),
                "usage_remaining_before_run": int(remaining_before_run),
                "usage_used_after_run": int(used_before_run),
                "usage_remaining_after_run": int(remaining_before_run),
                "was_requested": False,
                "error_message": "Google Routes monthly request limit would be exceeded. No Google request was sent.",
                "info_message": None,
            },
        }

    success_count = 0
    failure_count = 0
    result_frames: list[pd.DataFrame] = []

    for group_df in groups:
        body = build_google_request_body(group_df)
        response_payload, error_message = send_google_route_request(
            api_key=api_key,
            body=body,
        )

        if error_message:
            failure_count += 1
        else:
            success_count += 1

        result_frames.append(
            convert_group_response_to_rows(
                group_df=group_df,
                response_payload=response_payload,
            )
        )

    sent_request_count = len(groups)
    usage_state = increment_monthly_google_request_count(sent_request_count)

    google_results_df = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else pd.DataFrame(columns=["segment_id", "google_speed_kmh", "google_duration_seconds"])
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

    used_after_run = int(usage_state["request_count"])
    remaining_after_run = int(monthly_limit) - used_after_run

    return {
        "google_results_df": google_results_df,
        "diagnostics": {
            "selected_segment_count": int(len(selected_segments_df)),
            "group_count": int(len(groups)),
            "request_count_planned": int(planned_request_count),
            "request_count_sent": int(sent_request_count),
            "success_count": int(success_count),
            "failure_count": int(failure_count),
            "usage_month_key": get_current_month_key(),
            "usage_monthly_limit": int(monthly_limit),
            "usage_used_before_run": int(used_before_run),
            "usage_remaining_before_run": int(remaining_before_run),
            "usage_used_after_run": int(used_after_run),
            "usage_remaining_after_run": int(remaining_after_run),
            "was_requested": True,
            "error_message": None,
            "info_message": None,
        },
    }


def attach_google_results_to_map_gdf(
    gdf: pd.DataFrame,
    google_results_df: pd.DataFrame,
) -> pd.DataFrame:
    result = gdf.copy()

    if "id" not in result.columns:
        result["google_speed"] = pd.NA
        result["google_duration_seconds"] = pd.NA
        return result

    if google_results_df.empty:
        result["google_speed"] = pd.NA
        result["google_duration_seconds"] = pd.NA
        return result

    working_df = google_results_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str).str.strip()

    result["segment_id_str"] = result["id"].astype(str).str.strip()

    speed_lookup = dict(zip(working_df["segment_id"], working_df["google_speed_kmh"]))
    duration_lookup = dict(zip(working_df["segment_id"], working_df["google_duration_seconds"]))

    result["google_speed"] = result["segment_id_str"].map(speed_lookup)
    result["google_duration_seconds"] = result["segment_id_str"].map(duration_lookup)

    return result


def attach_google_results_to_snapshot_df(
    snapshot_df: pd.DataFrame,
    google_results_df: pd.DataFrame,
) -> pd.DataFrame:
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

    working_df = google_results_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str).str.strip()

    result["segment_id"] = result["segment_id"].astype(str).str.strip()

    return result.merge(
        working_df,
        on="segment_id",
        how="left",
    )
