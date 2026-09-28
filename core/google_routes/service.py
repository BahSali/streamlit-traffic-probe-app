from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import gspread
import pandas as pd
import requests
import streamlit as st

from core import config
from core.timing import timed


GOOGLE_ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_ROUTES_TIMEOUT_SECONDS = 20
GOOGLE_ROUTES_MAX_PARALLEL = 6

MAX_GAP_METERS = 15.0
MAX_TURN_DEG = 30.0
MAX_GROUP_SIZE = 17

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


@st.cache_resource
def get_google_sheet():
    gc = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
    spreadsheet = gc.open_by_key(st.secrets["sheets"]["spreadsheet_id"])
    worksheet = spreadsheet.worksheet(st.secrets["sheets"]["worksheet_name"])
    return worksheet


# Monthly usage worksheet (secrets [sheets]), header: month_key | request_count.
# It is an append-only log: every Google batch appends one row with the
# number of requests it reserves, and a month's usage is the sum of that
# month's rows (an older single total row per month still counts).
#
# Why a log: Sheets has no transactions. With read-then-update, two users
# pressing RUN together can both see room for their batch and both send
# (tested: 11 requests sent with a limit of 10). Appends with INSERT_ROWS
# get distinct rows, and a batch only counts the rows above its own, so
# concurrent batches are ordered and at most one of them gets the last
# requests. A denied batch sets its row to 0.


def configured_monthly_limit() -> int:
    return int(config.GOOGLE_ROUTES_MONTHLY_LIMIT)


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _month_usage(values: list[list[Any]], month_key: str, before_row: int | None = None) -> int:
    """Sum request_count of month_key rows (1-based sheet rows, header in row 1)."""
    total = 0
    last_row = len(values) if before_row is None else before_row - 1
    for row_number in range(2, last_row + 1):
        row = values[row_number - 1]
        if len(row) >= 2 and str(row[0]).strip() == month_key:
            total += _to_int(row[1])
    return total


def read_usage_state() -> dict[str, Any]:
    """
    Read this month's Google request usage (one sheet read, no write).
    """
    ws = get_google_sheet()
    month_key = get_current_month_key()
    return {
        "month_key": month_key,
        "request_count": _month_usage(ws.get_all_values(), month_key),
    }


def get_monthly_google_request_count() -> int:
    with timed("google_sheets.read_count"):
        return int(read_usage_state()["request_count"])


def _appended_row_number(response: dict[str, Any]) -> int:
    updated_range = response["updates"]["updatedRange"]  # e.g. "'Usage'!A7:B7"
    return int(re.search(r"![A-Z]+(\d+)", updated_range).group(1))


def reserve_google_requests(planned: int, limit: int | None = None) -> dict[str, Any]:
    """Reserve ``planned`` requests for this month before sending any.

    Returns month_key, allowed, used_before (requests reserved by rows above
    this one) and row (this reservation's sheet row).
    """
    limit = configured_monthly_limit() if limit is None else int(limit)
    ws = get_google_sheet()
    month_key = get_current_month_key()

    with timed("google_sheets.reserve", planned=planned):
        response = ws.append_row(
            [month_key, int(planned)],
            value_input_option="RAW",
            insert_data_option="INSERT_ROWS",
        )
        row = _appended_row_number(response)
        used_before = _month_usage(ws.get_all_values(), month_key, before_row=row)
        allowed = used_before + planned <= limit
        if not allowed:
            ws.update(f"B{row}", [[0]])

    return {"month_key": month_key, "allowed": allowed, "used_before": used_before, "row": row}


def record_attempted_requests(reservation: dict[str, Any], planned: int, attempted: int) -> None:
    """Correct a reservation when fewer requests were attempted than planned."""
    if attempted != planned:
        with timed("google_sheets.write_count"):
            get_google_sheet().update(f"B{reservation['row']}", [[int(attempted)]])


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


def parse_duration_to_seconds(duration_value: str | None) -> int | float | None:
    """Parse a Routes API duration such as "165s" or "3.5s" (int when whole)."""
    if not isinstance(duration_value, str):
        return None

    if not duration_value.endswith("s"):
        return None

    try:
        seconds = float(duration_value[:-1])
    except ValueError:
        return None
    return int(seconds) if seconds.is_integer() else seconds


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


GOOGLE_RESULT_COLUMNS = ["segment_id", "google_speed_kmh", "google_duration_seconds"]


def convert_group_response_to_rows(
    group_df: pd.DataFrame,
    response_payload: dict[str, Any] | None,
) -> pd.DataFrame:
    """One row per segment of the group.

    google_speed_kmh / google_duration_seconds are what the app displays.
    distance_m / duration_s are the leg values exactly as returned (empty when
    not returned); they are only used for the private observation log.
    """
    rows: list[dict[str, Any]] = []
    legs = []
    if response_payload and response_payload.get("routes"):
        legs = response_payload["routes"][0].get("legs", [])

    for row_index in range(len(group_df)):
        segment_id = str(group_df.iloc[row_index]["segment_id"]).strip()
        leg = legs[row_index] if row_index < len(legs) else {}
        distance_meters = leg.get("distanceMeters")
        returned_duration = parse_duration_to_seconds(leg.get("duration"))
        duration_seconds = returned_duration

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
                "segment_id": segment_id,
                "google_speed_kmh": google_speed_kmh,
                "google_duration_seconds": duration_seconds if duration_seconds is not None else pd.NA,
                "distance_m": distance_meters if distance_meters is not None else pd.NA,
                "duration_s": returned_duration if returned_duration is not None else pd.NA,
            }
        )

    return pd.DataFrame(rows)


def _diagnostics(
    *,
    month_key: str | None,
    limit: int,
    used_before: int,
    used_after: int | None = None,
    selected_segment_count: int = 0,
    group_count: int = 0,
    planned: int = 0,
    sent: int = 0,
    success: int = 0,
    failure: int = 0,
    error_message: str | None = None,
    info_message: str | None = None,
) -> dict[str, Any]:
    used_after = used_before if used_after is None else used_after
    return {
        "selected_segment_count": int(selected_segment_count),
        "group_count": int(group_count),
        "request_count_planned": int(planned),
        "request_count_sent": int(sent),
        "success_count": int(success),
        "failure_count": int(failure),
        "usage_month_key": month_key,
        "usage_monthly_limit": int(limit),
        "usage_used_before_run": int(used_before),
        "usage_remaining_before_run": int(limit) - int(used_before),
        "usage_used_after_run": int(used_after),
        "usage_remaining_after_run": int(limit) - int(used_after),
        "was_requested": sent > 0,
        "error_message": error_message,
        "info_message": info_message,
    }


def build_empty_google_result(
    message: str | None = None,
    used_before_run: int | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    if used_before_run is None:
        used_before_run = get_monthly_google_request_count()

    return {
        "google_results_df": pd.DataFrame(columns=GOOGLE_RESULT_COLUMNS),
        "observation_batch": None,
        "diagnostics": _diagnostics(
            month_key=get_current_month_key(),
            limit=configured_monthly_limit(),
            used_before=used_before_run,
            error_message=error_message,
            info_message=message,
        ),
    }


def limit_exceeded_message(planned: int, used: int, limit: int, month_key: str) -> str:
    return (
        f"Google Routes monthly limit reached: this RUN needs {planned} request(s), but "
        f"{used} of {limit} are already used in {month_key} (UTC), leaving {max(0, limit - used)}. "
        "No Google request was sent."
    )


def fetch_google_speeds_for_selected_segments(
    selected_gdf: pd.DataFrame,
    monthly_limit: int | None = None,
) -> dict[str, Any]:
    """
    Google fetch for the selected subset only.

    Returns google_results_df (displayed values), diagnostics, and
    observation_batch: the batch timestamp and the returned values per
    segment, or None when no request was sent.
    """
    limit = configured_monthly_limit() if monthly_limit is None else int(monthly_limit)

    if selected_gdf.empty:
        return build_empty_google_result(
            "No segments selected for Google Routes. No Google request was sent."
        )

    api_key = get_google_routes_api_key()
    if not api_key:
        return build_empty_google_result(
            error_message="Missing Google Maps / Routes API key in Streamlit secrets."
        )

    try:
        selected_segments_df = extract_google_ready_segments(selected_gdf)
    except Exception as exc:
        return build_empty_google_result(error_message=f"Google segment preparation failed: {exc}")

    if selected_segments_df.empty:
        return build_empty_google_result(
            "Selected segments do not have usable start/end coordinates for Google Routes."
        )

    groups = build_groups_for_selected_segments(selected_segments_df)
    planned_request_count = len(groups)

    if planned_request_count == 0:
        return build_empty_google_result(
            "No Google Routes groups could be built from the selected subset."
        )

    # Reserve the planned HTTP requests (one per group) before sending any.
    reservation = reserve_google_requests(planned_request_count, limit=limit)
    used_before_run = reservation["used_before"]
    month_key = reservation["month_key"]

    if not reservation["allowed"]:
        result = build_empty_google_result(used_before_run=used_before_run)
        result["diagnostics"] = _diagnostics(
            month_key=month_key,
            limit=limit,
            used_before=used_before_run,
            selected_segment_count=len(selected_segments_df),
            group_count=len(groups),
            planned=planned_request_count,
            error_message=limit_exceeded_message(planned_request_count, used_before_run, limit, month_key),
        )
        return result

    batch_started_at = datetime.now(timezone.utc)

    # One request per group, sent concurrently; each group is requested exactly once.
    with timed("google_routes.requests", requests=len(groups), segments=len(selected_segments_df)):
        with ThreadPoolExecutor(max_workers=min(GOOGLE_ROUTES_MAX_PARALLEL, len(groups))) as executor:
            responses = list(
                executor.map(
                    lambda group_df: send_google_route_request(
                        api_key=api_key,
                        body=build_google_request_body(group_df),
                    ),
                    groups,
                )
            )

    # Every request was attempted (failures included) and counts toward the limit.
    attempted = len(responses)
    record_attempted_requests(reservation, planned_request_count, attempted)

    failure_count = sum(1 for _, error_message in responses if error_message)
    success_count = attempted - failure_count
    rows_df = pd.concat(
        [
            convert_group_response_to_rows(group_df=group_df, response_payload=response_payload)
            for group_df, (response_payload, _) in zip(groups, responses)
        ],
        ignore_index=True,
    )

    google_results_df = rows_df[GOOGLE_RESULT_COLUMNS].copy()
    google_results_df["segment_id"] = google_results_df["segment_id"].astype(str).str.strip()
    google_results_df["google_speed_kmh"] = pd.to_numeric(google_results_df["google_speed_kmh"], errors="coerce")
    google_results_df["google_duration_seconds"] = pd.to_numeric(
        google_results_df["google_duration_seconds"], errors="coerce"
    )

    observations = pd.DataFrame(
        {
            "segment_id": google_results_df["segment_id"],
            "distance_m": pd.to_numeric(rows_df["distance_m"], errors="coerce"),
            "duration_s": pd.to_numeric(rows_df["duration_s"], errors="coerce"),
            "speed_kmh": google_results_df["google_speed_kmh"],
        }
    )

    return {
        "google_results_df": google_results_df,
        "observation_batch": {"started_at": batch_started_at, "observations": observations},
        "diagnostics": _diagnostics(
            month_key=month_key,
            limit=limit,
            used_before=used_before_run,
            used_after=used_before_run + attempted,
            selected_segment_count=len(selected_segments_df),
            group_count=len(groups),
            planned=planned_request_count,
            sent=attempted,
            success=success_count,
            failure=failure_count,
        ),
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
