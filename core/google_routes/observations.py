"""Private log of Google Routes observations, and the three wide CSV exports.

Google data may not be redistributed: these values are written only to a
separate private spreadsheet (secrets ``[google_observations]``) shared with
the owner's account and the service account, and exported only with
scripts/export_google_observations.py by someone holding those credentials.
Nothing here is shown on the page, logged, or committed.

Worksheet (row 1 is the header)::

    batch_id | batch_timestamp_utc | segment_id | distance_m | duration_s | speed_kmh

- One batch = the Google requests sent by one RUN. Reused (cached) Google
  results and RUNs stopped by the monthly limit send nothing and add nothing.
- A batch is appended in one API call: a marker row (empty segment_id) plus
  one row per segment for which Google returned a value. It is written
  completely or not at all.
- batch_timestamp_utc: when the batch's requests were sent, ISO 8601 in UTC
  with milliseconds, e.g. ``2026-01-15T07:07:31.123Z``.
- distance_m and duration_s are the route-leg ``distanceMeters`` and
  ``duration`` exactly as returned; speed_kmh is the speed the app derives
  (distance / duration; a 0 s leg shorter than 20 m is treated as 1 s, as on
  the map). A value Google did not return is left empty.
- Retries are idempotent: a batch whose batch_id is already present is not
  written again, and rows are never updated.

Recording is not lossless: a batch that cannot be written stays queued in the
browser session that ran it and is retried on each rerun; it is lost if that
session ends or the app restarts before a write succeeds.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("estimator.observations")


class ObservationStoreError(RuntimeError):
    """The observation log could not be written; the message is safe to show."""

HEADER = ["batch_id", "batch_timestamp_utc", "segment_id", "distance_m", "duration_s", "speed_kmh"]
QUANTITIES = ["distance_m", "duration_s", "speed_kmh"]
TIMESTAMP_COLUMN = "timestamp_utc"
EXPORT_FILE_NAMES = {
    "distance_m": "google_distance_m.csv",
    "duration_s": "google_duration_s.csv",
    "speed_kmh": "google_speed_kmh.csv",
}


def format_timestamp(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"


def new_batch(started_at: datetime, observations: pd.DataFrame) -> dict[str, Any]:
    """A batch ready to store: a unique id, its timestamp and the observed rows."""
    observed = observations[observations[QUANTITIES].notna().any(axis=1)]
    return {
        "batch_id": uuid.uuid4().hex,
        "batch_timestamp_utc": format_timestamp(started_at),
        "observations": observed[["segment_id", *QUANTITIES]].reset_index(drop=True),
    }


def _cell(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    return int(value) if value.is_integer() else value


def batch_rows(batch: dict[str, Any]) -> list[list[Any]]:
    rows = [[batch["batch_id"], batch["batch_timestamp_utc"], "", "", "", ""]]
    for record in batch["observations"].itertuples(index=False):
        rows.append(
            [
                batch["batch_id"],
                batch["batch_timestamp_utc"],
                str(record.segment_id),
                *(_cell(getattr(record, quantity)) for quantity in QUANTITIES),
            ]
        )
    return rows


def ensure_header(worksheet) -> None:
    current = worksheet.row_values(1)
    if not current:
        worksheet.append_row(HEADER, value_input_option="RAW")
    elif current[: len(HEADER)] != HEADER:
        raise ObservationStoreError(
            "the observation worksheet's first row is not the expected header; "
            "use a blank worksheet (the app writes the header)"
        )


def describe_error(exc: BaseException) -> str:
    """A short, credential-free reason for a failed observation write."""
    if isinstance(exc, ObservationStoreError):
        return str(exc)
    name = type(exc).__name__
    if name == "SpreadsheetNotFound":
        return "the observation spreadsheet was not found or is not shared with the service account"
    if name == "WorksheetNotFound":
        return "the worksheet named in [google_observations] worksheet_name does not exist"
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return f"Google Sheets API error (HTTP {status})"
    return f"{name} while writing to Google Sheets"


def persist_batch(worksheet, batch: dict[str, Any], attempts: int = 2, backoff_seconds: float = 0.5) -> None:
    """Append the batch unless it is already stored; raises ObservationStoreError if it cannot."""
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            if batch["batch_id"] in worksheet.col_values(1):
                return
            worksheet.append_rows(
                batch_rows(batch),
                value_input_option="RAW",
                insert_data_option="INSERT_ROWS",
            )
            logger.info(
                "stored Google observation batch %s (%d segments)",
                batch["batch_id"],
                len(batch["observations"]),
            )
            return
        except Exception as exc:  # network or API error: retry, the id check prevents duplicates
            last_error = exc
            logger.warning(
                "storing observation batch %s failed (attempt %d/%d): %s",
                batch["batch_id"], attempt, attempts, type(exc).__name__,
            )
            if attempt < attempts:
                time.sleep(backoff_seconds * attempt)
    raise ObservationStoreError(describe_error(last_error))


def read_observations(worksheet) -> pd.DataFrame:
    values = worksheet.get_all_values()
    if not values:
        return pd.DataFrame(columns=HEADER)
    return pd.DataFrame(values[1:], columns=values[0]).reindex(columns=HEADER)


def build_wide_exports(long_df: pd.DataFrame, segment_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Three aligned wide tables: timestamp_utc, then one column per segment id.

    One row per stored batch, in time order; only returned values are filled.
    """
    df = long_df.fillna("").astype(str)
    markers = df[df["segment_id"] == ""]
    batches = (
        markers.drop_duplicates("batch_id")[["batch_id", "batch_timestamp_utc"]]
        .sort_values(["batch_timestamp_utc", "batch_id"], kind="stable")
        .reset_index(drop=True)
    )

    observed = df[df["segment_id"] != ""].drop_duplicates(["batch_id", "segment_id"], keep="first")
    observed = observed[observed["batch_id"].isin(batches["batch_id"])]

    columns = [str(segment_id) for segment_id in segment_ids]
    exports = {}
    for quantity in QUANTITIES:
        values = pd.to_numeric(observed[quantity].replace("", pd.NA), errors="coerce")
        wide = (
            observed.assign(value=values)
            .pivot(index="batch_id", columns="segment_id", values="value")
            .reindex(index=batches["batch_id"], columns=columns)
        )
        wide.insert(0, TIMESTAMP_COLUMN, batches["batch_timestamp_utc"].to_numpy())
        exports[quantity] = wide.reset_index(drop=True)
    return exports


def write_exports(exports: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for quantity, frame in exports.items():
        path = out_dir / EXPORT_FILE_NAMES[quantity]
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def get_observation_worksheet():
    """The private observation worksheet; raises ObservationStoreError when unusable."""
    import gspread
    import streamlit as st

    def section(name):
        try:
            return st.secrets[name]
        except Exception:  # no secrets file, or no such section
            return None

    settings = section("google_observations")
    account = section("gcp_service_account")
    if settings is None or not settings.get("spreadsheet_id"):
        raise ObservationStoreError("[google_observations] spreadsheet_id is missing from the app's secrets")
    if account is None:
        raise ObservationStoreError("[gcp_service_account] is missing from the app's secrets")
    usage = section("sheets")
    if usage is not None and usage.get("spreadsheet_id") == settings["spreadsheet_id"]:
        raise ObservationStoreError(
            "[google_observations] points to the monthly usage spreadsheet; use the separate private spreadsheet"
        )
    try:
        return _open_worksheet(
            dict(account), settings["spreadsheet_id"], settings.get("worksheet_name", "observations"), gspread
        )
    except ObservationStoreError:
        raise
    except Exception as exc:
        raise ObservationStoreError(describe_error(exc)) from exc


def _open_worksheet(account: dict, spreadsheet_id: str, worksheet_name: str, gspread_module):
    client = gspread_module.service_account_from_dict(account)
    worksheet = client.open_by_key(spreadsheet_id).worksheet(worksheet_name)
    ensure_header(worksheet)
    return worksheet
