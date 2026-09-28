"""Authoritative monthly Google Routes request counter with atomic reservations.

Google Sheets cannot enforce a cap across app instances: it has no
transactions or conditional writes, so "read the count, then write" lets two
simultaneous RUNs both pass the check. This counter lives instead in one
Google Cloud Storage object per month (UTC)::

    gs://<bucket>/<prefix>/<YYYY-MM>.json
    {"month": "2026-09", "used": 12, "seeded_from_sheet": 3,
     "reservations": {"<id>": {"planned": 4, "attempted": 4,
                              "reserved_at": "...", "settled_at": "..."}}}

Every change is a compare-and-swap: read the object and its generation, then
write with ``ifGenerationMatch=<that generation>``. GCS rejects the write
(HTTP 412) if any other instance changed the object in between, and the
change is retried from a fresh read. A month's object is created with
``ifGenerationMatch=0`` (only if it does not exist yet). So a reservation
is granted only against the latest count, and granted reservations can
never add up to more than the limit.

- reserve() takes the planned requests out of the allowance before anything
  is sent, or refuses without writing.
- settle() records how many requests were actually attempted (failures
  included) and gives back the unattempted rest. It is idempotent.
- If an instance dies between reserve() and settle(), its reservation stays
  counted in full: the counter can overstate usage, never understate it.
- A new month's counter starts from ``seed(month)``: the value already in
  the legacy usage spreadsheet for that month (read-only).
"""
from __future__ import annotations

import json
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from google.api_core import exceptions as gexc


class UsageStoreError(RuntimeError):
    """The counter could not be read or updated; the message is safe to show."""


def next_month_start(month_key: str) -> str:
    year, month = map(int, month_key.split("-"))
    year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return f"{year:04d}-{month:02d}-01"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def describe_error(exc: BaseException) -> str:
    """A short, credential-free description of a storage failure."""
    if isinstance(exc, UsageStoreError):
        return str(exc)
    if isinstance(exc, gexc.Forbidden):
        return "the service account has no access to the usage bucket (HTTP 403)"
    if isinstance(exc, gexc.NotFound):
        return "the usage bucket was not found (HTTP 404)"
    if isinstance(exc, gexc.GoogleAPICallError):
        return f"Google Cloud Storage error (HTTP {exc.code})"
    return f"{type(exc).__name__} while contacting Google Cloud Storage"


class GcsUsageStore:
    def __init__(
        self,
        bucket,
        prefix: str = "google-routes-usage",
        seed: Callable[[str], int] = lambda month_key: 0,
        max_attempts: int = 50,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.seed = seed
        self.max_attempts = max_attempts

    def _path(self, month_key: str) -> str:
        return f"{self.prefix}/{month_key}.json"

    def _load(self, month_key: str) -> tuple[dict | None, int]:
        blob = self.bucket.get_blob(self._path(month_key))
        if blob is None:
            return None, 0
        data = blob.download_as_bytes(if_generation_match=blob.generation)
        return json.loads(data), int(blob.generation)

    def _new_document(self, month_key: str) -> dict:
        seeded = int(self.seed(month_key))
        return {"month": month_key, "used": seeded, "seeded_from_sheet": seeded, "reservations": {}}

    def _update(self, month_key: str, change: Callable[[dict], tuple[bool, Any]]) -> Any:
        """Apply change(doc) -> (write?, result) atomically; retry on conflicts."""
        try:
            for attempt in range(self.max_attempts):
                try:
                    doc, generation = self._load(month_key)
                    if doc is None:
                        doc = self._new_document(month_key)
                    write, result = change(doc)
                    if not write:
                        return result
                    self.bucket.blob(self._path(month_key)).upload_from_string(
                        json.dumps(doc, sort_keys=True),
                        content_type="application/json",
                        if_generation_match=generation,
                    )
                    return result
                except gexc.PreconditionFailed:
                    # Another instance changed (or created) the object: start again from a fresh read.
                    time.sleep(random.uniform(0, 0.05 * (attempt + 1)))
            raise UsageStoreError("the usage counter is busy (too many simultaneous updates)")
        except UsageStoreError:
            raise
        except Exception as exc:
            raise UsageStoreError(describe_error(exc)) from exc

    def used(self, month_key: str) -> int:
        try:
            doc, _ = self._load(month_key)
            return int(doc["used"]) if doc else int(self.seed(month_key))
        except UsageStoreError:
            raise
        except Exception as exc:
            raise UsageStoreError(describe_error(exc)) from exc

    def reserve(self, month_key: str, planned: int, limit: int) -> dict[str, Any]:
        reservation_id = uuid.uuid4().hex

        def change(doc: dict) -> tuple[bool, dict]:
            used = int(doc["used"])
            if used + planned > limit:
                return False, {"allowed": False, "used_before": used}
            doc["used"] = used + planned
            doc["reservations"][reservation_id] = {
                "planned": planned, "attempted": None, "reserved_at": _now(), "settled_at": None,
            }
            return True, {"allowed": True, "used_before": used}

        result = self._update(month_key, change)
        return {**result, "month_key": month_key, "reservation_id": reservation_id, "planned": planned}

    def settle(self, reservation: dict[str, Any], attempted: int) -> int:
        """Record the attempted requests (<= planned); returns the month's usage."""
        rid = reservation["reservation_id"]

        def change(doc: dict) -> tuple[bool, int]:
            entry = doc["reservations"].get(rid)
            if entry is None or entry["attempted"] is not None:
                return False, int(doc["used"])  # unknown or already settled
            counted = min(int(attempted), int(entry["planned"]))
            doc["used"] = int(doc["used"]) - (int(entry["planned"]) - counted)
            entry["attempted"] = counted
            entry["settled_at"] = _now()
            return True, int(doc["used"])

        return self._update(reservation["month_key"], change)
