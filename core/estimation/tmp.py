from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


BRUSSELS_TIMEZONE = "Europe/Brussels"
PARQUET_COMPONENT = "stib_vehicle_distance_parquetize"
PARQUET_MARGIN_MINUTES = 30
PARQUET_BUCKET_MINUTES = 15
REQUEST_TIMEOUT_SECONDS = 45

# --------------------------------------------------
# Output guardrails
# --------------------------------------------------
MIN_VALID_SPEED_KMH = 5.0
MAX_VALID_SPEED_KMH = 51.0
DEFAULT_GLOBAL_SPEED_KMH = 20.0


class RefinerCNN(nn.Module):
    """
    Local inference copy of the training model.

    Input:
        x: [B, W, S]

    Output:
        y_hat: [B, S]
    """

    def __init__(
        self,
        window: int,
        num_streets: int,
        hidden: int = 64,
        delta_scale: float = 5.0,
        enforce_nonneg: bool = True,
        topk: int = 4,
    ) -> None:
        super().__init__()

        self.num_streets = int(num_streets)
        self.hidden = int(hidden)
        self.delta_scale = float(delta_scale)
        self.enforce_nonneg = bool(enforce_nonneg)
        self.topk = int(topk)

        self.backbone = nn.Sequential(
            nn.Conv1d(num_streets, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.base_pool = nn.AdaptiveAvgPool1d(1)
        self.base_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_streets),
        )

        self.ref_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_streets),
        )

        self.gate_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_streets),
        )

        self.gate_bias = nn.Parameter(torch.tensor(1.0))

    def _topk_pool(self, h_seq: torch.Tensor) -> torch.Tensor:
        k = min(self.topk, h_seq.shape[-1])
        topk_vals, _ = torch.topk(h_seq, k=k, dim=-1)
        return topk_vals.mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [B, W, S] -> [B, S, W]
        h_seq = self.backbone(x)

        h_base = self.base_pool(h_seq).squeeze(-1)
        y_base = self.base_head(h_base)

        h_ref = self._topk_pool(h_seq)
        gate_logits = self.gate_head(h_ref) + self.gate_bias
        gate = torch.sigmoid(gate_logits)

        delta_raw = self.ref_head(h_ref)
        delta = torch.tanh(delta_raw) * self.delta_scale

        y_hat = y_base + gate * delta

        if self.enforce_nonneg:
            return F.softplus(y_hat)

        return y_hat


def build_empty_estimation_diagnostics() -> dict[str, Any]:
    return {
        "estimation_mode": "pt_inference_historical_tmp",
        "snapshot_found": False,
        "snapshot_time": None,
        "snapshot_bucket_time": None,
        "map_has_id_column": False,
        "matched_segments": 0,
        "model_loaded": False,
        "model_path": None,
        "window_size": None,
        "num_streets": None,
        "input_shape": None,
        "checkpoint_type": None,
        "used_fallback_window": False,
        "historical_window_ready": False,
        "historical_window_labels": [],
        "historical_window_times": [],
        "historical_non_null_counts": {},
        "historical_fetch_group_count": 0,
        "historical_fetch_windows": [],
        "raw_snapshot_rows": 0,
        "prepared_snapshot_size": 0,
        "prepared_snapshot_null_count": 0,
        "normalization_used": False,
        "output_inverse_transformed": False,
        "street_alignment_mode": "checkpoint_street_names",
        "missing_checkpoint_segments_in_snapshot": 0,
        "extra_snapshot_segments_not_in_checkpoint": 0,
        "postprocess_speed_floor": MIN_VALID_SPEED_KMH,
        "postprocess_speed_ceiling": MAX_VALID_SPEED_KMH,
        "error_message": None,
    }


def _normalize_segment_id_value(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    try:
        as_float = float(text)
        if np.isfinite(as_float) and as_float.is_integer():
            return str(int(as_float))
    except Exception:
        pass

    return text


def _normalize_segment_id_series(series: pd.Series) -> pd.Series:
    return series.apply(_normalize_segment_id_value)


def _to_numpy_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _inverse_transform_1d(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values * std) + mean


def _safe_speed_mask(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.isfinite(arr) & (arr >= MIN_VALID_SPEED_KMH) & (arr <= MAX_VALID_SPEED_KMH)


def _clip_speed_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    arr = np.clip(arr, MIN_VALID_SPEED_KMH, MAX_VALID_SPEED_KMH)
    return arr.astype(np.float32)


def _safe_median_speed(*arrays: Any, default: float = DEFAULT_GLOBAL_SPEED_KMH) -> float:
    parts: list[np.ndarray] = []

    for arr in arrays:
        if arr is None:
            continue
        vals = np.asarray(arr, dtype=np.float32).reshape(-1)
        vals = vals[_safe_speed_mask(vals)]
        if vals.size > 0:
            parts.append(vals)

    if not parts:
        return float(default)

    merged = np.concatenate(parts)
    if merged.size == 0:
        return float(default)

    return float(np.median(merged))


def get_snapshot_timestamp(completed_snapshot_df: pd.DataFrame) -> pd.Timestamp | None:
    if completed_snapshot_df.empty or "snapshot_time" not in completed_snapshot_df.columns:
        return None

    valid_times = completed_snapshot_df["snapshot_time"].dropna()
    if valid_times.empty:
        return None

    value = pd.to_datetime(valid_times.iloc[0], errors="coerce")
    return None if pd.isna(value) else value


def floor_to_15_minutes(value: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).floor(f"{PARQUET_BUCKET_MINUTES}min")


def resolve_checkpoint_path() -> Path:
    return Path(__file__).resolve().parent / "cnn_trained model.pt"


def load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint, "checkpoint_dict"

    if isinstance(checkpoint, dict):
        return {
            "model_state_dict": checkpoint,
            "metadata": {},
        }, "raw_state_dict"

    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def infer_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state_dict = checkpoint["model_state_dict"]

    conv1_weight = state_dict.get("backbone.0.weight")
    if conv1_weight is None:
        raise KeyError("Checkpoint is missing 'backbone.0.weight'.")

    hidden = int(conv1_weight.shape[0])
    checkpoint_num_streets = int(conv1_weight.shape[1])

    model_kwargs = dict(checkpoint.get("model_kwargs", {}) or {})

    window_size = int(checkpoint.get("window_size", 8))
    num_streets = int(checkpoint.get("num_streets", checkpoint_num_streets))
    if num_streets != checkpoint_num_streets:
        num_streets = checkpoint_num_streets

    use_normalization = bool(checkpoint.get("use_normalization", False))
    normalize_target = bool(checkpoint.get("normalize_target", False))
    enforce_nonneg = not (use_normalization and normalize_target)

    delta_scale = float(model_kwargs.get("delta_scale", 5.0))
    topk = int(model_kwargs.get("topk", 4))
    hidden = int(model_kwargs.get("hidden", hidden))

    street_names = [str(x).strip() for x in checkpoint.get("street_names", [])]
    street_names = [_normalize_segment_id_value(x) for x in street_names]

    return {
        "window_size": window_size,
        "num_streets": num_streets,
        "hidden": hidden,
        "topk": topk,
        "delta_scale": delta_scale,
        "enforce_nonneg": enforce_nonneg,
        "street_names": street_names,
        "recent_steps": int(checkpoint.get("recent_steps", 4)),
        "use_daily_lag": bool(checkpoint.get("use_daily_lag", True)),
        "use_weekly_lag": bool(checkpoint.get("use_weekly_lag", True)),
        "num_similar_days": int(checkpoint.get("num_similar_days", 0)),
        "padding_value": float(checkpoint.get("padding_value", 0.0)),
        "use_normalization": use_normalization,
        "normalize_input": bool(checkpoint.get("normalize_input", False)),
        "normalize_target": normalize_target,
        "x_scaler_mean": _to_numpy_or_none(checkpoint.get("x_scaler_mean")),
        "x_scaler_std": _to_numpy_or_none(checkpoint.get("x_scaler_std")),
        "y_scaler_mean": _to_numpy_or_none(checkpoint.get("y_scaler_mean")),
        "y_scaler_std": _to_numpy_or_none(checkpoint.get("y_scaler_std")),
    }


def build_model_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    config = infer_model_config(checkpoint=checkpoint)

    model = RefinerCNN(
        window=config["window_size"],
        num_streets=config["num_streets"],
        hidden=config["hidden"],
        delta_scale=config["delta_scale"],
        enforce_nonneg=config["enforce_nonneg"],
        topk=config["topk"],
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return model, config


@st.cache_resource(show_spinner=False)
def load_model_bundle(checkpoint_path_str: str) -> tuple[nn.Module, dict[str, Any], str]:
    checkpoint_path = Path(checkpoint_path_str)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model file was not found at: {checkpoint_path}")

    checkpoint, checkpoint_type = load_checkpoint(checkpoint_path)
    model, config = build_model_from_checkpoint(checkpoint=checkpoint)

    if not config["street_names"]:
        raise ValueError("Checkpoint does not contain 'street_names' metadata.")

    return model, config, checkpoint_type


@st.cache_data(show_spinner=False, ttl=3600)
def build_segment_lookup_from_gpkg(gpkg_path: str) -> pd.DataFrame:
    gdf = gpd.read_file(gpkg_path).copy()

    required_columns = {"id", "start_id", "bus_lines"}
    missing_columns = required_columns - set(gdf.columns)

    if missing_columns:
        raise ValueError(
            f"GPKG is missing required columns for historical mapping: {sorted(missing_columns)}"
        )

    lookup_df = gdf[["id", "start_id", "bus_lines"]].copy()
    lookup_df["segment_id"] = _normalize_segment_id_series(lookup_df["id"])
    lookup_df["pointId"] = pd.to_numeric(lookup_df["start_id"], errors="coerce")
    lookup_df["bus_lines"] = lookup_df["bus_lines"].fillna("").astype(str)

    lookup_df["lineId"] = lookup_df["bus_lines"].str.split(",")
    lookup_df = lookup_df.explode("lineId")
    lookup_df["lineId"] = lookup_df["lineId"].astype(str).str.strip()

    lookup_df = lookup_df[
        lookup_df["pointId"].notna()
        & lookup_df["lineId"].ne("")
        & lookup_df["segment_id"].ne("")
    ].copy()

    lookup_df["pointId"] = lookup_df["pointId"].astype(int)
    lookup_df = lookup_df[["segment_id", "pointId", "lineId"]].drop_duplicates()

    return lookup_df


def auth_request(url: str, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def download_and_concatenate_parquets(url_list: list[str]) -> pa.Table:
    arrow_table = None

    for url in url_list:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = BytesIO(response.content)
        table = pq.read_table(data)

        if arrow_table is None:
            arrow_table = table
        else:
            arrow_table = pa.concat_tables([arrow_table, table])

    if arrow_table is None:
        raise ValueError("No parquet files were returned by the API.")

    return arrow_table


@st.cache_data(show_spinner=False, ttl=300)
def fetch_raw_bucket_speeds(
    token: str,
    window_start_iso: str,
    window_end_iso: str,
) -> pd.DataFrame:
    window_start = pd.Timestamp(window_start_iso)
    window_end = pd.Timestamp(window_end_iso)

    start_ts = float(window_start.timestamp())
    end_ts = float(window_end.timestamp())

    response = auth_request(
        (
            "https://api.mobilitytwin.brussels/parquetized"
            f"?start_timestamp={start_ts}"
            f"&end_timestamp={end_ts}"
            f"&component={PARQUET_COMPONENT}"
        ),
        token=token,
    )

    parquet_urls = response.get("results", [])
    if not parquet_urls:
        return pd.DataFrame(columns=["local_time", "lineId", "pointId", "speed"])

    arrow_table = download_and_concatenate_parquets(parquet_urls)

    con = duckdb.connect()
    try:
        con.register("combined_data", arrow_table)

        df = con.execute(
            f"""
            WITH entries AS (
                SELECT
                    CAST(lineId AS VARCHAR) AS lineId,
                    CAST(pointId AS BIGINT) AS pointId,
                    directionId,
                    distanceFromPoint,
                    (date AT TIME ZONE 'UTC' AT TIME ZONE '{BRUSSELS_TIMEZONE}')::timestamp AS local_date
                FROM combined_data
            ),
            filtered AS (
                SELECT
                    *,
                    COUNT(*) OVER (
                        PARTITION BY directionId, pointId, local_date, lineId
                    ) AS row_count
                FROM entries
            ),
            delta_table AS (
                SELECT
                    local_date,
                    lineId,
                    directionId,
                    pointId,
                    distanceFromPoint,
                    distanceFromPoint
                        - LAG(distanceFromPoint) OVER (
                            PARTITION BY pointId, directionId, lineId
                            ORDER BY local_date
                        ) AS distance_delta,
                    local_date
                        - LAG(local_date) OVER (
                            PARTITION BY pointId, directionId, lineId
                            ORDER BY local_date
                        ) AS time_delta
                FROM filtered
                WHERE row_count = 1
            ),
            speed_table AS (
                SELECT
                    local_date,
                    lineId,
                    pointId,
                    distance_delta / epoch(time_delta) AS speed
                FROM delta_table
                WHERE epoch(time_delta) < 30
                  AND distance_delta < 600
            )
            SELECT
                time_bucket(INTERVAL '{PARQUET_BUCKET_MINUTES} minutes', local_date) AS local_time,
                lineId,
                pointId,
                AVG(speed) * 3.6 AS speed
            FROM speed_table
            WHERE speed > 0
            GROUP BY local_time, lineId, pointId
            """
        ).df()
    finally:
        con.close()

    if df.empty:
        return pd.DataFrame(columns=["local_time", "lineId", "pointId", "speed"])

    df["local_time"] = pd.to_datetime(df["local_time"], errors="coerce")
    df["lineId"] = df["lineId"].astype(str).str.strip()
    df["pointId"] = pd.to_numeric(df["pointId"], errors="coerce").astype("Int64")
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")

    return df


def group_bucket_times_for_compact_fetch(
    bucket_times: list[pd.Timestamp],
) -> list[tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]]:
    if not bucket_times:
        return []

    sorted_times = sorted(pd.Timestamp(value) for value in bucket_times)

    groups: list[list[pd.Timestamp]] = []
    current_group: list[pd.Timestamp] = [sorted_times[0]]

    for value in sorted_times[1:]:
        previous = current_group[-1]

        if value - previous <= pd.Timedelta(hours=2):
            current_group.append(value)
        else:
            groups.append(current_group)
            current_group = [value]

    groups.append(current_group)

    windows: list[tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]] = []
    for group in groups:
        start = min(group) - pd.Timedelta(minutes=PARQUET_MARGIN_MINUTES)
        end = max(group) + pd.Timedelta(minutes=PARQUET_BUCKET_MINUTES)
        windows.append((start, end, group))

    return windows


@st.cache_data(show_spinner=False, ttl=300)
def fetch_segment_snapshots_for_multiple_buckets(
    token: str,
    gpkg_path: str,
    bucket_time_isos: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bucket_times = [pd.Timestamp(value) for value in bucket_time_isos]

    if not bucket_times:
        return pd.DataFrame(), {
            "historical_fetch_group_count": 0,
            "historical_fetch_windows": [],
        }

    lookup_df = build_segment_lookup_from_gpkg(gpkg_path)
    grouped_windows = group_bucket_times_for_compact_fetch(bucket_times)

    merged_frames: list[pd.DataFrame] = []
    fetch_windows = [
        {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "bucket_count": len(bucket_group),
        }
        for window_start, window_end, bucket_group in grouped_windows
    ]

    for window_start, window_end, bucket_group in grouped_windows:
        raw_speed_df = fetch_raw_bucket_speeds(
            token=token,
            window_start_iso=window_start.isoformat(),
            window_end_iso=window_end.isoformat(),
        )

        if raw_speed_df.empty:
            continue

        filtered_df = raw_speed_df.loc[
            raw_speed_df["local_time"].isin(bucket_group)
        ].copy()

        if filtered_df.empty:
            continue

        joined_df = pd.merge(
            filtered_df,
            lookup_df,
            on=["pointId", "lineId"],
            how="inner",
        )

        if joined_df.empty:
            continue

        merged_frames.append(joined_df)

    if not merged_frames:
        return pd.DataFrame(index=pd.Index([], name="segment_id")), {
            "historical_fetch_group_count": len(grouped_windows),
            "historical_fetch_windows": fetch_windows,
        }

    merged_df = pd.concat(merged_frames, ignore_index=True)

    segment_df = (
        merged_df.groupby(["segment_id", "local_time"], as_index=False)["speed"]
        .mean()
        .rename(columns={"speed": "historical_speed_kmh"})
    )

    pivot_df = segment_df.pivot(
        index="segment_id",
        columns="local_time",
        values="historical_speed_kmh",
    ).sort_index()

    pivot_df.index = pivot_df.index.astype(str)

    return pivot_df, {
        "historical_fetch_group_count": len(grouped_windows),
        "historical_fetch_windows": fetch_windows,
    }


def build_historical_window_plan_from_checkpoint(
    snapshot_time: pd.Timestamp,
    recent_steps: int,
    use_daily_lag: bool,
    use_weekly_lag: bool,
) -> list[tuple[str, pd.Timestamp | None]]:
    bucket_time = floor_to_15_minutes(snapshot_time)
    plan: list[tuple[str, pd.Timestamp | None]] = []

    day_start = bucket_time.normalize()

    for step_back in range(recent_steps, 0, -1):
        candidate = bucket_time - pd.Timedelta(minutes=PARQUET_BUCKET_MINUTES * step_back)
        label = f"recent_t_minus_{PARQUET_BUCKET_MINUTES * step_back}m"
        plan.append((label, candidate if candidate >= day_start else None))

    if use_daily_lag:
        plan.append(("daily_t_minus_1d", bucket_time - pd.Timedelta(days=1)))

    if use_weekly_lag:
        plan.append(("weekly_t_minus_1w", bucket_time - pd.Timedelta(weeks=1)))
        plan.append(("weekly_t_minus_2w", bucket_time - pd.Timedelta(weeks=2)))
        plan.append(("weekly_t_minus_3w", bucket_time - pd.Timedelta(weeks=3)))

    return plan


def prepare_snapshot_series_aligned_to_checkpoint(
    completed_snapshot_df: pd.DataFrame,
    checkpoint_street_names: list[str],
) -> tuple[pd.Series, dict[str, int]]:
    required_columns = {"segment_id", "final_speed_kmh"}
    missing_columns = required_columns - set(completed_snapshot_df.columns)

    if missing_columns:
        raise ValueError(
            f"Completed snapshot is missing required columns: {sorted(missing_columns)}"
        )

    working_df = completed_snapshot_df.copy()
    working_df["segment_id"] = _normalize_segment_id_series(working_df["segment_id"])
    working_df["final_speed_kmh"] = pd.to_numeric(
        working_df["final_speed_kmh"],
        errors="coerce",
    )

    snapshot_series = (
        working_df.drop_duplicates(subset=["segment_id"], keep="last")
        .set_index("segment_id")["final_speed_kmh"]
        .astype("float32")
    )

    checkpoint_index = pd.Index(
        [_normalize_segment_id_value(x) for x in checkpoint_street_names],
        name="segment_id",
    )
    aligned = snapshot_series.reindex(checkpoint_index)

    meta = {
        "missing_checkpoint_segments_in_snapshot": int(aligned.isna().sum()),
        "extra_snapshot_segments_not_in_checkpoint": int(
            (~snapshot_series.index.isin(checkpoint_index)).sum()
        ),
    }

    return aligned, meta


def build_historical_feature_matrix(
    token: str,
    gpkg_path: str,
    snapshot_time: pd.Timestamp,
    ordered_segment_ids: list[str],
    recent_steps: int,
    use_daily_lag: bool,
    use_weekly_lag: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    plan = build_historical_window_plan_from_checkpoint(
        snapshot_time=snapshot_time,
        recent_steps=recent_steps,
        use_daily_lag=use_daily_lag,
        use_weekly_lag=use_weekly_lag,
    )

    real_bucket_times = [bucket_time for _, bucket_time in plan if bucket_time is not None]

    bucket_matrix, fetch_meta = fetch_segment_snapshots_for_multiple_buckets(
        token=token,
        gpkg_path=gpkg_path,
        bucket_time_isos=tuple(bucket.isoformat() for bucket in real_bucket_times),
    )

    historical_df = pd.DataFrame(index=ordered_segment_ids)
    historical_df.index.name = "segment_id"

    non_null_counts: dict[str, int] = {}

    for label, bucket_time in plan:
        if bucket_time is not None and bucket_time in bucket_matrix.columns:
            aligned_series = bucket_matrix[bucket_time].reindex(ordered_segment_ids)
        else:
            aligned_series = pd.Series(index=ordered_segment_ids, dtype="float32")

        historical_df[label] = pd.to_numeric(aligned_series, errors="coerce").astype("float32")
        non_null_counts[label] = int(historical_df[label].notna().sum())

    diagnostics = {
        "historical_window_ready": True,
        "historical_window_labels": [label for label, _ in plan],
        "historical_window_times": [
            bucket.isoformat() if bucket is not None else None for _, bucket in plan
        ],
        "historical_non_null_counts": non_null_counts,
        **fetch_meta,
    }

    return historical_df, diagnostics


def fill_missing_historical_values(
    historical_df: pd.DataFrame,
    fallback_series: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    result = historical_df.copy()

    checkpoint_mean = config.get("x_scaler_mean")
    if checkpoint_mean is not None:
        checkpoint_mean_series = pd.Series(
            checkpoint_mean,
            index=result.index,
            dtype="float32",
        )
    else:
        checkpoint_mean_series = pd.Series(index=result.index, dtype="float32")

    global_value = _safe_median_speed(
        fallback_series,
        checkpoint_mean_series,
        default=DEFAULT_GLOBAL_SPEED_KMH,
    )

    for column in result.columns:
        col = pd.to_numeric(result[column], errors="coerce")
        col = col.fillna(fallback_series)
        col = col.fillna(checkpoint_mean_series)
        col = col.fillna(global_value)
        col = col.clip(lower=MIN_VALID_SPEED_KMH, upper=MAX_VALID_SPEED_KMH)
        result[column] = col.astype("float32")

    return result


def normalize_historical_input_if_needed(
    historical_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, bool]:
    use_norm = bool(config.get("use_normalization", False)) and bool(config.get("normalize_input", False))
    mean = config.get("x_scaler_mean")
    std = config.get("x_scaler_std")

    if not use_norm or mean is None or std is None:
        return historical_df, False

    values = historical_df.to_numpy(dtype=np.float32)  # [S, W]
    values = (values - mean[:, None]) / std[:, None]

    out = pd.DataFrame(
        values,
        index=historical_df.index,
        columns=historical_df.columns,
    ).astype("float32")

    return out, True


def inverse_transform_predictions_if_needed(
    y_hat: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    use_norm = bool(config.get("use_normalization", False)) and bool(config.get("normalize_target", False))
    mean = config.get("y_scaler_mean")
    std = config.get("y_scaler_std")

    if not use_norm or mean is None or std is None:
        return y_hat, False

    restored = _inverse_transform_1d(y_hat, mean=mean, std=std)
    return restored, True


def build_model_input_window_from_historical(historical_df: pd.DataFrame) -> torch.Tensor:
    values = historical_df.to_numpy(dtype="float32")  # [S, W]
    values = values.T  # [W, S]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)  # [1, W, S]


def _build_snapshot_fallback_series_for_checkpoint(
    completed_snapshot_df: pd.DataFrame,
    checkpoint_segment_ids: list[str],
) -> pd.Series:
    working_df = completed_snapshot_df.copy()
    if "segment_id" not in working_df.columns:
        return pd.Series(index=checkpoint_segment_ids, dtype="float32")

    working_df["segment_id_norm"] = _normalize_segment_id_series(working_df["segment_id"])

    candidate_cols = [
        "final_speed_kmh",
        "live_speed_kmh",
        "interpolated_speed_kmh",
    ]

    for col in candidate_cols:
        if col not in working_df.columns:
            working_df[col] = np.nan
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    tmp = working_df.drop_duplicates(subset=["segment_id_norm"], keep="last").copy()
    tmp["fallback_speed"] = tmp["final_speed_kmh"]
    tmp["fallback_speed"] = tmp["fallback_speed"].fillna(tmp["live_speed_kmh"])
    tmp["fallback_speed"] = tmp["fallback_speed"].fillna(tmp["interpolated_speed_kmh"])

    fallback_series = tmp.set_index("segment_id_norm")["fallback_speed"]
    fallback_series = fallback_series.reindex(checkpoint_segment_ids).astype("float32")

    return fallback_series


@st.cache_data(show_spinner=False, ttl=90)
def run_tmp_model_inference(
    completed_snapshot_df: pd.DataFrame,
    token: str,
    gpkg_path: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    diagnostics = build_empty_estimation_diagnostics()

    if completed_snapshot_df.empty:
        diagnostics["error_message"] = "Completed STIB snapshot is empty."
        return pd.DataFrame(columns=["segment_id", "est_speed"]), diagnostics

    snapshot_time = get_snapshot_timestamp(completed_snapshot_df)
    diagnostics["snapshot_found"] = snapshot_time is not None
    diagnostics["snapshot_time"] = snapshot_time.isoformat() if snapshot_time is not None else None

    if snapshot_time is None:
        diagnostics["error_message"] = "Snapshot time could not be extracted."
        return pd.DataFrame(columns=["segment_id", "est_speed"]), diagnostics

    diagnostics["snapshot_bucket_time"] = floor_to_15_minutes(snapshot_time).isoformat()
    diagnostics["raw_snapshot_rows"] = int(len(completed_snapshot_df))

    checkpoint_path = resolve_checkpoint_path()
    diagnostics["model_path"] = str(checkpoint_path)

    try:
        model, config, checkpoint_type = load_model_bundle(str(checkpoint_path))
    except Exception as exc:
        diagnostics["error_message"] = f"Model load failed: {exc}"
        return pd.DataFrame(columns=["segment_id", "est_speed"]), diagnostics

    diagnostics["checkpoint_type"] = checkpoint_type
    diagnostics["model_loaded"] = True
    diagnostics["window_size"] = int(config["window_size"])
    diagnostics["num_streets"] = int(config["num_streets"])

    checkpoint_segment_ids = [_normalize_segment_id_value(x) for x in config["street_names"]]

    base_snapshot_series, align_meta = prepare_snapshot_series_aligned_to_checkpoint(
        completed_snapshot_df=completed_snapshot_df,
        checkpoint_street_names=checkpoint_segment_ids,
    )
    diagnostics.update(align_meta)
    diagnostics["prepared_snapshot_size"] = int(len(base_snapshot_series))
    diagnostics["prepared_snapshot_null_count"] = int(base_snapshot_series.isna().sum())

    snapshot_fallback_series = _build_snapshot_fallback_series_for_checkpoint(
        completed_snapshot_df=completed_snapshot_df,
        checkpoint_segment_ids=checkpoint_segment_ids,
    )

    global_default = _safe_median_speed(
        snapshot_fallback_series,
        base_snapshot_series,
        config.get("y_scaler_mean"),
        config.get("x_scaler_mean"),
        default=DEFAULT_GLOBAL_SPEED_KMH,
    )

    base_snapshot_series = base_snapshot_series.fillna(snapshot_fallback_series)
    base_snapshot_series = base_snapshot_series.fillna(global_default)
    base_snapshot_series = base_snapshot_series.clip(
        lower=MIN_VALID_SPEED_KMH,
        upper=MAX_VALID_SPEED_KMH,
    ).astype("float32")

    if base_snapshot_series.empty:
        diagnostics["error_message"] = "No aligned snapshot values were found."
        return pd.DataFrame(columns=["segment_id", "est_speed"]), diagnostics

    ordered_segment_ids = list(base_snapshot_series.index.astype(str))

    if len(ordered_segment_ids) != int(config["num_streets"]):
        raise ValueError(
            "Current aligned snapshot size does not match checkpoint street count. "
            f"current={len(ordered_segment_ids)} checkpoint={config['num_streets']}"
        )

    try:
        historical_df, historical_meta = build_historical_feature_matrix(
            token=token,
            gpkg_path=gpkg_path,
            snapshot_time=snapshot_time,
            ordered_segment_ids=ordered_segment_ids,
            recent_steps=int(config["recent_steps"]),
            use_daily_lag=bool(config["use_daily_lag"]),
            use_weekly_lag=bool(config["use_weekly_lag"]),
        )
        diagnostics.update(historical_meta)

        historical_df = fill_missing_historical_values(
            historical_df=historical_df,
            fallback_series=base_snapshot_series,
            config=config,
        )
    except Exception as exc:
        diagnostics["used_fallback_window"] = True
        diagnostics["error_message"] = f"Historical feature extraction failed: {exc}"

        plan = build_historical_window_plan_from_checkpoint(
            snapshot_time=snapshot_time,
            recent_steps=int(config["recent_steps"]),
            use_daily_lag=bool(config["use_daily_lag"]),
            use_weekly_lag=bool(config["use_weekly_lag"]),
        )

        fallback_df = pd.DataFrame(index=ordered_segment_ids)
        safe_snapshot = base_snapshot_series.fillna(global_default).clip(
            lower=MIN_VALID_SPEED_KMH,
            upper=MAX_VALID_SPEED_KMH,
        )

        for label, _ in plan:
            fallback_df[label] = safe_snapshot

        historical_df = fallback_df.astype("float32")
        diagnostics["historical_window_ready"] = False
        diagnostics["historical_window_labels"] = [label for label, _ in plan]
        diagnostics["historical_window_times"] = [
            bucket.isoformat() if bucket is not None else None for _, bucket in plan
        ]
        diagnostics["historical_non_null_counts"] = {
            label: int(np.sum(np.isfinite(safe_snapshot.to_numpy(dtype=np.float32))))
            for label, _ in plan
        }

    if historical_df.shape[1] != int(config["window_size"]):
        raise ValueError(
            "Historical feature window size does not match checkpoint window size. "
            f"historical={historical_df.shape[1]} checkpoint={config['window_size']}"
        )

    historical_df, normalization_used = normalize_historical_input_if_needed(
        historical_df=historical_df,
        config=config,
    )
    diagnostics["normalization_used"] = normalization_used

    x = build_model_input_window_from_historical(historical_df)
    diagnostics["input_shape"] = tuple(int(v) for v in x.shape)

    with torch.inference_mode():
        y_hat = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)

    y_hat, inverse_used = inverse_transform_predictions_if_needed(
        y_hat=y_hat,
        config=config,
    )
    diagnostics["output_inverse_transformed"] = inverse_used

    prediction_df = pd.DataFrame(
        {
            "segment_id": ordered_segment_ids,
            "est_speed": pd.to_numeric(y_hat, errors="coerce"),
        }
    )

    street_fallback = base_snapshot_series.reindex(prediction_df["segment_id"]).to_numpy(dtype=np.float32)

    y_mean = config.get("y_scaler_mean")
    x_mean = config.get("x_scaler_mean")

    if y_mean is not None:
        second_fallback = np.asarray(y_mean, dtype=np.float32)
    elif x_mean is not None:
        second_fallback = np.asarray(x_mean, dtype=np.float32)
    else:
        second_fallback = np.full(len(prediction_df), global_default, dtype=np.float32)

    second_fallback = np.asarray(second_fallback, dtype=np.float32)
    if second_fallback.shape[0] != len(prediction_df):
        second_fallback = np.full(len(prediction_df), global_default, dtype=np.float32)

    values = np.array(prediction_df["est_speed"], dtype=np.float32, copy=True)

    invalid_mask = ~_safe_speed_mask(values)

    valid_street_mask = _safe_speed_mask(street_fallback)
    use_street_mask = invalid_mask & valid_street_mask
    values[use_street_mask] = street_fallback[use_street_mask]

    invalid_mask = ~_safe_speed_mask(values)
    valid_second_mask = _safe_speed_mask(second_fallback)
    use_second_mask = invalid_mask & valid_second_mask
    values[use_second_mask] = second_fallback[use_second_mask]

    invalid_mask = ~_safe_speed_mask(values)
    values[invalid_mask] = global_default

    values = np.clip(values, MIN_VALID_SPEED_KMH, MAX_VALID_SPEED_KMH).astype(np.float32)
    prediction_df["est_speed"] = values

    return prediction_df, diagnostics


def attach_prediction_df_to_gdf(
    gdf: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    result = gdf.copy()

    if "id" not in result.columns:
        result["est_speed"] = pd.NA
        return result, 0

    if prediction_df.empty:
        result["est_speed"] = pd.NA
        return result, 0

    pred_df = prediction_df.copy()
    pred_df["segment_id_norm"] = _normalize_segment_id_series(pred_df["segment_id"])

    lookup = dict(
        zip(
            pred_df["segment_id_norm"],
            pd.to_numeric(pred_df["est_speed"], errors="coerce"),
        )
    )

    result["segment_id_str"] = _normalize_segment_id_series(result["id"])
    result["est_speed"] = result["segment_id_str"].map(lookup)

    global_default = _safe_median_speed(
        pred_df["est_speed"],
        default=DEFAULT_GLOBAL_SPEED_KMH,
    )

    if "bus_speed" in result.columns:
        bus_speed_numeric = pd.to_numeric(result["bus_speed"], errors="coerce")
        use_bus_mask = result["est_speed"].isna() & _safe_speed_mask(bus_speed_numeric)
        result.loc[use_bus_mask, "est_speed"] = bus_speed_numeric.loc[use_bus_mask]

    est_num = pd.to_numeric(result["est_speed"], errors="coerce")
    invalid_mask = est_num.isna() | (est_num < MIN_VALID_SPEED_KMH) | (est_num > MAX_VALID_SPEED_KMH)
    result.loc[invalid_mask, "est_speed"] = global_default

    result["est_speed"] = pd.to_numeric(result["est_speed"], errors="coerce").clip(
        lower=MIN_VALID_SPEED_KMH,
        upper=MAX_VALID_SPEED_KMH,
    )

    matched_segments = int(result["est_speed"].notna().sum())

    return result, matched_segments


def attach_estimated_speed_to_snapshot(
    completed_snapshot_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    if completed_snapshot_df.empty:
        result = completed_snapshot_df.copy()
        result["estimated_speed"] = pd.NA
        return result

    if "segment_id" not in completed_snapshot_df.columns:
        raise ValueError("completed_snapshot_df must contain 'segment_id'")

    result = completed_snapshot_df.copy()
    result["segment_id_norm"] = _normalize_segment_id_series(result["segment_id"])

    pred_df = prediction_df.copy()
    pred_df["segment_id_norm"] = _normalize_segment_id_series(pred_df["segment_id"])

    merged = result.merge(
        pred_df[["segment_id_norm", "est_speed"]].rename(columns={"est_speed": "estimated_speed"}),
        on="segment_id_norm",
        how="left",
    )

    candidate_final = pd.to_numeric(merged.get("final_speed_kmh", np.nan), errors="coerce")
    candidate_live = pd.to_numeric(merged.get("live_speed_kmh", np.nan), errors="coerce")
    candidate_interp = pd.to_numeric(merged.get("interpolated_speed_kmh", np.nan), errors="coerce")
    est = pd.to_numeric(merged["estimated_speed"], errors="coerce")

    global_default = _safe_median_speed(
        est,
        candidate_final,
        candidate_live,
        candidate_interp,
        default=DEFAULT_GLOBAL_SPEED_KMH,
    )

    est = est.fillna(candidate_final)
    est = est.fillna(candidate_live)
    est = est.fillna(candidate_interp)
    est = est.fillna(global_default)
    est = est.clip(lower=MIN_VALID_SPEED_KMH, upper=MAX_VALID_SPEED_KMH)

    merged["estimated_speed"] = est
    merged = merged.drop(columns=["segment_id_norm"])

    return merged


def build_estimation_artifacts(
    completed_snapshot_df: pd.DataFrame,
    token: str,
    gpkg_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prediction_df, diagnostics = run_tmp_model_inference(
        completed_snapshot_df=completed_snapshot_df,
        token=token,
        gpkg_path=gpkg_path,
    )

    enriched_df = attach_estimated_speed_to_snapshot(
        completed_snapshot_df=completed_snapshot_df,
        prediction_df=prediction_df,
    )

    return prediction_df, enriched_df, diagnostics


def attach_tmp_estimated_speeds(
    gdf: pd.DataFrame,
    completed_snapshot_df: pd.DataFrame,
    token: str,
    gpkg_path: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = gdf.copy()
    diagnostics = build_empty_estimation_diagnostics()
    diagnostics["map_has_id_column"] = "id" in result.columns

    if "id" not in result.columns:
        diagnostics["error_message"] = "The Brussels map file does not contain an 'id' column."
        result["est_speed"] = pd.NA
        return result, diagnostics

    try:
        prediction_df, _, diagnostics = build_estimation_artifacts(
            completed_snapshot_df=completed_snapshot_df,
            token=token,
            gpkg_path=gpkg_path,
        )
    except Exception as exc:
        diagnostics["error_message"] = f"PT inference failed: {exc}"
        result["est_speed"] = pd.NA
        return result, diagnostics

    result, matched_segments = attach_prediction_df_to_gdf(
        gdf=result,
        prediction_df=prediction_df,
    )
    diagnostics["matched_segments"] = matched_segments

    return result, diagnostics


@st.cache_data(show_spinner=False, ttl=90)
def enrich_snapshot_with_estimation(
    completed_snapshot_df: pd.DataFrame,
    token: str,
    gpkg_path: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, enriched_df, diagnostics = build_estimation_artifacts(
        completed_snapshot_df=completed_snapshot_df,
        token=token,
        gpkg_path=gpkg_path,
    )

    return enriched_df, diagnostics
