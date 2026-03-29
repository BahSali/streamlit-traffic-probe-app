from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import streamlit as st
import torch
import torch.nn as nn


BRUSSELS_TIMEZONE = "Europe/Brussels"
PARQUET_COMPONENT = "stib_vehicle_distance_parquetize"
PARQUET_MARGIN_MINUTES = 30
PARQUET_BUCKET_MINUTES = 15
REQUEST_TIMEOUT_SECONDS = 45


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
            return torch.nn.functional.softplus(y_hat)

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
        "error_message": None,
    }


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


def build_historical_window_plan(snapshot_time: pd.Timestamp) -> list[tuple[str, pd.Timestamp]]:
    """
    Build an 8-step window matching the training window size.

    Order:
        1. t-60m
        2. t-45m
        3. t-30m
        4. t-15m
        5. t-1d
        6. t-1w
        7. t-2w
        8. t-3w
    """
    bucket_time = floor_to_15_minutes(snapshot_time)

    return [
        ("recent_t_minus_60m", bucket_time - pd.Timedelta(minutes=60)),
        ("recent_t_minus_45m", bucket_time - pd.Timedelta(minutes=45)),
        ("recent_t_minus_30m", bucket_time - pd.Timedelta(minutes=30)),
        ("recent_t_minus_15m", bucket_time - pd.Timedelta(minutes=15)),
        ("daily_t_minus_1d", bucket_time - pd.Timedelta(days=1)),
        ("weekly_t_minus_1w", bucket_time - pd.Timedelta(weeks=1)),
        ("similar_t_minus_2w", bucket_time - pd.Timedelta(weeks=2)),
        ("similar_t_minus_3w", bucket_time - pd.Timedelta(weeks=3)),
    ]


def prepare_snapshot_series(completed_snapshot_df: pd.DataFrame) -> pd.Series:
    required_columns = {"segment_id", "final_speed_kmh"}
    missing_columns = required_columns - set(completed_snapshot_df.columns)
    if missing_columns:
        raise ValueError(
            f"Completed snapshot is missing required columns: {sorted(missing_columns)}"
        )

    working_df = completed_snapshot_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str).str.strip()
    working_df["final_speed_kmh"] = pd.to_numeric(
        working_df["final_speed_kmh"],
        errors="coerce",
    )

    series = (
        working_df
        .drop_duplicates(subset=["segment_id"], keep="last")
        .set_index("segment_id")["final_speed_kmh"]
        .sort_index()
        .fillna(0.0)
        .astype("float32")
    )

    return series


def resolve_checkpoint_path() -> Path:
    return Path(__file__).resolve().parent / "cnn_trained model.pt"


def load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint, "checkpoint_dict"

    if isinstance(checkpoint, dict):
        return {
            "model_state_dict": checkpoint,
            "metadata": {},
        }, "raw_state_dict"

    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def infer_model_config(checkpoint: dict[str, Any], inferred_num_streets: int) -> dict[str, Any]:
    metadata = checkpoint.get("metadata", {}) or {}
    state_dict = checkpoint["model_state_dict"]

    conv1_weight = state_dict.get("backbone.0.weight")
    if conv1_weight is None:
        raise KeyError("Checkpoint is missing 'backbone.0.weight'.")

    hidden = int(conv1_weight.shape[0])
    checkpoint_num_streets = int(conv1_weight.shape[1])

    window_size = int(metadata.get("window_size", 8))
    num_streets = int(metadata.get("num_streets", checkpoint_num_streets))

    if num_streets != checkpoint_num_streets:
        num_streets = checkpoint_num_streets

    gate_bias_tensor = state_dict.get("gate_bias", None)
    enforce_nonneg = gate_bias_tensor is None

    return {
        "window_size": window_size,
        "num_streets": num_streets,
        "hidden": hidden,
        "topk": 4,
        "delta_scale": 5.0,
        "enforce_nonneg": enforce_nonneg,
    }


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    inferred_num_streets: int,
) -> tuple[nn.Module, dict[str, Any]]:
    config = infer_model_config(checkpoint, inferred_num_streets=inferred_num_streets)

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
    lookup_df["segment_id"] = lookup_df["id"].astype(str).str.strip()
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


def fetch_raw_bucket_speeds(
    token: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    start_ts = float(pd.Timestamp(window_start).timestamp())
    end_ts = float(pd.Timestamp(window_end).timestamp())

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
    """
    Group target buckets into compact fetch windows instead of one large multi-week range.
    """
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


def fetch_segment_snapshots_for_multiple_buckets(
    token: str,
    gpkg_path: str,
    bucket_times: list[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch several compact windows instead of one large multi-week window.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame:
            Index: segment_id
            Columns: bucket_time (Timestamp)
            Values: historical speed in km/h
        Diagnostics:
            metadata about grouped fetch windows
    """
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
            window_start=window_start,
            window_end=window_end,
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


def build_historical_feature_matrix(
    token: str,
    gpkg_path: str,
    snapshot_time: pd.Timestamp,
    ordered_segment_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    plan = build_historical_window_plan(snapshot_time)
    bucket_times = [bucket_time for _, bucket_time in plan]

    bucket_matrix, fetch_meta = fetch_segment_snapshots_for_multiple_buckets(
        token=token,
        gpkg_path=gpkg_path,
        bucket_times=bucket_times,
    )

    historical_df = pd.DataFrame(index=ordered_segment_ids)
    historical_df.index.name = "segment_id"

    non_null_counts: dict[str, int] = {}

    for label, bucket_time in plan:
        if bucket_time in bucket_matrix.columns:
            aligned_series = bucket_matrix[bucket_time].reindex(ordered_segment_ids)
        else:
            aligned_series = pd.Series(index=ordered_segment_ids, dtype="float32")

        historical_df[label] = pd.to_numeric(
            aligned_series,
            errors="coerce",
        ).astype("float32")
        non_null_counts[label] = int(historical_df[label].notna().sum())

    diagnostics = {
        "historical_window_ready": True,
        "historical_window_labels": [label for label, _ in plan],
        "historical_window_times": [bucket.isoformat() for _, bucket in plan],
        "historical_non_null_counts": non_null_counts,
        **fetch_meta,
    }

    return historical_df, diagnostics


def fill_missing_historical_values(
    historical_df: pd.DataFrame,
    fallback_series: pd.Series,
) -> pd.DataFrame:
    """
    Fill missing historical bucket values with the current completed snapshot speed.
    This keeps the app runnable even when some API windows are sparse.
    """
    result = historical_df.copy()

    for column in result.columns:
        result[column] = result[column].fillna(fallback_series)

    result = result.fillna(0.0).astype("float32")
    return result


def build_model_input_window_from_historical(historical_df: pd.DataFrame) -> torch.Tensor:
    """
    Convert aligned historical features to model input [1, W, S].

    Input DataFrame shape:
        [S, W]

    Output tensor shape:
        [1, W, S]
    """
    ordered_columns = historical_df.columns.tolist()
    values = historical_df[ordered_columns].to_numpy(dtype="float32")  # [S, W]
    values = values.T  # [W, S]
    tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0)  # [1, W, S]
    return tensor


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
    diagnostics["snapshot_time"] = (
        snapshot_time.isoformat() if snapshot_time is not None else None
    )

    if snapshot_time is None:
        diagnostics["error_message"] = "Snapshot time could not be extracted."
        return pd.DataFrame(columns=["segment_id", "est_speed"]), diagnostics

    diagnostics["snapshot_bucket_time"] = floor_to_15_minutes(snapshot_time).isoformat()
    diagnostics["raw_snapshot_rows"] = int(len(completed_snapshot_df))

    base_snapshot_series = prepare_snapshot_series(completed_snapshot_df)
    diagnostics["prepared_snapshot_size"] = int(len(base_snapshot_series))
    diagnostics["prepared_snapshot_null_count"] = int(base_snapshot_series.isna().sum())

    if base_snapshot_series.empty:
        diagnostics["error_message"] = "No valid final_speed_kmh values were found."
        return pd.DataFrame(columns=["segment_id", "est_speed"]), diagnostics

    checkpoint_path = resolve_checkpoint_path()
    diagnostics["model_path"] = str(checkpoint_path)

    if not checkpoint_path.exists():
        diagnostics["error_message"] = f"Model file was not found at: {checkpoint_path}"
        return pd.DataFrame(columns=["segment_id", "est_speed"]), diagnostics

    checkpoint, checkpoint_type = load_checkpoint(checkpoint_path)
    diagnostics["checkpoint_type"] = checkpoint_type

    model, config = build_model_from_checkpoint(
        checkpoint=checkpoint,
        inferred_num_streets=len(base_snapshot_series),
    )

    diagnostics["model_loaded"] = True
    diagnostics["window_size"] = int(config["window_size"])
    diagnostics["num_streets"] = int(config["num_streets"])

    ordered_segment_ids = list(base_snapshot_series.index.astype(str))

    if len(ordered_segment_ids) != int(config["num_streets"]):
        raise ValueError(
            "Current snapshot size does not match checkpoint street count. "
            f"current={len(ordered_segment_ids)} checkpoint={config['num_streets']}"
        )

    try:
        historical_df, historical_meta = build_historical_feature_matrix(
            token=token,
            gpkg_path=gpkg_path,
            snapshot_time=snapshot_time,
            ordered_segment_ids=ordered_segment_ids,
        )
        diagnostics.update(historical_meta)

        historical_df = fill_missing_historical_values(
            historical_df=historical_df,
            fallback_series=base_snapshot_series,
        )
    except Exception as exc:
        diagnostics["used_fallback_window"] = True
        diagnostics["error_message"] = f"Historical feature extraction failed: {exc}"

        plan = build_historical_window_plan(snapshot_time)
        fallback_df = pd.DataFrame(index=ordered_segment_ids)

        for label, _ in plan:
            fallback_df[label] = base_snapshot_series

        historical_df = fallback_df.astype("float32")
        diagnostics["historical_window_ready"] = False
        diagnostics["historical_window_labels"] = [label for label, _ in plan]
        diagnostics["historical_window_times"] = [bucket.isoformat() for _, bucket in plan]
        diagnostics["historical_non_null_counts"] = {
            label: int(len(base_snapshot_series)) for label, _ in plan
        }

    if historical_df.shape[1] != int(config["window_size"]):
        raise ValueError(
            "Historical feature window size does not match checkpoint window size. "
            f"historical={historical_df.shape[1]} checkpoint={config['window_size']}"
        )

    x = build_model_input_window_from_historical(historical_df)
    diagnostics["input_shape"] = tuple(int(v) for v in x.shape)

    with torch.no_grad():
        y_hat = model(x).squeeze(0).detach().cpu().numpy()

    prediction_df = pd.DataFrame(
        {
            "segment_id": ordered_segment_ids,
            "est_speed": y_hat,
        }
    )

    prediction_df["est_speed"] = pd.to_numeric(prediction_df["est_speed"], errors="coerce")
    prediction_df["est_speed"] = prediction_df["est_speed"].clip(lower=0)

    return prediction_df, diagnostics


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
        prediction_df, diagnostics = run_tmp_model_inference(
            completed_snapshot_df=completed_snapshot_df,
            token=token,
            gpkg_path=gpkg_path,
        )
    except Exception as exc:
        diagnostics["error_message"] = f"PT inference failed: {exc}"
        result["est_speed"] = pd.NA
        return result, diagnostics

    if prediction_df.empty:
        result["est_speed"] = pd.NA
        return result, diagnostics

    lookup = dict(
        zip(
            prediction_df["segment_id"].astype(str).str.strip(),
            prediction_df["est_speed"],
        )
    )

    result["segment_id_str"] = result["id"].astype(str).str.strip()
    result["est_speed"] = result["segment_id_str"].map(lookup)
    diagnostics["matched_segments"] = int(result["est_speed"].notna().sum())

    return result, diagnostics

def attach_estimated_speed_to_snapshot(
    completed_snapshot_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge model predictions into the completed snapshot DataFrame.

    Adds:
        estimated_speed column
    """
    if completed_snapshot_df.empty:
        result = completed_snapshot_df.copy()
        result["estimated_speed"] = pd.NA
        return result

    if "segment_id" not in completed_snapshot_df.columns:
        raise ValueError("completed_snapshot_df must contain 'segment_id'")

    result = completed_snapshot_df.copy()

    prediction_df = prediction_df.copy()
    prediction_df["segment_id"] = prediction_df["segment_id"].astype(str).str.strip()

    result["segment_id"] = result["segment_id"].astype(str).str.strip()

    merged = result.merge(
        prediction_df.rename(columns={"est_speed": "estimated_speed"}),
        on="segment_id",
        how="left",
    )

    return merged

@st.cache_data(show_spinner=False, ttl=90)
def enrich_snapshot_with_estimation(
    completed_snapshot_df: pd.DataFrame,
    token: str,
    gpkg_path: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full pipeline:
        snapshot → model → merged snapshot with estimated_speed
    """
    prediction_df, diagnostics = run_tmp_model_inference(
        completed_snapshot_df=completed_snapshot_df,
        token=token,
        gpkg_path=gpkg_path,
    )

    enriched_df = attach_estimated_speed_to_snapshot(
        completed_snapshot_df=completed_snapshot_df,
        prediction_df=prediction_df,
    )

    return enriched_df, diagnostics
