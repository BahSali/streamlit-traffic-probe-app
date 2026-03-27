from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn


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
        "estimation_mode": "pt_inference_tmp",
        "snapshot_found": False,
        "snapshot_time": None,
        "map_has_id_column": False,
        "matched_segments": 0,
        "model_loaded": False,
        "model_path": None,
        "window_size": None,
        "num_streets": None,
        "input_shape": None,
        "checkpoint_type": None,
        "used_fallback_window": False,
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
    """
    Temporary local convention:
    Put the model file at:
        core/estimation/cnn_trained model.pt
    """
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

    window_size = int(metadata.get("window_size", 8))
    num_streets = int(metadata.get("num_streets", inferred_num_streets))

    gate_bias_tensor = state_dict.get("gate_bias", None)
    enforce_nonneg = gate_bias_tensor is None

    conv1_weight = state_dict.get("backbone.0.weight")
    if conv1_weight is None:
        raise KeyError("Checkpoint is missing 'backbone.0.weight'.")

    hidden = int(conv1_weight.shape[0])

    return {
        "window_size": window_size,
        "num_streets": num_streets,
        "hidden": hidden,
        "topk": 4,
        "delta_scale": 5.0,
        "enforce_nonneg": enforce_nonneg,
    }


def build_model_from_checkpoint(checkpoint: dict[str, Any], inferred_num_streets: int) -> tuple[nn.Module, dict[str, Any]]:
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


def build_temporary_input_window(snapshot_series: pd.Series, window_size: int) -> torch.Tensor:
    """
    Temporary inference-only window builder.

    Until historical Brussels Mobility features are added, we repeat the current
    snapshot across the full temporal window:
        [current, current, ..., current]

    Output shape:
        [1, W, S]
    """
    values = snapshot_series.to_numpy(dtype="float32")
    repeated = [values.copy() for _ in range(window_size)]
    tensor = torch.tensor(repeated, dtype=torch.float32).unsqueeze(0)
    return tensor


def run_tmp_model_inference(
    completed_snapshot_df: pd.DataFrame,
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

    snapshot_series = prepare_snapshot_series(completed_snapshot_df)

    if snapshot_series.empty:
        diagnostics["error_message"] = "No valid final_speed_kmh values were found."
        return pd.DataFrame(columns=["segment_id", "est_speed"]), diagnostics

    checkpoint_path = resolve_checkpoint_path()
    diagnostics["model_path"] = str(checkpoint_path)

    if not checkpoint_path.exists():
        diagnostics["error_message"] = (
            f"Model file was not found at: {checkpoint_path}"
        )
        return pd.DataFrame(columns=["segment_id", "est_speed"]), diagnostics

    checkpoint, checkpoint_type = load_checkpoint(checkpoint_path)
    diagnostics["checkpoint_type"] = checkpoint_type

    model, config = build_model_from_checkpoint(
        checkpoint=checkpoint,
        inferred_num_streets=len(snapshot_series),
    )

    diagnostics["model_loaded"] = True
    diagnostics["window_size"] = int(config["window_size"])
    diagnostics["num_streets"] = int(config["num_streets"])

    if int(config["num_streets"]) != len(snapshot_series):
        raise ValueError(
            "Checkpoint num_streets does not match the current snapshot size. "
            f"checkpoint={config['num_streets']} current={len(snapshot_series)}"
        )

    x = build_temporary_input_window(
        snapshot_series=snapshot_series,
        window_size=int(config["window_size"]),
    )
    diagnostics["input_shape"] = tuple(int(v) for v in x.shape)
    diagnostics["used_fallback_window"] = True

    with torch.no_grad():
        y_hat = model(x).squeeze(0).detach().cpu().numpy()

    prediction_df = pd.DataFrame(
        {
            "segment_id": snapshot_series.index.astype(str),
            "est_speed": y_hat,
        }
    )

    prediction_df["est_speed"] = pd.to_numeric(prediction_df["est_speed"], errors="coerce")
    prediction_df["est_speed"] = prediction_df["est_speed"].clip(lower=0)

    return prediction_df, diagnostics


def attach_tmp_estimated_speeds(
    gdf: pd.DataFrame,
    completed_snapshot_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Temporary map-2 estimator using the trained .pt model.

    Current temporary behavior:
    - loads the checkpoint
    - creates a fallback [1, W, S] tensor by repeating the current snapshot
    - runs inference
    - maps predictions back to GeoDataFrame rows by segment id

    Later:
    - replace the fallback window with real historical feature extraction
    - add saved normalization artifacts
    """
    result = gdf.copy()
    diagnostics = build_empty_estimation_diagnostics()
    diagnostics["map_has_id_column"] = "id" in result.columns

    if "id" not in result.columns:
        diagnostics["error_message"] = "The Brussels map file does not contain an 'id' column."
        result["est_speed"] = pd.NA
        return result, diagnostics

    try:
        prediction_df, diagnostics = run_tmp_model_inference(completed_snapshot_df)
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
