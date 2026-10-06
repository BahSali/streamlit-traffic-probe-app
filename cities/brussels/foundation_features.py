"""TabPFN feature construction for the Brussels foundation-model map.

Reproduces the feature table the saved TabPFN model was trained on (see
data/tabpfn_model.json): one row per road segment with the STIB speed of the
segment and of its 10 closest connected segments at t, t-15, ..., t-135 min,
plus segment metadata and the time of day. Google speeds are not needed.

Two segment-ID namespaces meet here:
  fid  1..1366, the GPKG row key. adjacency_binary.csv is labelled by fid and
       the model's segment_id is fid - 1.
  id   the original segment ID (with gaps). MobilityTwin/STIB data and the app
       use it.
Every join goes through the explicit fid -> id table of StaticInputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse.csgraph import shortest_path

from core.config import DATA_DIR
from core.timing import timed

GPKG_PATH = DATA_DIR / "Brussels_map_6km.gpkg"
ADJACENCY_PATH = DATA_DIR / "adjacency_binary.csv"
MODEL_PATH = DATA_DIR / "tabpfn_model.json"

N_NEIGHBOURS = 10
N_LAGS = 10  # t-0 (current snapshot) and 9 previous quarter-hours
LAG_MINUTES = 15
KM_PER_DEGREE_LON = 70.3  # at 50.85 N, as in training
KM_PER_DEGREE_LAT = 111.2

STATIC_FEATURES = [
    "segment_id", "direction", "bus_lines", "start_lon", "start_lat", "end_lon", "end_lat",
    "minute_of_day", "weekday",
]
STIB_FEATURES = [
    name
    for k in range(N_LAGS)
    for name in [f"stib_self_t-{k}"] + [f"stib_n{j}_t-{k}" for j in range(N_NEIGHBOURS)]
]
# The exact training column order: X = df[FEATURES].
FEATURES = STATIC_FEATURES + STIB_FEATURES
# direction and bus_lines, as in the model record's categorical_features_indices.
CATEGORICAL_FEATURES = ["direction", "bus_lines"]


class FoundationInputError(Exception):
    """The static files, the STIB data or the features are unusable."""


@dataclass(frozen=True)
class StaticInputs:
    fid: np.ndarray  # fid per position, 1..N
    original_id: np.ndarray  # original segment id (str) per position
    metadata: pd.DataFrame  # STATIC_FEATURES' metadata columns, in position order
    neighbours: np.ndarray  # N x 10 positions of the closest segments, -1 = none


def neighbour_positions(adjacency: np.ndarray, start_lon, start_lat, end_lon, end_lat) -> np.ndarray:
    """The 10 closest connected segments of each segment (positions).

    Fewest hops in the undirected road graph, ties by Manhattan distance between
    segment midpoints; -1 where fewer than 10 are reachable.
    """
    hops = shortest_path(adjacency, unweighted=True, directed=False)
    np.fill_diagonal(hops, np.inf)  # never the segment itself
    x = (np.asarray(start_lon) + np.asarray(end_lon)) / 2 * KM_PER_DEGREE_LON
    y = (np.asarray(start_lat) + np.asarray(end_lat)) / 2 * KM_PER_DEGREE_LAT
    manhattan = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :])
    order = np.lexsort((manhattan, hops), axis=1)[:, :N_NEIGHBOURS]
    return np.where(np.isfinite(np.take_along_axis(hops, order, 1)), order, -1)


def build_static_inputs(segments: pd.DataFrame, adjacency: pd.DataFrame) -> StaticInputs:
    """segments: GPKG attributes indexed by fid; adjacency: labelled by fid."""
    fid = segments.index.to_numpy().astype(int)
    if len(fid) == 0 or not segments.index.is_unique:
        raise FoundationInputError("GPKG fid values are missing or not unique.")
    if not (adjacency.index.astype(int).to_numpy() == fid).all() or not (
        adjacency.columns.astype(int).to_numpy() == fid
    ).all():
        raise FoundationInputError("adjacency_binary.csv labels do not match the GPKG fid values.")
    matrix = adjacency.to_numpy()
    if matrix.shape[0] != matrix.shape[1]:
        raise FoundationInputError("The adjacency matrix is not square.")

    bus_lines = segments["bus_lines"].fillna("").astype(str)
    # Ordinal code of the sorted distinct bus_lines values, as in training.
    codes = {value: i for i, value in enumerate(sorted(bus_lines.unique()))}
    metadata = pd.DataFrame(
        {
            "direction": segments["direction"].to_numpy(),
            "bus_lines": bus_lines.map(codes).to_numpy(),
            "start_lon": segments["start_lon"].to_numpy(),
            "start_lat": segments["start_lat"].to_numpy(),
            "end_lon": segments["end_lon"].to_numpy(),
            "end_lat": segments["end_lat"].to_numpy(),
        }
    )
    with timed("tabpfn.neighbour_table", segments=len(fid)):
        neighbours = neighbour_positions(
            matrix, metadata.start_lon, metadata.start_lat, metadata.end_lon, metadata.end_lat
        )
    original_id = segments["id"].astype(str).str.strip().to_numpy()
    return StaticInputs(fid=fid, original_id=original_id, metadata=metadata, neighbours=neighbours)


@st.cache_resource(show_spinner=False)
def load_static_inputs(gpkg_path: str = str(GPKG_PATH), adjacency_path: str = str(ADJACENCY_PATH)) -> StaticInputs:
    """Static adjacency / metadata / neighbour lookup, built once per process."""
    with timed("tabpfn.static_files_read"):
        segments = gpd.read_file(gpkg_path, fid_as_index=True)
        adjacency = pd.read_csv(adjacency_path, index_col=0)
    with timed("tabpfn.static_inputs_build"):
        return build_static_inputs(segments, adjacency)


def lag_bucket_times(snapshot_time) -> list[pd.Timestamp]:
    """Quarter-hour buckets t-15 ... t-135 (t = the snapshot's quarter hour).

    Built explicitly, without clipping at midnight: the buckets of the previous
    day are needed just after midnight.
    """
    t = pd.Timestamp(snapshot_time).floor(f"{LAG_MINUTES}min")
    return [t - pd.Timedelta(minutes=LAG_MINUTES * k) for k in range(1, N_LAGS)]


def build_feature_frame(
    static: StaticInputs,
    current_speeds: pd.Series,
    history: pd.DataFrame,
    snapshot_time,
) -> pd.DataFrame:
    """The TabPFN feature table, one row per segment in static (fid) order.

    current_speeds: STIB speed at t, indexed by original segment id (str).
    history: STIB speeds indexed by original segment id (str), columns = bucket times.
    """
    t = pd.Timestamp(snapshot_time).floor(f"{LAG_MINUTES}min")
    n = len(static.fid)
    ids = pd.Index(static.original_id)

    lags = [pd.to_numeric(current_speeds, errors="coerce").reindex(ids).to_numpy(dtype=float)]
    for bucket in lag_bucket_times(snapshot_time):
        if bucket in history.columns:
            lags.append(pd.to_numeric(history[bucket], errors="coerce").reindex(ids).to_numpy(dtype=float))
        else:
            lags.append(np.full(n, np.nan))

    missing = static.neighbours < 0
    blocks = []
    for speeds in lags:
        of_neighbours = speeds[np.maximum(static.neighbours, 0)]
        of_neighbours[missing] = np.nan
        blocks += [speeds[:, None], of_neighbours]
    stib = np.concatenate(blocks, axis=1)

    df = pd.DataFrame(
        {
            "segment_id": static.fid - 1,
            "minute_of_day": t.hour * 60 + t.minute,
            "weekday": t.weekday(),
        }
    )
    df = pd.concat(
        [df, static.metadata, pd.DataFrame(stib, columns=STIB_FEATURES)],
        axis=1,
    )
    return df[FEATURES]
