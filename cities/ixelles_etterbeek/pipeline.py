"""Ixelles-Etterbeek estimation pipeline.

    MobilityTwin API --dataset_generator.py--> STIB_speeds.csv (work dir)
                     --fusion_model.py------> results CSV (read by the page)

The results CSV is committed, so the page shows those values without calling
the API; a new run happens only when the file is missing or the user ticks
"Force refresh".
"""
import os

import pandas as pd

from core.config import DATA_DIR, MODELS_DIR, OUTPUT_DIR
from core.data_sources import get_mobility_twin_token

NETWORK_CSV = DATA_DIR / "Brux_net.csv"
SEGMENTS_CSV = DATA_DIR / "Etterbeek_STIB_segments.csv"
RESULTS_CSV = DATA_DIR / "ixelles_etterbeek_results.csv"
MODEL_WEIGHTS = MODELS_DIR / "full_model_new_loss_v2.pth"
WORK_DIR = OUTPUT_DIR / "ixelles_etterbeek"
CSV_SEP = ";"


def run_estimation_pipeline(results_path=RESULTS_CSV, sep=CSV_SEP, force=False):
    """
    Idempotent: if results exist, don't work, unless force=True.
    """
    if (not force) and os.path.exists(results_path):
        return results_path

    # Imported here so torch/duckdb are only loaded when the pipeline really runs.
    from cities.ixelles_etterbeek import dataset_generator, fusion_model

    token = get_mobility_twin_token()
    if not token:
        raise RuntimeError("Missing MOBILITY_TWIN_TOKEN in Streamlit secrets or environment.")

    speeds_csv = dataset_generator.main(token=token, segments_csv=SEGMENTS_CSV, work_dir=WORK_DIR)
    fusion_model.main(
        weights_path=MODEL_WEIGHTS,
        speeds_csv=speeds_csv,
        segments_csv=SEGMENTS_CSV,
        output_path=results_path,
    )

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found after pipeline run.")
    return results_path


def load_results_dict(results_path=RESULTS_CSV, sep=CSV_SEP):
    df = pd.read_csv(results_path, sep=sep, encoding="latin1")
    df["SegmentID"] = df["SegmentID"].astype(str)
    return {row["SegmentID"]: row.to_dict() for _, row in df.iterrows()}
