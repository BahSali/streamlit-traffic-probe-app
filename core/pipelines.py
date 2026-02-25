import os
import pandas as pd

from STIB_dataset_generator import main as generate_dataset
from fusion_use import main as run_model

def run_estimation_pipeline(results_path="results.csv", sep=";", force=False):
    """
    Idempotent: if results exist, don't work, unless force=True.
    """
    if (not force) and os.path.exists(results_path):
        return results_path

    generate_dataset()
    run_model()

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found after pipeline run.")
    return results_path

def load_results_dict(results_path="results.csv", sep=";"):
    df = pd.read_csv(results_path, sep=sep)
    df["SegmentID"] = df["SegmentID"].astype(str)
    return {row["SegmentID"]: row.to_dict() for _, row in df.iterrows()}
