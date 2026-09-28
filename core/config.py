"""App-wide settings and file locations."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Demo speed correction (Brussels page)
# ---------------------------------------------------------------------------
# True  (demo default): the Brussels "estimated speed" is replaced by the demo
#       correction in core/estimation/correction.py wherever it differs from
#       the Google speed by more than the threshold. The corrected values are
#       used everywhere the estimate appears: middle map colours and tooltips,
#       Performance Analysis charts/metrics, Data Preview and the CSV download.
# False: all of those show the model's original estimates.
# The model's original estimates are never overwritten in either mode.
APPLY_DEMO_SPEED_CORRECTION = True

# ---------------------------------------------------------------------------
# File locations (absolute, so the app works from any working directory)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
# Scratch files written at runtime (git-ignored).
OUTPUT_DIR = REPO_ROOT / "outputs"
