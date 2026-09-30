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
# Google Routes monthly request limit (Brussels page)
# ---------------------------------------------------------------------------
# Maximum Google Routes HTTP requests per calendar month (UTC), counted in the
# existing usage worksheet (secrets: [sheets]). A RUN whose planned requests
# do not fit in what is left sends none. Shown as "Google left" in Overview.
# Set e.g. 10 temporarily to test the limit.
GOOGLE_ROUTES_MONTHLY_LIMIT = 5000

# ---------------------------------------------------------------------------
# Optional foundation-model map (Brussels page)
# ---------------------------------------------------------------------------
# False: the usual three maps in one row (Bus | Estimated | Google).
# True:  a fourth synced map with the foundation model's estimates, in a 2x2
#        grid (Bus | Estimated / Foundation model | Google); its estimate is
#        also added to every map's tooltip.
# FOUNDATION_MODEL_NAME is only the name shown in the panel title and the
# tooltips; the data fields are generic (foundation_model_*), so another model
# only needs a new name here and new predictions in
# cities/brussels/foundation_model.py.
SHOW_FOUNDATION_MODEL_MAP = False
FOUNDATION_MODEL_NAME = "TabPFN"

# ---------------------------------------------------------------------------
# File locations (absolute, so the app works from any working directory)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
# Scratch files written at runtime (git-ignored).
OUTPUT_DIR = REPO_ROOT / "outputs"
