"""App-wide settings and file locations."""
from pathlib import Path

# ---------------------------------------------------------------------------
# File locations (absolute, so the app works from any working directory)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
# Scratch files written at runtime (git-ignored).
OUTPUT_DIR = REPO_ROOT / "outputs"
