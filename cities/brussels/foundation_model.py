"""Foundation-model speed estimates for the optional fourth Brussels map.

Used by speed_layers.py only when core.config.SHOW_FOUNDATION_MODEL_MAP is
True. The display name comes from core.config.FOUNDATION_MODEL_NAME.

PLACEHOLDER: no foundation model is integrated yet. predict_foundation_model_speeds()
returns a constant so the map, colours, tooltips and synchronisation can be
checked. Replace its body with the real model's predictions; nothing else
needs to change.
"""
from __future__ import annotations

import pandas as pd

# PLACEHOLDER value (km/h) given to every road segment.
PLACEHOLDER_SPEED_KMH = 25.0


def predict_foundation_model_speeds(gdf: pd.DataFrame) -> pd.Series:
    """Estimated speed (km/h) for each row of the Brussels map gdf, aligned to gdf.index.

    Missing values are allowed and are shown as "N/A" on the map.
    """
    # PLACEHOLDER: constant dummy estimate, not a model prediction.
    return pd.Series(PLACEHOLDER_SPEED_KMH, index=gdf.index, dtype=float)
