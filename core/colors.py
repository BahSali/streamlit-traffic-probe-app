"""Speed colour scale and legend used by every map."""
import pandas as pd

# Line colours for segments without a value.
NO_DATA_COLOR = "#222222"
NO_GOOGLE_DATA_COLOR = "#000000"


def get_speed_color(x):
    try:
        x = float(x)
    except Exception:
        return "gray"
    if x < 10: return "#8B0000"
    if x < 20: return "#FF0000"
    if x < 30: return "#FFA500"
    if x < 40: return "#FFFF00"
    if x < 50: return "#9ACD32"
    return "#00B050"


def speed_color_or(value, missing_color: str) -> str:
    """Colour for a speed value, or missing_color when the value is missing."""
    if value is None or pd.isna(value):
        return missing_color
    return get_speed_color(float(value))

# (lower bound, label) of the legend rows; the colour is that of get_speed_color.
LEGEND_RANGES = [(0, "0–10"), (10, "10–20"), (20, "20–30"), (30, "30–40"), (40, "40–50"), (50, "50+")]


def legend_html():
    """The speed legend. Each swatch stays on one line with its label (no wrapping between them);
    core.styles keeps the column that holds it wide enough (class speed-legend)."""
    rows = "".join(
        "<div style='display:flex;align-items:center;gap:6px;white-space:nowrap;line-height:1.9;'>"
        f"<span style='flex:none;width:18px;height:14px;border-radius:3px;background:{get_speed_color(low)};'></span>{label}</div>"
        for low, label in LEGEND_RANGES
    )
    return (
        "<div class='speed-legend' style='font-size:0.9rem;'>"
        "<div style='font-weight:bold;margin-bottom:6px;line-height:1.3;'>Speed <span style='display:inline-block;'>(km/h)</span></div>"
        f"{rows}</div>"
    )
