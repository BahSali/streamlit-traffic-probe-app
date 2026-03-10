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

def legend_html():
    return """
    <div style='font-weight:bold;margin-bottom:8px;'>Color Key</div>
    <div style='line-height:2;'>
      <span style="display:inline-block;width:22px;height:18px;background:#8B0000;border-radius:4px;margin-right:8px;"></span> 0–10<br>
      <span style="display:inline-block;width:22px;height:18px;background:#FF0000;border-radius:4px;margin-right:8px;"></span> 10–20<br>
      <span style="display:inline-block;width:22px;height:18px;background:#FFA500;border-radius:4px;margin-right:8px;"></span> 20–30<br>
      <span style="display:inline-block;width:22px;height:18px;background:#FFFF00;border-radius:4px;margin-right:8px;"></span> 30–40<br>
      <span style="display:inline-block;width:22px;height:18px;background:#9ACD32;border-radius:4px;margin-right:8px;"></span> 40–50<br>
      <span style="display:inline-block;width:22px;height:18px;background:#00B050;border-radius:4px;margin-right:8px;"></span> 50+
    </div>
    """
