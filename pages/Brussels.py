from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.colors import get_speed_color, legend_html
from core.data_sources import (
    load_completed_stib_snapshot,
    load_gpkg,
    load_live_stib_segment_speed_lookup,
)
from core.estimation.tmp import (
    attach_prediction_df_to_gdf,
    build_estimation_artifacts,
)
from core.google_routes.service import (
    GOOGLE_ROUTES_MONTHLY_LIMIT,
    attach_google_results_to_map_gdf,
    attach_google_results_to_snapshot_df,
    fetch_google_speeds_for_selected_segments,
    get_monthly_google_request_count,
)
from core.nav_panel import render_left_panel
from core.styles import inject_styles
from core.ui.brussels_controls import brussels_left_controls
from visualisation.brussels_results import render_brussels_results_visualisation

st.set_page_config(page_title="Brussels", layout="wide")
inject_styles()

settings_box, content_box = render_left_panel("Brussels")

MAP_PATH = "data/Brussels_map_6km.gpkg"
STIB_SECRET_KEY = "MOBILITY_TWIN_TOKEN"


if "brussels_colorized" not in st.session_state:
    st.session_state["brussels_colorized"] = False

if "brussels_applied_segment_names" not in st.session_state:
    st.session_state["brussels_applied_segment_names"] = []

if "brussels_applied_bus_ids" not in st.session_state:
    st.session_state["brussels_applied_bus_ids"] = []

if "brussels_refresh_key" not in st.session_state:
    st.session_state["brussels_refresh_key"] = 0

if "brussels_pending_google_fetch" not in st.session_state:
    st.session_state["brussels_pending_google_fetch"] = False

if "brussels_google_results_df" not in st.session_state:
    st.session_state["brussels_google_results_df"] = pd.DataFrame(
        columns=["segment_id", "google_speed_kmh", "google_duration_seconds"]
    )

if "brussels_google_diagnostics" not in st.session_state:
    used_count = get_monthly_google_request_count()
    st.session_state["brussels_google_diagnostics"] = {
        "selected_segment_count": 0,
        "group_count": 0,
        "request_count_planned": 0,
        "request_count_sent": 0,
        "success_count": 0,
        "failure_count": 0,
        "usage_month_key": None,
        "usage_monthly_limit": GOOGLE_ROUTES_MONTHLY_LIMIT,
        "usage_used_before_run": used_count,
        "usage_remaining_before_run": GOOGLE_ROUTES_MONTHLY_LIMIT - used_count,
        "usage_used_after_run": used_count,
        "usage_remaining_after_run": GOOGLE_ROUTES_MONTHLY_LIMIT - used_count,
        "was_requested": False,
        "error_message": None,
        "info_message": None,
    }


@st.cache_data(show_spinner=False)
def parse_bus_lines(value) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    tokens = re.findall(r"[A-Za-z0-9]+", text)
    return [token.strip() for token in tokens if token.strip()]


@st.cache_data(show_spinner=False)
def load_brussels_map(path: str = MAP_PATH):
    gdf = load_gpkg(path).copy()

    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf.reset_index(drop=True)
    gdf["map_fid"] = gdf.index.astype(int)

    for column in ["start_name", "end_name", "bus_lines"]:
        if column not in gdf.columns:
            gdf[column] = ""

    gdf["segment_name"] = (
        gdf["start_name"].fillna("").astype(str).str.strip()
        + " - "
        + gdf["end_name"].fillna("").astype(str).str.strip()
    )
    gdf["segment_name"] = gdf["segment_name"].replace(" - ", "").fillna("N/A")

    gdf["bus_lines_display"] = gdf["bus_lines"].fillna("").astype(str)
    gdf["bus_line_list"] = gdf["bus_lines"].apply(parse_bus_lines)

    return gdf

def build_segment_metadata_df(gdf: pd.DataFrame) -> pd.DataFrame:
    metadata_df = gdf.copy()

    if "id" not in metadata_df.columns:
        raise KeyError("Brussels map gdf must contain an 'id' column.")

    metadata_df["segment_id"] = metadata_df["id"].astype(str).str.strip()

    for column in ["segment_name", "bus_lines"]:
        if column not in metadata_df.columns:
            metadata_df[column] = ""

    metadata_df = (
        metadata_df[["segment_id", "segment_name", "bus_lines"]]
        .drop_duplicates(subset=["segment_id"])
        .reset_index(drop=True)
    )

    return metadata_df


def attach_segment_metadata(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    source_id_col: str = "segment_id",
) -> pd.DataFrame:
    if df.empty or source_id_col not in df.columns:
        return df.copy()

    result = df.copy()
    result[source_id_col] = result[source_id_col].astype(str).str.strip()

    metadata_for_merge = metadata_df.rename(columns={"segment_id": source_id_col})

    for col in ["segment_name", "bus_lines"]:
        if col in result.columns:
            result = result.drop(columns=[col])

    result = result.merge(
        metadata_for_merge,
        on=source_id_col,
        how="left",
        validate="many_to_one",
    )

    return result

@st.cache_data(show_spinner=False)
def get_filter_options():
    gdf = load_brussels_map()

    segment_options = sorted(
        [
            value
            for value in gdf["segment_name"].dropna().astype(str).unique().tolist()
            if value.strip()
        ]
    )

    bus_id_options = sorted(
        {
            bus_id
            for bus_list in gdf["bus_line_list"].tolist()
            for bus_id in bus_list
            if str(bus_id).strip()
        },
        key=lambda value: (len(str(value)), str(value)),
    )

    return segment_options, bus_id_options


def get_selected_mask(
    gdf: pd.DataFrame,
    selected_segment_names: list[str],
    selected_bus_ids: list[str],
) -> pd.Series:
    selected_segment_names = {
        str(value).strip()
        for value in (selected_segment_names or [])
        if str(value).strip()
    }
    selected_bus_ids = {
        str(value).strip()
        for value in (selected_bus_ids or [])
        if str(value).strip()
    }

    mask = pd.Series(False, index=gdf.index)

    if selected_segment_names:
        mask = mask | gdf["segment_name"].isin(selected_segment_names)

    if selected_bus_ids:
        mask = mask | gdf["bus_line_list"].apply(
            lambda bus_list: any(bus_id in selected_bus_ids for bus_id in bus_list)
        )

    return mask


def get_selected_map_fids(
    gdf: pd.DataFrame,
    selected_segment_names: list[str],
    selected_bus_ids: list[str],
) -> set[int]:
    mask = get_selected_mask(
        gdf=gdf,
        selected_segment_names=selected_segment_names,
        selected_bus_ids=selected_bus_ids,
    )

    selected_rows = gdf.loc[mask]

    if selected_rows.empty:
        return set()

    return set(selected_rows["map_fid"].astype(int).tolist())


def get_live_stib_token() -> str | None:
    return st.secrets.get(STIB_SECRET_KEY)


def format_speed(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.1f} km/h"


def format_duration_seconds(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{int(float(value))} s"


def build_dark_fallback_color(_: object = None) -> str:
    return "#222222"


def build_google_color(value) -> str:
    if value is None or pd.isna(value):
        return "#000000"

    return get_speed_color(float(value))


def convert_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def attach_live_stib_bus_speeds(gdf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    result = gdf.copy()

    diagnostics = {
        "token_found": False,
        "map_has_id_column": "id" in result.columns,
        "lookup_size": 0,
        "common_segment_ids": 0,
        "matched_segments": 0,
        "error_message": None,
    }

    result["bus_speed"] = pd.NA

    token = get_live_stib_token()
    if not token:
        diagnostics["error_message"] = "Missing MobilityTwin token in Streamlit secrets."
        return result, diagnostics

    diagnostics["token_found"] = True

    if "id" not in result.columns:
        diagnostics["error_message"] = "The Brussels map file does not contain an 'id' column."
        return result, diagnostics

    try:
        speed_lookup = load_live_stib_segment_speed_lookup(
            token=token,
            gpkg_path=MAP_PATH,
        )
    except Exception as exc:
        diagnostics["error_message"] = f"Live STIB data could not be loaded: {exc}"
        return result, diagnostics

    diagnostics["lookup_size"] = len(speed_lookup)

    result["segment_id_str"] = result["id"].astype(str)
    map_segment_ids = set(result["segment_id_str"].tolist())
    lookup_segment_ids = set(speed_lookup.keys())

    diagnostics["common_segment_ids"] = len(map_segment_ids.intersection(lookup_segment_ids))
    result["bus_speed"] = result["segment_id_str"].map(speed_lookup)
    diagnostics["matched_segments"] = int(result["bus_speed"].notna().sum())

    return result, diagnostics


@st.cache_data(show_spinner=False, ttl=90)
def get_completed_snapshot_for_ui(refresh_key: int) -> pd.DataFrame:
    token = get_live_stib_token()
    if not token:
        return pd.DataFrame()

    return load_completed_stib_snapshot(
        token=token,
        gpkg_path=MAP_PATH,
        lookback_minutes=60,
        bucket_minutes=5,
        interpolation_method="latest",
    )


def finalize_map_columns(gdf: pd.DataFrame) -> pd.DataFrame:
    result = gdf.copy()

    if "bus_speed" not in result.columns:
        result["bus_speed"] = pd.NA

    if "est_speed" not in result.columns:
        result["est_speed"] = pd.NA

    if "google_speed" not in result.columns:
        result["google_speed"] = pd.NA

    if "google_duration_seconds" not in result.columns:
        result["google_duration_seconds"] = pd.NA

    result["bus_speed_str"] = result["bus_speed"].apply(format_speed)
    result["est_speed_str"] = result["est_speed"].apply(format_speed)
    result["google_speed_str"] = result["google_speed"].apply(format_speed)
    result["google_duration_str"] = result["google_duration_seconds"].apply(format_duration_seconds)

    result["bus_color"] = result["bus_speed"].apply(
        lambda value: get_speed_color(float(value))
        if value is not None and not pd.isna(value)
        else build_dark_fallback_color()
    )
    result["est_color"] = result["est_speed"].apply(
        lambda value: get_speed_color(float(value))
        if value is not None and not pd.isna(value)
        else build_dark_fallback_color()
    )
    result["google_color"] = result["google_speed"].apply(build_google_color)
    result["bus_highlight_color"] = result["bus_speed"].apply(
        lambda value: get_speed_color(float(value))
        if value is not None and not pd.isna(value)
        else build_dark_fallback_color()
    )
    result["est_highlight_color"] = result["est_speed"].apply(
        lambda value: get_speed_color(float(value))
        if value is not None and not pd.isna(value)
        else build_dark_fallback_color()
    )
    result["google_highlight_color"] = result["google_speed"].apply(build_google_color)

    return result

def reorder_columns(df: pd.DataFrame, priority_cols: list[str]) -> pd.DataFrame:
    existing_priority = [col for col in priority_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_priority]
    return df[existing_priority + remaining_cols]


def maybe_execute_google_fetch() -> None:
    """
    Execute Google Routes only when RUN has been pressed.
    """
    if not st.session_state.get("brussels_pending_google_fetch", False):
        return

    used_before_run = get_monthly_google_request_count()

    try:
        gdf = load_brussels_map().copy()

        selected_mask = get_selected_mask(
            gdf=gdf,
            selected_segment_names=st.session_state["brussels_applied_segment_names"],
            selected_bus_ids=st.session_state["brussels_applied_bus_ids"],
        )

        selected_google_gdf = gdf.loc[selected_mask].copy()

        google_result = fetch_google_speeds_for_selected_segments(
            selected_gdf=selected_google_gdf,
            monthly_limit=GOOGLE_ROUTES_MONTHLY_LIMIT,
        )

        st.session_state["brussels_google_results_df"] = google_result["google_results_df"]
        st.session_state["brussels_google_diagnostics"] = google_result["diagnostics"]

    except Exception as exc:
        st.session_state["brussels_google_results_df"] = pd.DataFrame(
            columns=["segment_id", "google_speed_kmh", "google_duration_seconds"]
        )
        st.session_state["brussels_google_diagnostics"] = {
            "selected_segment_count": 0,
            "group_count": 0,
            "request_count_planned": 0,
            "request_count_sent": 0,
            "success_count": 0,
            "failure_count": 0,
            "usage_month_key": None,
            "usage_monthly_limit": GOOGLE_ROUTES_MONTHLY_LIMIT,
            "usage_used_before_run": used_before_run,
            "usage_remaining_before_run": GOOGLE_ROUTES_MONTHLY_LIMIT - used_before_run,
            "usage_used_after_run": used_before_run,
            "usage_remaining_after_run": GOOGLE_ROUTES_MONTHLY_LIMIT - used_before_run,
            "was_requested": False,
            "error_message": f"Google Routes execution failed: {exc}",
            "info_message": None,
        }
    finally:
        st.session_state["brussels_pending_google_fetch"] = False


def prepare_brussels_page_payload(
    colorized: bool,
    selected_segment_names: tuple[str, ...],
    selected_bus_ids: tuple[str, ...],
    refresh_key: int,
) -> dict:
    gdf = load_brussels_map().copy()
    segment_metadata_df = build_segment_metadata_df(gdf)

    selected_map_fids = get_selected_map_fids(
        gdf=gdf,
        selected_segment_names=list(selected_segment_names),
        selected_bus_ids=list(selected_bus_ids),
    )

    diagnostics = {
        "token_found": False,
        "map_has_id_column": "id" in gdf.columns,
        "lookup_size": 0,
        "common_segment_ids": 0,
        "matched_segments": 0,
        "error_message": None,
    }

    estimation_diagnostics = {
        "estimation_mode": "disabled",
        "snapshot_found": False,
        "snapshot_time": None,
        "snapshot_bucket_time": None,
        "map_has_id_column": "id" in gdf.columns,
        "matched_segments": 0,
        "model_loaded": False,
        "historical_window_ready": False,
        "used_fallback_window": False,
        "error_message": None,
    }

    completed_snapshot_df = pd.DataFrame()
    enriched_snapshot_df = pd.DataFrame()

    google_results_df = st.session_state.get(
        "brussels_google_results_df",
        pd.DataFrame(columns=["segment_id", "google_speed_kmh", "google_duration_seconds"]),
    )
    google_diagnostics = st.session_state.get("brussels_google_diagnostics", {})

    if colorized:
        gdf, diagnostics = attach_live_stib_bus_speeds(gdf)

        token = get_live_stib_token()
        if token:
            try:
                completed_snapshot_df = get_completed_snapshot_for_ui(refresh_key)
                completed_snapshot_df = attach_segment_metadata(
                    completed_snapshot_df,
                    segment_metadata_df,
                    source_id_col="segment_id",
                )
                prediction_df, enriched_snapshot_df, estimation_diagnostics = build_estimation_artifacts(
                    completed_snapshot_df=completed_snapshot_df,
                    token=token,
                    gpkg_path=MAP_PATH,
                )

                gdf, matched_segments = attach_prediction_df_to_gdf(
                    gdf=gdf,
                    prediction_df=prediction_df,
                )
                estimation_diagnostics["matched_segments"] = matched_segments

            except Exception as exc:
                gdf["est_speed"] = pd.NA
                estimation_diagnostics = {
                    "estimation_mode": "pt_inference_historical_tmp",
                    "snapshot_found": False,
                    "snapshot_time": None,
                    "snapshot_bucket_time": None,
                    "map_has_id_column": "id" in gdf.columns,
                    "matched_segments": 0,
                    "model_loaded": False,
                    "historical_window_ready": False,
                    "used_fallback_window": False,
                    "error_message": f"Temporary estimation failed: {exc}",
                }
        else:
            gdf["est_speed"] = pd.NA
            estimation_diagnostics = {
                "estimation_mode": "pt_inference_historical_tmp",
                "snapshot_found": False,
                "snapshot_time": None,
                "snapshot_bucket_time": None,
                "map_has_id_column": "id" in gdf.columns,
                "matched_segments": 0,
                "model_loaded": False,
                "historical_window_ready": False,
                "used_fallback_window": False,
                "error_message": "Missing MobilityTwin token in Streamlit secrets.",
            }

        gdf = attach_google_results_to_map_gdf(
            gdf=gdf,
            google_results_df=google_results_df,
        )

        if not enriched_snapshot_df.empty:
            enriched_snapshot_df = attach_google_results_to_snapshot_df(
                snapshot_df=enriched_snapshot_df,
                google_results_df=google_results_df,
            )
        enriched_snapshot_df = attach_segment_metadata(
            enriched_snapshot_df,
            segment_metadata_df,
            source_id_col="segment_id",
        )
    else:
        gdf["bus_speed"] = pd.NA
        gdf["est_speed"] = pd.NA
        gdf["google_speed"] = pd.NA
        gdf["google_duration_seconds"] = pd.NA

    gdf = finalize_map_columns(gdf)

    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    geojson = json.loads(gdf.to_json())

    return {
        "geojson": geojson,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "selected_google_count": int(gdf["google_speed"].notna().sum()),
        "segment_count": int(len(gdf)),
        "live_bus_count": int(gdf["bus_speed"].notna().sum()),
        "diagnostics": diagnostics,
        "estimation_diagnostics": estimation_diagnostics,
        "completed_snapshot_df": completed_snapshot_df,
        "enriched_snapshot_df": enriched_snapshot_df,
        "google_diagnostics": google_diagnostics,
    }


def build_three_map_html(geojson_obj, center_lat, center_lon):
    geojson_str = json.dumps(geojson_obj)

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
  />
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      font-family: Arial, sans-serif;
      background: #ffffff;
    }}

    .wrapper {{
      width: 100%;
      height: 600px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .titles {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      font-weight: 700;
      color: #1f2937;
    }}

    .titles div {{
      text-align: center;
      padding: 8px 6px;
      background: #f3f4f6;
      border-radius: 8px;
      border: 1px solid #e5e7eb;
      line-height: 1.25;
      font-size: 13px;
    }}

    .maps {{
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      min-height: 0;
    }}

    .map {{
      width: 100%;
      height: 520px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      overflow: hidden;
    }}

    .leaflet-tooltip {{
      padding: 0;
      margin: 0;
      border-radius: 6px;
    }}

    .leaflet-tooltip-content {{
      font-size: 12px;
      line-height: 1.35;
      margin: 6px 8px;
      min-width: 180px;
      max-width: 240px;
      white-space: normal;
      word-break: normal;
      overflow-wrap: break-word;
    }}

    .metric-highlight {{
      display: inline-block;
      color: #ffffff;
      padding: 1px 5px;
      border-radius: 4px;
      font-weight: 700;
      margin-left: 3px;
    }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="titles">
      <div>Observed Bus-Derived Speeds (STIB)</div>
      <div>Model-Derived Street Speed Estimates</div>
      <div>Google Routes API-Derived Speeds</div>
    </div>
    <div class="maps">
      <div id="map1" class="map"></div>
      <div id="map2" class="map"></div>
      <div id="map3" class="map"></div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/leaflet.sync@0.2.4/L.Map.Sync.min.js"></script>

  <script>
    const data = {geojson_str};

    const center = [{center_lat}, {center_lon}];
    const zoom = 12;

    const map1 = L.map('map1', {{ zoomControl: true }}).setView(center, zoom);
    const map2 = L.map('map2', {{ zoomControl: true }}).setView(center, zoom);
    const map3 = L.map('map3', {{ zoomControl: true }}).setView(center, zoom);

    const tile1 = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }});

    const tile2 = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }});

    const tile3 = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap'
    }});

    tile1.addTo(map1);
    tile2.addTo(map2);
    tile3.addTo(map3);

    map1.sync(map2);
    map1.sync(map3);
    map2.sync(map1);
    map2.sync(map3);
    map3.sync(map1);
    map3.sync(map2);

    function styleFactory(colorKey) {{
      return function(feature) {{
        return {{
          color: feature.properties[colorKey] || '#222222',
          weight: 3,
          opacity: 0.95
        }};
      }};
    }}

    function busTooltip(feature, layer) {{
      const p = feature.properties || {{}};

      const html = `
        <div>
          <b>Segment:</b> ${{p.segment_name ?? 'N/A'}}<br>
          <b>Bus lines:</b> ${{p.bus_lines_display ?? ''}}<br>
          <b>Bus speed:</b>
          <span class="metric-highlight" style="background:${{p.bus_highlight_color ?? '#222222'}};">
            ${{p.bus_speed_str ?? 'N/A'}}
          </span><br>
          <b>Estimated speed:</b> ${{p.est_speed_str ?? 'N/A'}}
        </div>
      `;
      layer.bindTooltip(html, {{
        sticky: true,
        direction: 'top',
        offset: [0, -8],
        opacity: 0.95
      }});
    }}

    function estimatedTooltip(feature, layer) {{
      const p = feature.properties || {{}};

      const html = `
        <div>
          <b>Segment:</b> ${{p.segment_name ?? 'N/A'}}<br>
          <b>Bus lines:</b> ${{p.bus_lines_display ?? ''}}<br>
          <b>Bus speed:</b> ${{p.bus_speed_str ?? 'N/A'}}<br>
          <b>Estimated speed:</b>
          <span class="metric-highlight" style="background:${{p.est_highlight_color ?? '#222222'}};">
            ${{p.est_speed_str ?? 'N/A'}}
          </span>
        </div>
      `;
      layer.bindTooltip(html, {{
        sticky: true,
        direction: 'top',
        offset: [0, -8],
        opacity: 0.95
      }});
    }}

    function googleTooltip(feature, layer) {{
      const p = feature.properties || {{}};

      const html = `
        <div>
          <b>Segment:</b> ${{p.segment_name ?? 'N/A'}}<br>
          <b>Bus lines:</b> ${{p.bus_lines_display ?? ''}}<br>
          <b>Bus speed:</b> ${{p.bus_speed_str ?? 'N/A'}}<br>
          <b>Estimated speed:</b> ${{p.est_speed_str ?? 'N/A'}}<br>
          <b>Google-reported speed:</b>
          <span class="metric-highlight" style="background:${{p.google_highlight_color ?? '#000000'}};">
            ${{p.google_speed_str ?? 'N/A'}}
          </span><br>
          <b>Google duration:</b> ${{p.google_duration_str ?? 'N/A'}}
        </div>
      `;
      layer.bindTooltip(html, {{
        sticky: true,
        direction: 'top',
        offset: [0, -8],
        opacity: 0.95
      }});
    }}

    const layer1 = L.geoJSON(data, {{
      style: styleFactory('bus_color'),
      onEachFeature: busTooltip
    }}).addTo(map1);

    const layer2 = L.geoJSON(data, {{
      style: styleFactory('est_color'),
      onEachFeature: estimatedTooltip
    }}).addTo(map2);

    const layer3 = L.geoJSON(data, {{
      style: styleFactory('google_color'),
      onEachFeature: googleTooltip
    }}).addTo(map3);

    try {{
      const bounds = layer1.getBounds();
      map1.fitBounds(bounds);
    }} catch (error) {{
      console.log(error);
    }}
  </script>
</body>
</html>
"""
    return html


segment_options, bus_id_options = get_filter_options()

controls = brussels_left_controls(
    settings_box,
    segment_options=segment_options,
    bus_id_options=bus_id_options,
    applied_segment_names=st.session_state["brussels_applied_segment_names"],
    applied_bus_ids=st.session_state["brussels_applied_bus_ids"],
)

if controls["colorize_clicked"]:
    st.session_state["brussels_applied_segment_names"] = list(
        controls["filters"]["segment_names"]
    )
    st.session_state["brussels_applied_bus_ids"] = list(
        controls["filters"]["bus_ids"]
    )
    st.session_state["brussels_colorized"] = True
    st.session_state["brussels_refresh_key"] += 1
    st.session_state["brussels_pending_google_fetch"] = True
    st.rerun()

if controls["reset_clicked"]:
    st.session_state["brussels_colorized"] = False
    st.session_state["brussels_applied_segment_names"] = []
    st.session_state["brussels_applied_bus_ids"] = []
    st.session_state["brussels_refresh_key"] += 1
    st.session_state["brussels_pending_google_fetch"] = False
    st.session_state["brussels_google_results_df"] = pd.DataFrame(
        columns=["segment_id", "google_speed_kmh", "google_duration_seconds"]
    )
    used_count = get_monthly_google_request_count()
    st.session_state["brussels_google_diagnostics"] = {
        "selected_segment_count": 0,
        "group_count": 0,
        "request_count_planned": 0,
        "request_count_sent": 0,
        "success_count": 0,
        "failure_count": 0,
        "usage_month_key": None,
        "usage_monthly_limit": GOOGLE_ROUTES_MONTHLY_LIMIT,
        "usage_used_before_run": used_count,
        "usage_remaining_before_run": GOOGLE_ROUTES_MONTHLY_LIMIT - used_count,
        "usage_used_after_run": used_count,
        "usage_remaining_after_run": GOOGLE_ROUTES_MONTHLY_LIMIT - used_count,
        "was_requested": False,
        "error_message": None,
        "info_message": None,
    }
    st.rerun()


maybe_execute_google_fetch()

with content_box:
    st.markdown("<h2 style='color:#009688;'>Brussels</h2>", unsafe_allow_html=True)
    st.caption(
        "Three synced maps for bus-derived, model-derived, and Google-derived speed comparison."
    )

    with st.spinner("Preparing Brussels maps and speed layers..."):
        payload = prepare_brussels_page_payload(
            colorized=st.session_state["brussels_colorized"],
            selected_segment_names=tuple(st.session_state["brussels_applied_segment_names"]),
            selected_bus_ids=tuple(st.session_state["brussels_applied_bus_ids"]),
            refresh_key=st.session_state["brussels_refresh_key"],
        )

    diagnostics = payload["diagnostics"]
    estimation_diagnostics = payload.get("estimation_diagnostics", {})
    enriched_snapshot_df = payload.get("enriched_snapshot_df", pd.DataFrame())
    google_diagnostics = payload.get("google_diagnostics", {})

    if diagnostics["error_message"]:
        st.warning(diagnostics["error_message"])

    st.caption(
        "Live STIB diagnostics — "
        f"token: {'yes' if diagnostics['token_found'] else 'no'}, "
        f"map has id: {'yes' if diagnostics['map_has_id_column'] else 'no'}, "
        f"lookup size: {diagnostics['lookup_size']}, "
        f"common segment ids: {diagnostics['common_segment_ids']}, "
        f"matched segments: {diagnostics['matched_segments']}"
    )

    if estimation_diagnostics:
        st.caption(
            "Map 2 estimation — "
            f"mode: {estimation_diagnostics.get('estimation_mode', 'unknown')}, "
            f"snapshot found: {'yes' if estimation_diagnostics.get('snapshot_found') else 'no'}, "
            f"snapshot time: {estimation_diagnostics.get('snapshot_time') or 'N/A'}, "
            f"bucket time: {estimation_diagnostics.get('snapshot_bucket_time') or 'N/A'}, "
            f"model loaded: {'yes' if estimation_diagnostics.get('model_loaded') else 'no'}, "
            f"historical ready: {'yes' if estimation_diagnostics.get('historical_window_ready') else 'no'}, "
            f"fallback window: {'yes' if estimation_diagnostics.get('used_fallback_window') else 'no'}, "
            f"matched segments: {estimation_diagnostics.get('matched_segments', 0)}"
        )

        st.caption(
            "Historical coverage — "
            f"{estimation_diagnostics.get('historical_non_null_counts', {})}"
        )

    if google_diagnostics:
        st.caption(
            "Google Routes diagnostics — "
            f"requested: {'yes' if google_diagnostics.get('was_requested') else 'no'}, "
            f"selected segments: {google_diagnostics.get('selected_segment_count', 0)}, "
            f"groups: {google_diagnostics.get('group_count', 0)}, "
            f"planned requests: {google_diagnostics.get('request_count_planned', 0)}, "
            f"sent requests: {google_diagnostics.get('request_count_sent', 0)}, "
            f"success: {google_diagnostics.get('success_count', 0)}, "
            f"failure: {google_diagnostics.get('failure_count', 0)}, "
            f"monthly used after run: {google_diagnostics.get('usage_used_after_run', 0)}, "
            f"monthly remaining: {google_diagnostics.get('usage_remaining_after_run', 0)}, "
            f"monthly limit: {google_diagnostics.get('usage_monthly_limit', GOOGLE_ROUTES_MONTHLY_LIMIT)}"
        )

    if estimation_diagnostics.get("error_message"):
        st.warning(estimation_diagnostics["error_message"])

    if google_diagnostics.get("info_message"):
        st.info(google_diagnostics["info_message"])

    if google_diagnostics.get("error_message"):
        st.warning(google_diagnostics["error_message"])

    html = build_three_map_html(
        payload["geojson"],
        payload["center_lat"],
        payload["center_lon"],
    )

    col_map, col_legend = st.columns([10, 1], vertical_alignment="top")

    with col_map:
        components.html(html, height=640, scrolling=False)

    with col_legend:
        st.markdown(legend_html(), unsafe_allow_html=True)

    st.markdown("### Results")
    priority_cols = ["timestamp", "segment_id", "segment_name", "bus_lines"]
    existing_priority_cols = [c for c in priority_cols if c in enriched_snapshot_df.columns]
    
    enriched_snapshot_df = enriched_snapshot_df[
        existing_priority_cols + [c for c in enriched_snapshot_df.columns if c not in existing_priority_cols]
    ]
    if st.session_state["brussels_colorized"] and not enriched_snapshot_df.empty:
        enriched_snapshot_df = reorder_columns(
            enriched_snapshot_df,
            ["timestamp", "segment_id", "segment_name", "bus_lines"],
        )
        st.download_button(
            label="Download results",
            data=convert_dataframe_to_csv_bytes(enriched_snapshot_df),
            file_name="results.csv",
            mime="text/csv",
            use_container_width=False,
        )

    st.markdown("---")
    st.markdown("### Visualisation")
    if st.session_state["brussels_colorized"] and not enriched_snapshot_df.empty:
        render_brussels_results_visualisation(enriched_snapshot_df)
    
    st.markdown("---")
    st.markdown("### Overview")

    google_used = google_diagnostics.get("usage_used_after_run", get_monthly_google_request_count())
    google_remaining = google_diagnostics.get(
        "usage_remaining_after_run",
        GOOGLE_ROUTES_MONTHLY_LIMIT - google_used,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Google used", google_used)
    col2.metric("Google left", google_remaining)
    col3.metric("Google segments", payload["selected_google_count"])
    col4.metric("STIB live", payload["live_bus_count"])
