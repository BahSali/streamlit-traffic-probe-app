# Urban Area Average Speed Estimator

Streamlit app that estimates road-segment traffic speeds from bus (STIB) probe
data and compares them with Google Routes speeds.

```bash
pip install -r requirements.txt
streamlit run app.py
```

`app.py` is the home page; each city/map is a page in `pages/`, reached through
the **Map selector** in the left panel.

## Secrets

Put these in `.streamlit/secrets.toml` (git-ignored) or the Streamlit Cloud
secrets settings. `MOBILITY_TWIN_TOKEN` can also be an environment variable.

| Key | Used by |
| --- | --- |
| `MOBILITY_TWIN_TOKEN` | Brussels live/historical STIB data; Ixelles-Etterbeek pipeline runs |
| `GOOGLE_MAPS_API_KEY` (or `GOOGLE_ROUTES_API_KEY`) | Brussels Google Routes speeds (paid API) |
| `gcp_service_account`, `sheets.spreadsheet_id`, `sheets.worksheet_name` | Brussels monthly Google request counter (Google Sheet) |
| `google_observations.spreadsheet_id`, `google_observations.worksheet_name` | Private log of Google Routes observations (see below); optional |

## Project layout

```
app.py                         Home page
pages/                         One file per city page (layout only)
  Ixelles_Etterbeek.py
  Brussels.py
cities/                        City-specific data loading, estimation and controls
  ixelles_etterbeek/
    pipeline.py                Entry point: run_estimation_pipeline(), load_results_dict()
    dataset_generator.py       Step 1: MobilityTwin -> STIB_speeds.csv (in outputs/)
    fusion_model.py            Step 2: CNN -> data/ixelles_etterbeek_results.csv
  brussels/
    map_data.py                Map loading, segment metadata, filter options
    controls.py                Left-panel filters and RUN / Reset buttons
    session.py                 Session state, RUN / Reset, Google Routes fetch
    speed_layers.py            Joins live STIB, model and Google speeds
    model.py                   Estimation model inference
    synced_maps.py/.html       The three synced Leaflet maps
    charts.py                  "Performance Analysis" charts
scripts/                       export_google_observations.py (private CSV exports)
core/                          Shared by all cities
  config.py                    Settings (demo correction flag) and file locations
  layout.py                    Page registry, setup_page(), left panel, page_header()
  styles.py, colors.py         CSS, speed colour scale and legend
  map_render.py                Map display helpers (map + legend, folium map)
  data_sources.py              Cached file loaders, MobilityTwin token and loaders
  stib_*.py                    MobilityTwin STIB live/historical data
  google_routes/service.py     Google Routes requests and monthly usage limit
  google_routes/observations.py  Private observation log and CSV exports
  estimation/correction.py     Demo speed correction
  pipelines.py                 Old import path for cities/ixelles_etterbeek/pipeline.py
data/                          Networks (CSV / GeoPackage) and committed results
models/                        Model weights
outputs/                       Runtime scratch files (git-ignored)
tests/                         Offline tests (no network, no paid APIs)
```

## Demo speed correction (Brussels)

**Switch:** `APPLY_DEMO_SPEED_CORRECTION` in [`core/config.py`](core/config.py).
It is `True` by default for the demo. Set it to `False` to show the model's
own estimates. No other file needs to change.

When `True`, the Brussels estimate for a segment is replaced wherever it
differs from that segment's Google speed by more than 8.5 km/h: it becomes the
Google speed minus a random 0–5 km/h, never below 0. The rule and its
parameters are in [`core/estimation/correction.py`](core/estimation/correction.py).
Only segments that have a Google speed (those selected before pressing RUN)
can change.

The flag affects every place the Brussels estimate appears:

- the middle map ("Estimated Speeds"): line colours and tooltips
- the "Estimated speed" line in the tooltips of the other two maps
- **Performance Analysis**: metrics, charts and the Data Preview tab
- **Results**: the `estimated_speed` column of the downloaded CSV

When `True`, the middle map is titled "Estimated Speeds" and the page caption
says "estimated"; when `False` they read "Estimated Speeds (Model)" and
"model-derived". The model's own estimates are kept internally and never
overwritten. They are not displayed or exported in either mode. The
Ixelles-Etterbeek page is not affected by this flag.

## Brussels RUN, caching and timing logs

Pressing **RUN** updates the page in one pass. The status box under the
button names the current stage (fetching Google speeds, loading bus data,
estimating speeds, updating maps). The maps keep showing the previous result,
without fading, until the new one replaces it. Each RUN's result is kept for
the session, so editing a filter afterwards redraws nothing and fetches
nothing; the left panel shows when the maps were last updated.

What is cached, and for how long:

| Data | Cache | Why it is safe |
| --- | --- | --- |
| Map geometry (GeoJSON) and uncoloured network | process lifetime | static file |
| GPKG segment metadata for the STIB fetchers | process lifetime | static file |
| Model checkpoint | process lifetime | static file |
| Live and last-hour STIB data, model estimates | 90 s (unchanged) | short enough to be current |
| Model history windows ending < 2 h ago | 5 min (unchanged) | recent data |
| Model history windows ending ≥ 2 h ago (1 day, 1–3 weeks) | 6 h | past data does not change |
| Google Routes speeds | reused only when the same selection is RUN again within 90 s (the page says so) | avoids paying twice for the same request |

Every stage is timed in the **server log** (Streamlit Cloud: *Manage app* →
logs), never on the page. Lines look like:

```
timing google_routes.requests 1.007s requests=28 segments=53
timing mobilitytwin.historical_request 0.607s caller=model window_start=... files=3 rows=...
timing brussels.run_update 9.946s
```

Main stages: `google_sheets.read_count` / `reserve` / `write_count`, `google_observations.store`,
`google_routes.requests`, `mobilitytwin.live_request`,
`mobilitytwin.historical_request` (per window, with file and row counts),
`brussels.live_bus_speeds`, `brussels.import_model`, `model.history_features`,
`model.inference`, `brussels.geojson_serialize`, `brussels.maps_html`,
`brussels.charts`, and `brussels.run_update` (the whole RUN). Set the
environment variable `ESTIMATOR_TIMING_LOG=0` to turn them off.

## Google Routes: monthly limit and private observation log

### Monthly request limit

**Setting:** `GOOGLE_ROUTES_MONTHLY_LIMIT` in [`core/config.py`](core/config.py)
(default 5000; set e.g. 10 to test). Overview ("Google used" / "Google left")
and the diagnostics line use this value.

Usage is kept in the existing worksheet (`[sheets]`, header
`month_key | request_count`, month in UTC as `YYYY-MM`). It is an append-only
log: each RUN that needs Google adds one row reserving its planned HTTP
requests (one per group of adjacent segments, not one per segment), and a
month's usage is the **sum** of that month's rows. An existing single total
row per month keeps counting. To correct the usage, add a row with the
difference.

- A RUN is checked before anything is sent: if its planned requests do not
  fit in what is left, nothing is sent, its row is set to 0, and the page
  shows the usage, the limit and the month.
- Every attempted request counts, including failed ones. Google results
  reused from the same selection within 90 s send and count nothing.
- Two users pressing RUN at the same moment: each reservation is appended as
  its own row (`INSERT_ROWS`) and only the rows above it count, so both
  cannot take the last requests. This relies on Google Sheets applying
  appends one after the other. If the sheet write fails after a reservation,
  that reservation still counts, so the error is on the safe side.

### Private observation log

Google data may not be redistributed. The route-leg values are therefore
stored only in a **separate private spreadsheet** and never shown, logged or
committed. Setup:

1. Create a new Google spreadsheet with a worksheet named e.g.
   `observations`. Share it only with your account and the service
   account's e-mail (Editor).
2. Add to the app's secrets:

   ```toml
   [google_observations]
   spreadsheet_id = "<the new spreadsheet's id>"
   worksheet_name = "observations"
   ```

   Without this section the app works as before and stores nothing (a
   warning is logged).

Each RUN that actually sends Google requests appends one batch in a single
call: `batch_id | batch_timestamp_utc | segment_id | distance_m | duration_s |
speed_kmh`. The first row of a batch has an empty `segment_id` and marks the
batch; then one row per segment for which Google returned a value.

- `batch_timestamp_utc`: when the batch's requests were sent, ISO 8601 in
  UTC with milliseconds, e.g. `2026-01-15T07:07:31.123Z`.
- `distance_m` and `duration_s`: the route leg's `distanceMeters` and
  `duration` as returned. `speed_kmh`: the speed the app derives and displays
  (a 0 s leg shorter than 20 m is treated as 1 s). Missing values stay empty.
- Reused Google results and RUNs blocked by the limit add nothing.
- Writes are retried; a batch already present (same `batch_id`) is never
  written again and rows are never updated. A write that failed is retried
  on the next rerun of the page.

**Exports** (only with the credentials, from your own machine):

```bash
python scripts/export_google_observations.py --secrets .streamlit/secrets.toml --out exports
```

This writes `google_distance_m.csv`, `google_duration_s.csv` and
`google_speed_kmh.csv`. Each has `timestamp_utc` first, then one column per
Brussels segment id (all 1,366, ascending, same order in the three files),
and one row per stored batch in time order. Cells without a returned value
are empty. `exports/` is git-ignored; never commit these files.

## Adding a city

1. **Data**: put the road network (CSV or GeoPackage) in `data/` and any model
   weights in `models/`. Refer to them through `core.config.DATA_DIR` /
   `MODELS_DIR`, not relative paths.
2. **City code**: create `cities/<city>/` for the code that only this city
   needs: loading its data, running its estimation, its controls. Start with
   one module (e.g. `pipeline.py`) and split only when it grows. The two
   existing cities are examples at both ends: `ixelles_etterbeek` is a
   pipeline plus a folium map, `brussels` has live data, three maps and Google
   Routes.
3. **Page**: create `pages/<City>.py`. Keep it to layout and use the shared
   helpers:

   ```python
   import folium
   import streamlit as st

   from cities.my_city.pipeline import load_segments, load_estimates
   from core.colors import get_speed_color
   from core.layout import page_header, setup_page
   from core.map_render import show_folium_map, show_map_with_legend

   settings_box, content_box = setup_page("My City")  # must be the first Streamlit call

   with settings_box:
       line_weight = st.slider("Line weight", 1.0, 8.0, 4.0, key="mycity_weight")

   with content_box:
       page_header("My City", "One-line description of the map.")
       segments = load_segments()
       estimates = load_estimates()  # {segment_id: speed_kmh}
       m = folium.Map(location=[50.84, 4.37], zoom_start=13, control_scale=True)
       for _, row in segments.iterrows():
           folium.PolyLine(
               row["coords"],
               color=get_speed_color(estimates.get(row["id"])),
               weight=line_weight,
           ).add_to(m)
       show_map_with_legend(lambda: show_folium_map(m, key="mycity_map"))
   ```

   Give widget keys and `st.session_state` entries a city prefix
   (`mycity_...`) so they don't clash with other pages.
4. **Register** the page in `PAGES` in [`core/layout.py`](core/layout.py):
   `"My City": "pages/My_City.py"`. Only files directly in `pages/` can be
   opened. Note that every file committed to this repository is public,
   including files in sub-folders of `pages/`.
5. **Secrets**: add any new keys to the table above and read them with
   `st.secrets`, never hard-coded.
6. **Test**: add a smoke test in `tests/test_pages.py` that runs the page with
   external calls faked (see `tests/conftest.py`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests fake MobilityTwin, Google Routes and Google Sheets, so they need no
secrets and make no network requests.
