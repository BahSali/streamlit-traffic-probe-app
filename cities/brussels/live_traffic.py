"""Brussels "Live Traffic Overview": Google's citywide live traffic layer on a map.

Independent of the RUN, the TabPFN toggle and the Google Routes selection. It is
a self-contained page embedded with st.iframe, like the synchronized maps: the
browser then sends the app's own address as the HTTP referrer, so the API key
(secret GOOGLE_LIVE_TRAFFIC_API_KEY) must be restricted to the app's address,
e.g. https://<app>.streamlit.app/*, and to the Maps JavaScript API only.

The page text is identical on every rerun, so the map is not reloaded; a new
RUN's id is part of it, so the map (and its traffic) is reloaded then.
Problems (no key, the script not loading, the key being refused) show a short
message in place of the map and never affect the rest of the page.
"""
from __future__ import annotations

import json

import streamlit as st

SECRET_NAME = "GOOGLE_LIVE_TRAFFIC_API_KEY"
CENTER = {"lat": 50.8503, "lng": 4.3517}  # Brussels
ZOOM = 12
HEIGHT = 420
LOAD_TIMEOUT_MS = 15000

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body, #map {{ height: 100%; margin: 0; }}
  #msg {{ position: absolute; inset: 0; display: none; align-items: center; justify-content: center;
          background: #f4f6f9; color: #4b5563; font: 14px sans-serif; text-align: center; padding: 16px; }}
</style></head>
<body>
<div id="map"></div><div id="msg"></div>
<script>
  // New RUN id: changes this page, so the map reloads (and fetches fresh traffic) only then.
  const refresh = {refresh};
  const key = {key};
  const msg = document.getElementById("msg");
  function fail(reason) {{
    msg.textContent = "Live traffic could not be loaded. " + reason;
    msg.style.display = "flex";
  }}
  const timer = setTimeout(() => fail("The map did not load in time."), {timeout});
  window.gm_authFailure = () => fail("Google refused the API key (check its allowed referrers and that the Maps JavaScript API is enabled).");
  window.initMap = () => {{
    try {{
      const map = new google.maps.Map(document.getElementById("map"), {{
        center: {center}, zoom: {zoom}, mapTypeControl: false, streetViewControl: false, fullscreenControl: true,
      }});
      new google.maps.TrafficLayer().setMap(map);
      clearTimeout(timer);
    }} catch (error) {{
      fail("The map could not be created.");
    }}
  }};
  const script = document.createElement("script");
  script.src = "https://maps.googleapis.com/maps/api/js?key=" + encodeURIComponent(key) + "&callback=initMap&loading=async";
  script.async = true;
  script.onerror = () => fail("The Google Maps script could not be loaded.");
  document.head.appendChild(script);
</script>
</body></html>"""


def _api_key() -> str | None:
    try:
        return st.secrets.get(SECRET_NAME) or None
    except Exception:  # no secrets file at all
        return None


def build_live_traffic_html(api_key: str, refresh_token: str) -> str:
    return PAGE.format(
        refresh=json.dumps(refresh_token), key=json.dumps(api_key), center=json.dumps(CENTER), zoom=ZOOM, timeout=LOAD_TIMEOUT_MS
    )


def render_live_traffic(refresh_token: str) -> None:
    """refresh_token: the current RUN's result id ("idle" before the first RUN)."""
    api_key = _api_key()
    if not api_key:
        st.caption(f"Live traffic is not available: the {SECRET_NAME} secret is not set.")
        return
    st.iframe(build_live_traffic_html(api_key, refresh_token), height=HEIGHT)
