"""Brussels "Live Traffic Overview": Google's citywide live traffic layer on a map.

Independent of the RUN, the TabPFN toggle and the Google Routes selection.

The map script runs in the app page itself (st.html with JavaScript), not in an
st.iframe: inside an iframe made with srcdoc the page's own address is
"about:srcdoc", which no HTTP-referrer restriction of an API key can match, and
Google then refuses the key (RefererNotAllowedMapError). In the app page the
address is the app's own, so the key (secret GOOGLE_LIVE_TRAFFIC_API_KEY) can
stay restricted to e.g. https://<app>.streamlit.app/* and to the Maps JavaScript
API only.

The page text is identical on every rerun, so the map is not reloaded; a new
RUN's id is part of it, so the map (and its traffic) is rebuilt then. The Google
script itself is loaded once per browser page.
Problems (no key, the script not loading, the key being refused) show a short
message in place of the map, with Google's exact error when it gives one, and
never affect the rest of the page.
"""
from __future__ import annotations

import json
import re
from string import Template

import streamlit as st

SECRET_NAME = "GOOGLE_LIVE_TRAFFIC_API_KEY"
CENTER = {"lat": 50.8503, "lng": 4.3517}  # Brussels
ZOOM = 12
HEIGHT = 420
LOAD_TIMEOUT_MS = 15000
# What to do about the errors Google reports most often (its exact error name is shown with the hint).
HINTS = {
    "RefererNotAllowedMapError": "Add the address below to the key's allowed HTTP referrers.",
    "ApiNotActivatedMapError": "Enable the Maps JavaScript API for the key's Google Cloud project.",
    "ApiTargetBlockedMapError": "Allow the Maps JavaScript API in the key's API restrictions.",
    "BillingNotEnabledMapError": "Enable billing for the key's Google Cloud project.",
    "InvalidKeyMapError": "Check that the key is copied correctly.",
}

PAGE = Template("""<div id="$box" style="position:relative;height:${height}px;width:100%;">
<div class="map" style="height:100%;width:100%;"></div>
<div class="msg" style="position:absolute;inset:0;display:none;align-items:center;justify-content:center;
  background:#f4f6f9;color:#4b5563;font:14px sans-serif;text-align:center;padding:16px;"></div>
</div>
<script>
(function () {
  // The RUN id is in this text, so the map is rebuilt only for a new RUN: $refresh
  const box = document.getElementById($box_json);
  const key = $key;
  const hints = $hints;
  function fail(text) {
    clearTimeout(timer);
    const msg = box.querySelector(".msg");
    msg.textContent = "Live traffic could not be loaded. " + text;
    msg.style.display = "flex";
  }
  // Google writes the exact reason of a refused key only to the console: keep the last one.
  if (!console.error.__liveTraffic) {
    const original = console.error;
    console.error = function (...args) {
      const text = args.join(" ");
      const code = /Google Maps JavaScript API error: (\\w+)/.exec(text);
      if (code) window.__liveTrafficError = {code: code[1], url: (/authorized: (\\S+)/.exec(text) || [])[1] || ""};
      return original.apply(console, args);
    };
    console.error.__liveTraffic = true;
  }
  window.gm_authFailure = () => setTimeout(() => {
    const e = window.__liveTrafficError || {};
    fail("Google refused the API key" + (e.code ? " (" + e.code + "). " + (hints[e.code] || "") : ".") +
         " Google sees this page as: " + (e.url || location.href));
  }, 300);
  const timer = setTimeout(() => fail("The map did not load in time."), $timeout);
  function init() {
    try {
      const map = new google.maps.Map(box.querySelector(".map"), {
        center: $center, zoom: $zoom, mapTypeControl: false, streetViewControl: false, fullscreenControl: true,
      });
      new google.maps.TrafficLayer().setMap(map);
      clearTimeout(timer);
    } catch (error) {
      fail("The map could not be created.");
    }
  }
  window.__liveTrafficInit = init;  // the Google script calls the latest one when it has loaded
  if (window.google && window.google.maps && window.google.maps.Map) {
    init();  // a later RUN: the API is already loaded
  } else if (!window.__liveTrafficScript) {
    window.__liveTrafficScript = true;
    const script = document.createElement("script");
    script.src = "https://maps.googleapis.com/maps/api/js?key=" + encodeURIComponent(key) + "&callback=__liveTrafficInit&loading=async";
    script.async = true;
    script.onerror = () => { window.__liveTrafficScript = false; fail("The Google Maps script could not be loaded."); };
    document.head.appendChild(script);
  }
})();
</script>""")


def _api_key() -> str | None:
    try:
        return (st.secrets.get(SECRET_NAME) or "").strip() or None
    except Exception:  # no secrets file at all
        return None


def build_live_traffic_html(api_key: str, refresh_token: str) -> str:
    box = "live-traffic-" + re.sub(r"[^A-Za-z0-9]", "-", refresh_token)
    return PAGE.substitute(
        box=box, box_json=json.dumps(box), height=HEIGHT, refresh=re.sub(r"[^A-Za-z0-9]", "-", refresh_token),
        key=json.dumps(api_key), hints=json.dumps(HINTS), center=json.dumps(CENTER), zoom=ZOOM, timeout=LOAD_TIMEOUT_MS,
    )


def render_live_traffic(refresh_token: str) -> None:
    """refresh_token: the current RUN's result id ("idle" before the first RUN)."""
    api_key = _api_key()
    if not api_key:
        st.caption(f"Live traffic is not available: the {SECRET_NAME} secret is not set.")
        return
    st.html(build_live_traffic_html(api_key, refresh_token), unsafe_allow_javascript=True)
