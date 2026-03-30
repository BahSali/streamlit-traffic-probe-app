from __future__ import annotations

import json
import streamlit as st
from urllib.parse import urlencode


def _clean(values):
    if not values:
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def render_brussels_api(
    selected_segment_names=None,
    selected_bus_ids=None,
    base_url: str = "http://localhost:8000/api/brussels/speeds",
):
    segment_names = _clean(selected_segment_names)
    bus_ids = _clean(selected_bus_ids)

    query_params = {
        "include_completed": "true",
        "include_estimated": "true",
    }

    if segment_names:
        query_params["segment_names"] = ",".join(segment_names)

    if bus_ids:
        query_params["bus_lines"] = ",".join(bus_ids)

    request_url = f"{base_url}?{urlencode(query_params)}"

    sample_response = {
        "city": "Brussels",
        "filters": {
            "segment_names": segment_names,
            "bus_lines": bus_ids,
        },
        "count": 1,
        "results": [
            {
                "segment_id": "1001",
                "segment_name": "Example segment",
                "bus_lines": ["12"],
                "completed_bus_speed_kmh": 18.2,
                "estimated_speed_kmh": 19.7,
            }
        ],
    }

    # --- UI rendering ---
    st.caption(
        "Retrieve completed bus-derived speeds and model-based estimated speeds "
        "for the current selection."
    )

    st.markdown("**Sample JSON Response**")
    st.code(json.dumps(sample_response, indent=2), language="json")
