from __future__ import annotations

import json
from urllib.parse import urlencode


def _clean_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def build_brussels_api_info(
    selected_segment_names: list[str] | None = None,
    selected_bus_ids: list[str] | None = None,
    base_url: str = "http://localhost:8000/api/brussels/speeds",
) -> dict:
    selected_segment_names = _clean_values(selected_segment_names)
    selected_bus_ids = _clean_values(selected_bus_ids)

    query_params: dict[str, str] = {
        "include_completed": "true",
        "include_estimated": "true",
    }

    if selected_segment_names:
        query_params["segment_names"] = ",".join(selected_segment_names)

    if selected_bus_ids:
        query_params["bus_lines"] = ",".join(selected_bus_ids)

    request_url = f"{base_url}?{urlencode(query_params)}"

    sample_response = {
        "city": "Brussels",
        "filters": {
            "segment_names": selected_segment_names,
            "bus_lines": selected_bus_ids,
        },
        "count": 2,
        "results": [
            {
                "segment_id": "1001",
                "segment_name": "Avenue Louise - Chaussée de Vleurgat",
                "bus_lines": ["8", "93"],
                "completed_bus_speed_kmh": 17.8,
                "estimated_speed_kmh": 19.4,
            },
            {
                "segment_id": "1002",
                "segment_name": "Rue de la Loi - Avenue des Arts",
                "bus_lines": ["12"],
                "completed_bus_speed_kmh": 21.1,
                "estimated_speed_kmh": 20.2,
            },
        ],
    }

    return {
        "section_title": "Data API",
        "section_caption": (
            "Retrieve completed bus-derived speeds and model-based estimated speeds "
            "for the current Brussels selection."
        ),
        "endpoint": base_url,
        "method": "GET",
        "query_parameters": [
            {
                "name": "segment_names",
                "type": "string",
                "required": False,
                "description": "Comma-separated road segment names.",
            },
            {
                "name": "bus_lines",
                "type": "string",
                "required": False,
                "description": "Comma-separated bus line identifiers.",
            },
            {
                "name": "include_completed",
                "type": "boolean",
                "required": False,
                "description": "Whether to include completed bus-derived speeds.",
            },
            {
                "name": "include_estimated",
                "type": "boolean",
                "required": False,
                "description": "Whether to include model-based estimated speeds.",
            },
        ],
        "request_url": request_url,
        "sample_response_json": json.dumps(
            sample_response,
            indent=2,
            ensure_ascii=False,
        ),
    }
