"""Brussels travel time (cities/brussels/travel_time.py): formula, sections, common basis, no double counting."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from cities.brussels import travel_time as tt


def segments():
    """Chain 1->2->3->4 (direction 1, lines 71/13+88), a lone segment, a direction-2 segment that starts where
    the chain ends (must not link), and two segments leaving node 10 (a branch: no link into them)."""
    rows = [  # id, start, end, direction, lines, length (m)
        ("a", 1, 2, 1, ["71"], 400), ("b", 2, 3, 1, ["13", "88"], 250), ("c", 3, 4, 1, ["71", "13", "88"], 500),
        ("d", 50, 51, 1, ["71"], 300), ("e", 4, 5, 2, ["71"], 300),
        ("f", 9, 10, 1, ["95"], 300), ("g", 10, 11, 1, ["95"], 300), ("h", 10, 12, 1, ["95"], 300),
    ]
    return pd.DataFrame(
        [dict(segment_id=i, segment_name=f"S{i}", bus_line_list=l, direction=d, start_id=s, end_id=e, fid=n, length_m=L)
         for n, (i, s, e, d, l, L) in enumerate(rows)]
    )


def test_sections_are_simple_ordered_paths_and_segments_are_counted_once():
    ordered = tt.order_sections(segments())
    assert [list(g.segment_id) for _, g in ordered.groupby("section")] == [["a", "b", "c"], ["d"], ["e"], ["f"], ["g"], ["h"]]
    # Selecting two lines that share segments gives each segment once.
    assert list(tt.scope_segments(segments(), ((), ("13", "88", "71"))).segment_id) == ["a", "b", "c", "d", "e"]


def test_times_use_the_formula_and_googles_reported_duration_on_one_common_basis():
    ordered, routes = tt.group_segments(segments().query("segment_id in ['a', 'b', 'c']"), [])  # one connected section
    snapshot = pd.DataFrame(
        {
            "segment_id": ["a", "b", "c"],
            "estimated_speed": [20.0, 10.0, 25.0],
            "google_speed_kmh": [np.nan, 12.0, 30.0],
            "google_duration_seconds": [np.nan, 75.0, 60.0],  # no Google value for a
        }
    )
    table = tt.build_table(ordered, routes, snapshot, {"a": 40.0, "b": 20.0, "c": 50.0})
    assert table.attrs["sources"] == ["google", "primary", "tabpfn"]  # Google first, like STIB in the other charts
    assert table["time_primary"].round(6).tolist() == [72.0, 90.0, 72.0]  # 3.6 * L / v
    assert table["time_google"].fillna(-1).tolist() == [-1, 75.0, 60.0]  # reported durations
    assert table["in_basis"].tolist() == [False, True, True]  # segment a lacks Google
    assert tt.totals(table) == {"primary": 162.0, "tabpfn": 45.0 + 36.0, "google": 135.0}
    assert tt.format_time(45.0) == "45 s (0.75 min)" and tt.format_time(125.0) == "125 s (2.08 min)"
    export = tt.export_table(table)  # what the CSV download contains
    assert list(export.columns) == [
        "segment_id", "street_name", "section", "route", "bus_lines", "length_m", "google_speed_kmh", "google_time_s",
        "traffic_estimation_model_speed_kmh", "traffic_estimation_model_time_s", "tabpfn_speed_kmh", "tabpfn_time_s", "used_in_totals",
    ]
    assert export["used_in_totals"].tolist() == [False, True, True]


def test_routes_follow_the_selected_lines_and_shared_segments_are_counted_once():
    scoped = tt.scope_segments(segments(), ((), ("13", "71")))  # a, b, c, d, e; c is on both lines
    grouped, routes = tt.group_segments(scoped, ["13", "71"])
    assert [(r["name"], r["ids"], r["parts"]) for r in routes] == [
        ("Line 13 · direction 1", ["b", "c"], 1),
        ("Line 71 · direction 1", ["a", "c", "d"], 3),  # cut into parts where the line leaves the selection
        ("Line 71 · direction 2", ["e"], 1),
    ]
    assert grouped["segment_id"].is_unique and sorted(grouped["segment_id"]) == ["a", "b", "c", "d", "e"]
    assert grouped.set_index("segment_id").loc["c", "route"] == "Line 13 · direction 1"  # shared: drawn once, under the first route

    snapshot = pd.DataFrame({"segment_id": list("abcde"), "estimated_speed": 18.0})  # t = 0.2 s per metre
    table = tt.build_table(grouped, routes, snapshot, None)
    assert table.set_index("segment_id").loc["c", "bus_lines"] == "13, 71, 88"  # every line serving the segment
    assert tt.shared_segments(table) == 1
    assert tt.totals(table) == {"primary": pytest.approx(350.0)}  # each segment once
    assert tt.route_summary(table)["primary"].tolist() == pytest.approx([150.0, 240.0, 60.0])  # rows include the shared c


@pytest.mark.parametrize("case", ["selection", "line", "single_no_google"])
def test_travel_time_section_renders(case):
    def page(case):
        import numpy as np
        import pandas as pd
        from cities.brussels import travel_time as tt
        from cities.brussels.travel_time_charts import render_travel_time

        seg = tt.segment_table()
        df = pd.DataFrame({"segment_id": seg.segment_id, "estimated_speed": 18.0})
        selection = {"selection": ((), ("71", "95")), "line": ((), ()), "single_no_google": (("TRONE - LUXEMBOURG",), ())}[case]
        if case != "single_no_google":
            df["google_duration_seconds"] = 60.0
            df["google_speed_kmh"] = 20.0
        render_travel_time(df, {i: 22.0 for i in seg.segment_id} if case != "single_no_google" else None, selection, "k")

    at = AppTest.from_function(page, args=(case,), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox[0].value == ("selection" if case != "line" else tt.bus_lines(tt.segment_table())[0])  # no selection: first line


def test_route_error_is_missing_never_zero_where_google_is_missing():
    from cities.brussels.travel_time_charts import NO_VALUE_COLOR, _error_color

    scoped = tt.scope_segments(segments(), ((), ("71",)))  # a, c, d, e
    snapshot = pd.DataFrame(
        {"segment_id": list("acde"), "estimated_speed": 18.0, "google_speed_kmh": [20.0, np.nan, np.nan, np.nan]}
    )
    grouped, routes = tt.group_segments(scoped, ["71"])
    error = tt.speed_error(tt.build_table(grouped, routes, snapshot, None), "primary")
    assert error.notna().sum() == 1 and error.dropna().iloc[0] == pytest.approx(2.0)  # only the Google segment has an error
    assert tt.lines_with_google(segments(), snapshot) == ["71"]
    assert _error_color(float("nan")) == NO_VALUE_COLOR and _error_color(0.0) != NO_VALUE_COLOR  # missing is grey, 0 km/h is a colour
