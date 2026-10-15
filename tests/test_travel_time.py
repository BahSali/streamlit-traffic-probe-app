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
    ordered = tt.order_sections(segments()).query("section == 1")  # a, b, c
    snapshot = pd.DataFrame(
        {
            "segment_id": ["a", "b", "c"],
            "estimated_speed": [20.0, 10.0, 25.0],
            "google_speed_kmh": [np.nan, 12.0, 30.0],
            "google_duration_seconds": [np.nan, 75.0, 60.0],  # no Google value for a
        }
    )
    table = tt.build_table(ordered, snapshot, {"a": 40.0, "b": 20.0, "c": 50.0})
    assert table.attrs["sources"] == ["primary", "tabpfn", "google"]
    assert table["time_primary"].round(6).tolist() == [72.0, 90.0, 72.0]  # 3.6 * L / v
    assert table["time_google"].fillna(-1).tolist() == [-1, 75.0, 60.0]  # reported durations
    assert table["in_basis"].tolist() == [False, True, True]  # segment a lacks Google
    assert tt.totals(table) == {"primary": 162.0, "tabpfn": 45.0 + 36.0, "google": 135.0}
    [model_panel, tabpfn_panel] = tt.contributions(table)
    assert model_panel["reference"] == "google" and model_panel["total"] == pytest.approx(27.0)  # 162 - 135


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
