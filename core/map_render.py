"""Map display helpers shared by the city pages."""
from collections.abc import Callable

import folium
import streamlit as st
from streamlit_folium import st_folium

from core.colors import legend_html


def center_from_bounds(gdf):
    minx, miny, maxx, maxy = gdf.total_bounds
    return [(miny+maxy)/2, (minx+maxx)/2]


def make_map(center, zoom):
    return folium.Map(location=center, zoom_start=zoom)


def show_folium_map(m: folium.Map, key: str) -> None:
    st_folium(m, width=850, height=550, key=key, returned_objects=[])


def show_map_with_legend(render_map: Callable[[], None], ratio=(4, 1)) -> None:
    """Draw a map next to the speed legend; render_map draws the map itself."""
    col_map, col_legend = st.columns(list(ratio), vertical_alignment="top")
    with col_map:
        render_map()
    with col_legend:
        st.markdown(legend_html(), unsafe_allow_html=True)
