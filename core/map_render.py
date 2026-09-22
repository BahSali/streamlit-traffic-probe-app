import folium

def center_from_bounds(gdf):
    minx, miny, maxx, maxy = gdf.total_bounds
    return [(miny+maxy)/2, (minx+maxx)/2]

def make_map(center, zoom):
    return folium.Map(location=center, zoom_start=zoom)
