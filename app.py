from STIB_dataset_generator import main as generate_dataset
from fusion_use import main as run_model
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import geopandas as gpd
import requests
from shapely.geometry import shape
import numpy as np

# ---------- Theme Style for Button ----------
btn_style = """
<style>
div.stButton > button:first-child {
    background-color: #009688;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.7em 1.5em;
    margin: 0.5em 0em;
    transition: background 0.2s;
}
div.stButton > button:first-child:hover {
    background-color: #00665c;
    color: #fff;
}
</style>
"""
st.markdown(btn_style, unsafe_allow_html=True)

# ---------- Page Title and Caption ----------
st.markdown(
    "<h1 style='text-align:center; color:#009688;'> Urban Area (Real-time) Average Speed Estimator</h1>",
    unsafe_allow_html=True
)
st.caption("Interactive visualisation of real-time probe-derived bus speeds alongside estimation of overall traffic conditions across road segments.")

# ---------- City Selector ----------
selected_page = st.selectbox(
    "Select area or city:",
    ["-- Select --", "Ixelles-Etterbeek", "Brussels", "York"]
)
st.markdown("---") 

# -------- Ixelles-Etterbeek -------
if selected_page == "Ixelles-Etterbeek":
    # ---------- Data Loading ----------
    df = pd.read_csv("Brux_net.csv", sep=';')
        
    df['id'] = df['id'].astype(str)
    
    with st.expander("📄 Show raw segment data"):
        st.dataframe(df)
    
    last_four_cols = df.columns[-4:]
    start_lat_col, start_lon_col, end_lat_col, end_lon_col = last_four_cols
    
    map_center = [
        df[start_lat_col].mean(),
        df[start_lon_col].mean()
    ]
    
    m = folium.Map(location=map_center, zoom_start=13)
    
    # ---------- Session State for Results ----------
    if "results_dict" not in st.session_state:
        st.session_state["results_dict"] = {}
    
    # ---------- Button & Processing ----------
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("Run Traffic Estimation (Click Me!)"):
            with st.spinner('⏳ Fetching Live Bus Speed (please wait)...'):
                generate_dataset()
            st.success('✅ Retrieve Live Bus Speeds!')
    
            with st.spinner('🤖 Running Estimator tool...'):
                run_model()
            st.success('✅ Estimation finished!')
    
            with st.spinner('📦 Loading estimation results...'):
                if os.path.exists('results.csv'):
                    results = pd.read_csv('results.csv', sep=';')
                    results['SegmentID'] = results['SegmentID'].astype(str)
                    st.session_state["results_dict"] = {
                        row['SegmentID']: row for _, row in results.iterrows()
                    }
                    st.success("🎉 Results loaded and mapped!")
                else:
                    st.error("results.csv not found. Please check your scripts.")
    
    results_dict = st.session_state["results_dict"]
    
    # ---------- Color Helper Function ----------
    def get_speed_color(pred):
        try:
            pred = float(pred)
        except:
            return "gray"
        if pred < 10:
            return "#8B0000"   # dark red
        elif pred < 20:
            return "#FF0000"   # red
        elif pred < 30:
            return "#FFA500"   # orange
        elif pred < 40:
            return "#FFFF00"   # yellow
        elif pred < 50:
            return "#9ACD32"   # light green
        else:
            return "#00B050"   # green
    
    # ---------- Draw Edges and Tooltips ----------
    for idx, row in df.iterrows():
        street_name = f"{row[df.columns[1]]} - {row[df.columns[2]]}"
        segment_id = str(row['id'])
        tooltip_text = f"Segment ID: {segment_id}<br> Name: {street_name}"
    
        color = "black"
        if results_dict and segment_id in results_dict:
            result_row = results_dict[segment_id]
            speed = result_row.get('Speed', 'N/A')
            prediction = result_row.get('Prediction', 'N/A')
            # tooltip_text += f"<br>STIB: {speed}<br>Estimation: {prediction}"
            tooltip_text += f"<br>STIB Speed: {float(speed):.1f}<br>Estimation: {float(prediction):.1f}"
            color = get_speed_color(prediction)
    
        folium.PolyLine(
            locations=[
                [row[start_lat_col], row[start_lon_col]],
                [row[end_lat_col], row[end_lon_col]]
            ],
            color=color,
            weight=4,
            tooltip=tooltip_text
        ).add_to(m)
    
    # ---------- Show Map and Color Legend ----------
    col1, col2 = st.columns([4, 1])
    with col1:
        st_folium(m, width=700, height=500)
    with col2:
        st.markdown("""
        <div style='font-weight:bold;margin-bottom:8px;'>Prediction Color Key</div>
        <div style='line-height:2;'>
            <span style="display:inline-block;width:22px;height:18px;background:#8B0000;border-radius:4px;margin-right:8px;"></span> 0–10
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#FF0000;border-radius:4px;margin-right:8px;"></span> 10–20
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#FFA500;border-radius:4px;margin-right:8px;"></span> 20–30
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#FFFF00;border-radius:4px;margin-right:8px;"></span> 30–40
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#9ACD32;border-radius:4px;margin-right:8px;"></span> 40–50
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#00B050;border-radius:4px;margin-right:8px;"></span> 50+
        </div>
        """, unsafe_allow_html=True)
    
    # # --- Google Maps Embed ---
    # GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
    # 
    # # Center of the map: use the same as your folium map
    # google_map_center = f"{map_center[0]},{map_center[1]}"
    # google_map_zoom = 13
    # 
    # google_maps_url = (
    #     f"https://www.google.com/maps/embed/v1/view"
    #     f"?key={GOOGLE_MAPS_API_KEY}"
    #     f"&center={google_map_center}"
    #     f"&zoom={google_map_zoom}"
    #     f"&maptype=roadmap"
    # )
    # 
    # st.markdown("---")  # Optional separator
    # st.markdown("<h4 style='text-align:center;color:#009688;'>Google Maps (Live Traffic)</h4>", unsafe_allow_html=True)
    # components.iframe(google_maps_url, width=700, height=500)


# -------- Brussels --------
if selected_page == "Brussels":

    # ---------- Button & Processing ----------
    col1, col2, col3 = st.columns([2, 3, 2])

    # Initialize state once
    if "colorized" not in st.session_state:
        st.session_state["colorized"] = False
    if "brussels_speeds" not in st.session_state:
        st.session_state["brussels_speeds"] = None

    # Button with callback to avoid rerun loop
    with col2:
        st.button(
            "Run Traffic Estimation (Click Me!)",
            key="run_colorize",
            on_click=lambda: st.session_state.update(colorized=True)
        )

    # --- Cached data fetch to avoid repeating downloads ---
    @st.cache_data(show_spinner=False)
    def fetch_stib_geojson():
        token = "dd246f5afde3eceba4aa392777df19de4f1b2e71339a2450c24e70e83a54ad3dbe1cff1a226b4f1b4a5954cab691d4e5eb4b74ab699520770bfba723041e9dca"
        url = "https://api.mobilitytwin.brussels/stib/shapefile"
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        shapefile_data = response.json()

        geometries = [shape(f["geometry"]) for f in shapefile_data["features"]]
        df = pd.DataFrame([f["properties"] for f in shapefile_data["features"]])
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
        return gdf

    gdf = fetch_stib_geojson()

    def get_speed_color(pred):
        try:
            pred = float(pred)
        except:
            return "gray"
        if pred < 10:
            return "#8B0000"   # dark red
        elif pred < 20:
            return "#FF0000"   # red
        elif pred < 30:
            return "#FFA500"   # orange
        elif pred < 40:
            return "#FFFF00"   # yellow
        elif pred < 50:
            return "#9ACD32"   # light green
        else:
            return "#00B050"   # green

    # Create test speeds only once after clicking the button
    if st.session_state["colorized"] and st.session_state["brussels_speeds"] is None:
        st.session_state["brussels_speeds"] = {
            i: float(3 + 30 * np.random.rand()) for i in range(len(gdf))
        }

    # --- Map setup like Ixelles-Etterbeek ---
    map_center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=map_center, zoom_start=11)

    # --- Styling by category (optional visual variety) ---
    for idx, row in gdf.iterrows():
        if st.session_state["colorized"] and st.session_state["brussels_speeds"] is not None:
            speed = st.session_state["brussels_speeds"][idx]
            color = get_speed_color(speed)
        else:
            color = "black"  # default color before pressing button
    
        tooltip = f"<b>Line:</b> {row.get('ligne', 'N/A')}<br><b>Variant:</b> {row.get('variante', 'N/A')}"
        folium.GeoJson(
            row.geometry.__geo_interface__,
            tooltip=tooltip,
            style_function=lambda x, color=color: {"color": color, "weight": 2.5}
        ).add_to(m)

    # --- Map and legend side-by-side like Ixelles-Etterbeek ---
    col1, col2 = st.columns([4, 1])
    with col1:
        st_folium(m, width=700, height=500)
    with col2:
        st.markdown("""
        <div style='font-weight:bold;margin-bottom:8px;'>Prediction Color Key</div>
        <div style='line-height:2;'>
            <span style="display:inline-block;width:22px;height:18px;background:#8B0000;border-radius:4px;margin-right:8px;"></span> 0–10
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#FF0000;border-radius:4px;margin-right:8px;"></span> 10–20
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#FFA500;border-radius:4px;margin-right:8px;"></span> 20–30
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#FFFF00;border-radius:4px;margin-right:8px;"></span> 30–40
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#9ACD32;border-radius:4px;margin-right:8px;"></span> 40–50
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#00B050;border-radius:4px;margin-right:8px;"></span> 50+
        </div>
        """, unsafe_allow_html=True)


# -------- York --------
if selected_page == "York":

    # ---------- Button & Processing ----------
    col1, col2, col3 = st.columns([2, 3, 2])

    # initialize session state
    if "colorized_york" not in st.session_state:
        st.session_state["colorized_york"] = False
    if "york_speeds" not in st.session_state:
        st.session_state["york_speeds"] = None

    # Button with callback
    with col2:
        st.button("Run Traffic Estimation (Click Me!)")
        #st.button(
        #    "Run Traffic Estimation (Click Me!)",
        #    key="run_york_colorize",
        #   on_click=lambda: st.session_state.update(colorized_york=True)
        #)

    # --- Load GeoPackage layer ---
    gdf = gpd.read_file("York_roads_within_3km.gpkg")
    
    import numpy as np
    def get_speed_color(pred):
        try:
            pred = float(pred)
        except:
            return "gray"
        if pred < 10:
            return "#8B0000"   # dark red
        elif pred < 20:
            return "#FF0000"   # red
        elif pred < 30:
            return "#FFA500"   # orange
        elif pred < 40:
            return "#FFFF00"   # yellow
        elif pred < 50:
            return "#9ACD32"   # light green
        else:
            return "#00B050"   # green

    # create random "speed" values once when button pressed
    if st.session_state["colorized_york"] and st.session_state["york_speeds"] is None:
        st.session_state["york_speeds"] = {
            i: float(10 + 20 * np.random.rand()) for i in range(len(gdf))
        }

    # --- Map setup ---
    map_center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    m = folium.Map(location=map_center, zoom_start=13)

    # --- Draw each segment ---
    for idx, row in gdf.iterrows():
        if st.session_state["colorized_york"] and st.session_state["york_speeds"] is not None:
            speed = st.session_state["york_speeds"][idx]
            color = get_speed_color(speed)
        else:
            color = "black"  # default color before pressing the button

        tooltip = "<br>".join([f"<b>{col}:</b> {row[col]}" for col in list(gdf.columns)[:3]])
        folium.GeoJson(
            row.geometry.__geo_interface__,
            tooltip=tooltip,
            style_function=lambda x, color=color: {"color": color, "weight": 2.5}
        ).add_to(m)

    # --- Map and color legend side-by-side ---
    col1, col2 = st.columns([4, 1])
    with col1:
        st_folium(m, width=700, height=500)
    with col2:
        st.markdown("""
        <div style='font-weight:bold;margin-bottom:8px;'>Prediction Color Key</div>
        <div style='line-height:2;'>
            <span style="display:inline-block;width:22px;height:18px;background:#8B0000;border-radius:4px;margin-right:8px;"></span> 0–10
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#FF0000;border-radius:4px;margin-right:8px;"></span> 10–20
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#FFA500;border-radius:4px;margin-right:8px;"></span> 20–30
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#FFFF00;border-radius:4px;margin-right:8px;"></span> 30–40
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#9ACD32;border-radius:4px;margin-right:8px;"></span> 40–50
            <br>
            <span style="display:inline-block;width:22px;height:18px;background:#00B050;border-radius:4px;margin-right:8px;"></span> 50+
        </div>
        """, unsafe_allow_html=True)
