from STIB_dataset_generator import main as generate_dataset
from fusion_use import main as run_model
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

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
    "<h1 style='text-align:center; color:#009688;'> Real-time Average Speed Estimator</h1>",
    unsafe_allow_html=True
)
st.caption("Visualizing real-time speeds and predictions for the bus network.")

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

        with st.spinner('📦 Loading prediction results...'):
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
    tooltip_text = f"ID: {segment_id}<br>Street: {street_name}"

    color = "black"
    if results_dict and segment_id in results_dict:
        result_row = results_dict[segment_id]
        speed = result_row.get('Speed', 'N/A')
        prediction = result_row.get('Prediction', 'N/A')
        # tooltip_text += f"<br>Speed: {speed}<br>Prediction: {prediction}"
        tooltip_text += f"<br>Speed: {float(speed):.2f}<br>Prediction: {float(prediction):.2f}"
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
