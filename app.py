from STIB_dataset_generator import main as generate_dataset
from fusion_use import main as run_model
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import subprocess
import os

df = pd.read_csv("Brux_net.csv", sep=';')
df['id'] = df['id'].astype(str)

st.title("Real-time Average Speed Estimator")

with st.expander("Show raw data"):
    st.dataframe(df)

last_four_cols = df.columns[-4:]
start_lat_col, start_lon_col, end_lat_col, end_lon_col = last_four_cols

map_center = [
    df[start_lat_col].mean(),
    df[start_lon_col].mean()
]

m = folium.Map(location=map_center, zoom_start=13)

# --- Session State ---
if "results_dict" not in st.session_state:
    st.session_state["results_dict"] = {}

# --- Button & Processing ---
col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.markdown("""
    <style>
    .custom-button {
        background-color: #007bff;
        color: white;
        padding: 0.75em 2em;
        font-size: 1.1em;
        border-radius: 2em;
        border: none;
        cursor: pointer;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: background 0.3s;
    }
    .custom-button:hover {
        background-color: #0056b3;
    }
    </style>
    <form action="" method="post">
        <button class="custom-button" type="submit">🚦 Retrieve Live Bus Speeds & Predict Traffic</button>
    </form>
""", unsafe_allow_html=True)

    if st.button("Run Traffic Estimation (Click Me!)"):
        with st.spinner('⏳ Generating latest dataset (please wait)...'):
            generate_dataset()
        st.success('✅ Dataset generated!')

        with st.spinner('🤖 Running prediction model...'):
            run_model()
        st.success('✅ Prediction finished!')

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

# --- Draw edges and tooltips ---
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

for idx, row in df.iterrows():
    street_name = f"{row[df.columns[1]]} - {row[df.columns[2]]}"
    segment_id = str(row['id'])
    tooltip_text = f"ID: {segment_id}<br>Street: {street_name}"

    color = "black"
    if results_dict and segment_id in results_dict:
        result_row = results_dict[segment_id]
        speed = result_row.get('Speed', 'N/A')
        prediction = result_row.get('Prediction', 'N/A')
        tooltip_text += f"<br>Speed: {speed}<br>Prediction: {prediction}"
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


# st.subheader("Brussels Case Study")
# st_folium(m, width=700, height=500)
col1, col2 = st.columns([4, 1])
with col1:
    st_folium(m, width=700, height=500)
with col2:
    st.markdown("""
    <span style="display:inline-block;width:20px;height:20px;background:#8B0000"></span> 0-10<br>
    <span style="display:inline-block;width:20px;height:20px;background:#FF0000"></span> 10-20<br>
    <span style="display:inline-block;width:20px;height:20px;background:#FFA500"></span> 20-30<br>
    <span style="display:inline-block;width:20px;height:20px;background:#FFFF00"></span> 30-40<br>
    <span style="display:inline-block;width:20px;height:20px;background:#9ACD32"></span> 40-50<br>
    <span style="display:inline-block;width:20px;height:20px;background:#00B050"></span> 50-60<br>
    """, unsafe_allow_html=True)
