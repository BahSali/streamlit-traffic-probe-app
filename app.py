import streamlit as st
from core.styles import inject_styles

st.set_page_config(page_title="App", layout="wide")
inject_styles()

st.title("App is running")
st.write("If you see this, the deployment is healthy.")
