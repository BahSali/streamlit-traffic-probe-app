import streamlit as st

def inject_styles():
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
