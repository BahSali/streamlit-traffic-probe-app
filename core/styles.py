import streamlit as st

def inject_styles():
    style = """
    <style>

    /* Hide Streamlit top bar */
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0px;
    }

    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0;
    }

    /* Remove empty space on top */
    .block-container {
        padding-top: 1.5rem;
    }

    /* Main background */
    .stApp {
        background-color: #F4F6F9;
        color: #1F2937;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #E9EEF3;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #00796B;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6em 1.4em;
        border: none;
    }

    div.stButton > button:first-child:hover {
        background-color: #005F56;
    }

    </style>
    """
    st.markdown(style, unsafe_allow_html=True)
