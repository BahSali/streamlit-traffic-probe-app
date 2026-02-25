import streamlit as st

def inject_styles():
    dark_style = """
    <style>
    /* Main background */
    .stApp {
        background-color: #0F172A;
        color: #E5E7EB;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #E5E7EB !important;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #009688;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6em 1.4em;
        border: none;
        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:first-child:hover {
        background-color: #00796B;
        transform: translateY(-1px);
    }

    /* Checkbox */
    .stCheckbox label {
        color: #E5E7EB;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #E5E7EB;
        font-weight: 500;
    }

    /* Dataframe */
    .stDataFrame {
        background-color: #1E293B;
        border-radius: 6px;
    }
    </style>
    """
    st.markdown(dark_style, unsafe_allow_html=True)
