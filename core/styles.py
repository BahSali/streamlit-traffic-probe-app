import streamlit as st

def inject_styles():
    light_style = """
    <style>

    /* Main background */
    .stApp {
        background-color: #F4F6F9;
        color: #1F2937;
    }

    /* Remove top padding gap */
    .block-container {
        padding-top: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #E9EEF3;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #1F2937 !important;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #00796B;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6em 1.4em;
        border: none;
        transition: all 0.15s ease-in-out;
    }

    div.stButton > button:first-child:hover {
        background-color: #005F56;
        transform: translateY(-1px);
    }

    /* Checkbox */
    .stCheckbox label {
        color: #1F2937;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #1F2937;
    }

    /* Dataframe */
    .stDataFrame {
        background-color: white;
        border-radius: 8px;
    }

    </style>
    """
    st.markdown(light_style, unsafe_allow_html=True)
