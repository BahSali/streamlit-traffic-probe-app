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

    /* Performance Analysis summary (cities/brussels/charts.py): smaller values, labels that wrap. */
    div[class*="st-key-bru_summary"] [data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    div[class*="st-key-bru_summary"] [data-testid="stMetricLabel"] p {
        font-size: 0.85rem;
        white-space: normal;
    }

    /* The speed legend (core.colors.legend_html) stays to the right of the map
       whenever the screen is wider than a phone: its column keeps the width of
       its labels and the map column shrinks instead of the row wrapping. */
    @media (min-width: 641px) {
        div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"] .speed-legend) {
            flex-wrap: nowrap;
        }
        div[data-testid="stColumn"]:has(.speed-legend) {
            min-width: 5.5rem;
        }
        div[data-testid="stColumn"]:not(:has(.speed-legend)):has(+ div[data-testid="stColumn"] .speed-legend) {
            min-width: 0;
        }
    }

    </style>
    """
    st.markdown(style, unsafe_allow_html=True)
