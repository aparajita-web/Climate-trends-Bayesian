import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parent
APP_PAGES = BASE_DIR / "app" / "app_pages"
APP_COMPONENTS = BASE_DIR / "app" / "app_components"
OUTPUT_DIR = BASE_DIR / "output"

sys.path.insert(0, str(APP_PAGES))
sys.path.insert(0, str(APP_COMPONENTS))

from styles import load_css

import prediction
import analysis
import about


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Decadal",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css()


# ---------------------------------------------------------
# Available cities
# ---------------------------------------------------------

CITIES = [
    "Paris",
    "Berlin",
    "Kolkata",
    "Hanoi",
]


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "decadal_page" not in st.session_state:
    st.session_state["decadal_page"] = "Predict"

if "decadal_dark" not in st.session_state:
    st.session_state["decadal_dark"] = False

if "predict_city" not in st.session_state:
    st.session_state["predict_city"] = "Paris"









# ---------------------------------------------------------
# Top navigation styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .decadal-brand {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #F4F7FA;
        padding-top: 5px;
    }

    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;

        color: #030506 !important;

        font-family: 'Geist', sans-serif !important;
        font-size: 20px !important;
        font-weight: 600 !important;

        padding: 6px 0 10px 0 !important;
        min-height: 0 !important;
    }

    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        background: transparent !important;
        color: #030506 !important;
    }

    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:focus {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# Navigation columns
nav_brand, nav_predict, nav_analysis, nav_about = st.columns(
    [2.3, 1.1, 1.1, 1.5],
    gap="small",
)


# Decadal brand
with nav_brand:
    st.markdown(
        '<div class="decadal-brand">Decadal</div>',
        unsafe_allow_html=True,
    )

def nav_button(column, label, target):
    with column:

        active = st.session_state["decadal_page"] == target

        if active:
            st.markdown(
                '<span class="nav-active"></span>',
                unsafe_allow_html=True,
            )

        if st.button(
            label,
            key=f"nav_{target}",
            use_container_width=False,
        ):
            st.session_state["decadal_page"] = target
            st.rerun()



# Navigation items
nav_button(nav_predict, "PREDICT", "Predict")
nav_button(nav_analysis, "ANALYSIS", "Analysis")
nav_button(nav_about, "ABOUT THE BUILD", "About the build")


#st.divider()



# ---------------------------------------------------------
# Current page
# ---------------------------------------------------------

page = st.session_state["decadal_page"]


# ---------------------------------------------------------
# Current city
# ---------------------------------------------------------

city = st.session_state.get(
    "predict_city",
    "Paris",
)

if city not in CITIES:
    city = "Paris"

plot_dir = OUTPUT_DIR / city


# ---------------------------------------------------------
# Route pages
# ---------------------------------------------------------

if page == "Predict":
    prediction.prediction_page(plot_dir)

elif page == "Analysis":
    analysis.analysis_page(plot_dir)

elif page == "About the build":
    about.about_page(plot_dir)