import sys
from pathlib import Path
import streamlit as st
import pandas as pd
sys.path.append("app/")

import eda
import prediction as pred

# ==========================================================
# Page configuration
# ==========================================================

st.set_page_config(
    page_title="Climate Trends Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# Load custom CSS
# ==========================================================

with open("app/styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    #st.image("streamlit_icons/gw.jpeg", width=120)

    st.title("ERA5 Temperature Explorer")

    city = st.selectbox(
        "Select City",
        [
            "Paris",
            "Berlin",
            "Kolkata",
            "Hanoi"
        ],
    )

    st.divider()

    st.markdown("## Contents")

    st.markdown("""
- Overview
- Exploratory Data Analysis
- Climate Trend
- Forecasting
- Model Comparison
- About
""")

# ==========================================================
# Directory containing plots
# ==========================================================

plot_dir = Path("output") / city

# ==========================================================
# OVERVIEW
# ==========================================================

icon_col, title_col = st.columns([1, 12])

with icon_col:
    st.markdown(
        "<h1 style='text-align:center;'>🌍</h1>",
        unsafe_allow_html=True,
    )

with title_col:
    st.title("Analysis on ERA5 Temperature Data")

st.markdown(
    """
This dashboard presents an exploratory analysis and forecasting of
**ERA5 monthly 2 m air temperature (t2m)** data.

It investigates long-term warming trends,
explores the temporal variability of temperature,
and compares statistical and machine learning models
for next-month temperature prediction.
"""
)

st.markdown(
    """
<div style="
background-color:#F8FAFC;
padding:18px;
border-radius:12px;
border-left:6px solid #14B8A6;
margin-top:15px;
margin-bottom:30px;
">

<b>Dataset</b> : ERA5 Monthly Reanalysis (2 m Temperature)<br>

<b>Analysis Period</b> : January 1950 – December 2025<br>

<b>Temporal Resolution</b> : Monthly

</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# ==========================================================
# Exploratory Data Analysis
# ==========================================================

eda.eda_page(plot_dir)

st.divider()

# ==========================================================
# Climate Trend
# ==========================================================

st.header("Climate Trend")

st.write(
    """
The long term warming trend estimate by Bayesian Inference
"""
)

trend_df = pd.read_csv(
    Path(plot_dir) / "warming_trend_per_decade.csv"
)

avg_trend = trend_df.loc[0, "Average_Trend"]
avg_trend_minus = trend_df.loc[0, "Average_Minus_Error"]
avg_trend_plus = trend_df.loc[0, "Average_Plus_Error"]

ext_trend = trend_df.loc[0, "Extreme_Trend"]
ext_trend_minus = trend_df.loc[0, "Extreme_Minus_Error"]
ext_trend_plus = trend_df.loc[0, "Extreme_Plus_Error"]

pred.warming_trend_card(
    avg_trend=avg_trend,
    avg_err=avg_trend_plus,
    extreme_trend=ext_trend,
    extreme_err=ext_trend_plus,
)

st.divider()

forecast_df = pd.read_csv(plot_dir / "forecast_next_month.csv")
pred.prediction_page(forecast_df)

st.divider()

# ==========================================================
# Model Comparison
# ==========================================================

st.header("Model Comparison")

st.write(
    """
Comparing the performance of each model using the evaluation metrics
"""
)

pred.evaluation_page(plot_dir)
st.divider()

# ==========================================================
# About
# ==========================================================

st.header("About")

st.write(
    """
For more see Github: https://github.com/aparajita-web/Climate-trends-Bayesian
"""
)