import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

#sys.path.append("app_pages/")
APP_PAGES = Path(__file__).resolve().parent / "app_pages"
sys.path.append(str(APP_PAGES))
import eda
import prediction as pred
import bayesian

# ==========================================================
# Page configuration
# ==========================================================
st.set_page_config(page_title="Climate Trends Dashboard", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")


# ==========================================================
# Load custom CSS
# ==========================================================
css_path = Path("app/styles.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================================================
# Sidebar
# ==========================================================
with st.sidebar:
    st.title("ERA5 Temperature Explorer")
    city = st.selectbox("Select City", ["Paris", "Berlin", "Kolkata", "Hanoi"])
    st.divider()
    st.markdown("## Contents")
    page = st.radio("Go to", ["Overview", "Bayesian Analysis", "Next Month Prediction"], label_visibility="collapsed")

# ==========================================================
# Output directory
# ==========================================================
#plot_dir = Path("../output") / city
plot_dir = Path(__file__).resolve().parent.parent / "output" / city

# ==========================================================
# Interactive temperature explorer
# ==========================================================
def interactive_temperature_explorer(plot_dir):
    dataset_path = plot_dir / "dataset.csv"
    if not dataset_path.exists():
        st.warning(f"Dataset not found: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    st.write("Explore the monthly temperature record interactively.")
    view = st.radio("View", ["Single Year", "Compare Years", "Full History"], horizontal=True)

    if view == "Single Year":
        years = sorted(df["year"].unique(), reverse=True)
        year = st.selectbox("Select year", years)
        year_data = df[df["year"] == year].copy()
        fig = px.line(year_data, x="date", y="actual", markers=True, title=f"Monthly Temperature — {year}")
        fig.update_layout(xaxis_title="Month", yaxis_title="Temperature (°C)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Compare Years":
        years = sorted(df["year"].unique())
        default_years = years[-3:] if len(years) >= 3 else years
        selected_years = st.multiselect("Select years to compare", years, default=default_years)

        if selected_years:
            compare_df = df[df["year"].isin(selected_years)].copy()
            compare_df["month_name"] = compare_df["date"].dt.strftime("%b")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            fig = px.line(compare_df, x="month_name", y="actual", color="year", markers=True, category_orders={"month_name": month_order}, title="Monthly Temperature Comparison")
            fig.update_layout(xaxis_title="Month", yaxis_title="Temperature (°C)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one year.")

    else:
        fig = px.line(df, x="date", y="actual", title="Full Historical Temperature Record")
        fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (°C)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# OVERVIEW
# ==========================================================
if page == "Overview":

    icon_col, title_col = st.columns([1, 12])
    with icon_col:
        st.markdown("<h1 style='text-align:center;'>🌍</h1>", unsafe_allow_html=True)
    with title_col:
        st.title("Analysis on ERA5 Temperature Data")

    st.markdown("This dashboard presents an exploratory analysis and forecasting of **ERA5 monthly 2 m air temperature (t2m)** data. It investigates long-term temperature behaviour, temporal variability, and next-month temperature prediction.")

    # ------------------------------------------------------
    # Key cards
    # ------------------------------------------------------
    trend_path = plot_dir / "warming_trend_per_decade.csv"
    forecast_path = plot_dir / "forecast_next_month.csv"

    trend_value = None
    forecast_value = None
    forecast_month = None

    if trend_path.exists():
        trend_df = pd.read_csv(trend_path)
        trend_value = trend_df.loc[0, "Average_Trend"]

    if forecast_path.exists():
        forecast_df = pd.read_csv(forecast_path)
        if "prediction" in forecast_df.columns:
            forecast_value = forecast_df["prediction"].iloc[0]
        elif "Climatology" in forecast_df.columns:
            forecast_value = forecast_df["Climatology"].iloc[0]
        if "forecast_month" in forecast_df.columns:
            forecast_month = str(forecast_df["forecast_month"].iloc[0])

    card1, card2 = st.columns(2)

    with card1:
        trend_text = f"{trend_value:.2f} °C / decade" if trend_value is not None else "Not available"
        st.markdown(f"""
        <div style="background-color:#F8FAFC;padding:24px;border-radius:14px;border:1px solid #E5E7EB;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <div style="font-size:18px;font-weight:600;">🌡️ Warming Trend</div>
        <div style="font-size:32px;font-weight:700;margin-top:8px;">{trend_text}</div>
        <div style="color:#6B7280;margin-top:6px;">Bayesian estimate of long-term temperature change</div>
        </div>
        """, unsafe_allow_html=True)

    with card2:
        temp_text = f"{forecast_value:.2f} °C" if forecast_value is not None else "Not available"
        month_text = f"Forecast for {forecast_month}" if forecast_month else "Next month forecast"
        st.markdown(f"""
        <div style="background-color:#F8FAFC;padding:24px;border-radius:14px;border:1px solid #E5E7EB;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <div style="font-size:18px;font-weight:600;">🔮 Next Month Temperature</div>
        <div style="font-size:32px;font-weight:700;margin-top:8px;">{temp_text}</div>
        <div style="color:#6B7280;margin-top:6px;">{month_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # Data Specifications
    # ------------------------------------------------------
    st.header("Data Specifications")

    data_info_path = plot_dir / "data_info.csv"

    if data_info_path.exists():
        data_info = pd.read_csv(data_info_path)
        info = dict(zip(data_info["Specification"], data_info["Value"]))

        st.markdown(f"""
        <div style="background-color:#F8FAFC;padding:20px 24px;border-radius:12px;border:1px solid #E5E7EB;">
        <b>Dataset</b> : ERA5 Monthly Reanalysis (2 m Temperature)<br>
        <b>City</b> : {city}<br>
        <b>Variable</b> : {info.get("Variable", "2 m Air Temperature (t2m)")}<br>
        <b>Latitude</b> : {info.get("Latitude", "N/A")}<br>
        <b>Longitude</b> : {info.get("Longitude", "N/A")}<br>
        <b>Date Start</b> : {info.get("Data Start", "N/A")}<br>
        <b>Date End</b> : {info.get("Data End", "N/A")}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Data specification file not found.")

    st.divider()

    # ------------------------------------------------------
    # Explore the Data
    # ------------------------------------------------------
    st.header("Explore the Data")
    st.write("Explore the monthly temperature record interactively.")
    interactive_temperature_explorer(plot_dir)

    st.divider()

    # ------------------------------------------------------
    # Explore Trends
    # ------------------------------------------------------
    st.header("Explore Trends")
    st.write("Explore the long-term behaviour and temporal structure of temperature.")

    trend_view = st.radio("Select analysis", ["Annual Temperature Trend", "Temperature Anomaly", "STL Decomposition"], horizontal=True, label_visibility="collapsed")

    if trend_view == "Annual Temperature Trend":
        eda.display_html_plot(plot_dir / "yearly_average.html", height=550)
        st.caption("Comparison of annual and 5-year averaged temperatures highlighting long-term warming.")

    elif trend_view == "Temperature Anomaly":
        eda.display_html_plot(plot_dir / "anomaly_yearly.html", height=550)
        st.caption("Temperature anomalies relative to the 1950–1980 climatological baseline.")

    elif trend_view == "STL Decomposition":
        eda.display_html_plot(plot_dir / "stl_decomposition.html", height=900)
        st.caption("Seasonal-Trend decomposition separates the temperature signal into trend, seasonal and residual components.")

# ==========================================================
# BAYESIAN ANALYSIS
# ==========================================================
elif page == "Bayesian Analysis":
    bayesian.bayesian_page(plot_dir)


# ==========================================================
# NEXT MONTH PREDICTION
# ==========================================================

elif page == "Next Month Prediction":
    pred.prediction_page(plot_dir)

# ==========================================================
# About
# ==========================================================
st.divider()
st.header("About")
st.write("""
For more information, see the
[Climate-trends-Bayesian GitHub repository](https://github.com/aparajita-web/Climate-trends-Bayesian).
""")

