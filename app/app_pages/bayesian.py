#from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

import streamlit as st
from pathlib import Path

import pandas as pd




def display_html_plot(plot_path, height=550):
    if not Path(plot_path).exists():
        st.warning(f"Plot not found: {plot_path}")
        return
    with open(plot_path, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=height, scrolling=False)

def bayesian_page(plot_dir):
    st.title("Bayesian Climate Trend Analysis")
    st.markdown("This analysis uses Bayesian inference to estimate the long-term temperature trend for both annual average and annual extreme temperatures, while accounting for uncertainty in the estimated trend.")

    trend_path = Path(plot_dir) / "warming_trend_per_decade.csv"
    if not trend_path.exists():
        st.warning("Bayesian trend results not found.")
        return

    trend_df = pd.read_csv(trend_path)

    avg_trend = trend_df.loc[0, "Average_Trend"]
    avg_trend_minus = trend_df.loc[0, "Average_Minus_Error"]
    avg_trend_plus = trend_df.loc[0, "Average_Plus_Error"]
    ext_trend = trend_df.loc[0, "Extreme_Trend"]
    ext_trend_minus = trend_df.loc[0, "Extreme_Minus_Error"]
    ext_trend_plus = trend_df.loc[0, "Extreme_Plus_Error"]

    # ==========================================================
    # Trend Estimates
    # ==========================================================
    st.subheader("Trend Estimates")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="background-color:#F8FAFC;padding:24px;border-radius:14px;border:1px solid #E5E7EB;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <div style="font-size:18px;font-weight:600;">🌡️ Annual Average Temperature</div>
        <div style="font-size:14px;color:#6B7280;margin-top:6px;">Mean temperature across all months of each year.</div>
        <div style="font-size:32px;font-weight:700;margin-top:16px;">{avg_trend:.2f} °C / decade</div>
        <div style="color:#6B7280;margin-top:6px;">Bayesian trend estimate</div>
        <div style="color:#6B7280;margin-top:4px;">−{avg_trend_minus:.2f} / +{avg_trend_plus:.2f} °C / decade uncertainty</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color:#F8FAFC;padding:24px;border-radius:14px;border:1px solid #E5E7EB;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <div style="font-size:18px;font-weight:600;">🔥 Annual Extreme Temperature</div>
        <div style="font-size:14px;color:#6B7280;margin-top:6px;">Maximum temperature recorded during each year.</div>
        <div style="font-size:32px;font-weight:700;margin-top:16px;">{ext_trend:.2f} °C / decade</div>
        <div style="color:#6B7280;margin-top:6px;">Bayesian trend estimate</div>
        <div style="color:#6B7280;margin-top:4px;">−{ext_trend_minus:.2f} / +{ext_trend_plus:.2f} °C / decade uncertainty</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==========================================================
    # Bayesian Model
    # ==========================================================
    st.header("Bayesian Model")
    st.write("The long-term temperature trend is estimated using an autoregressive Bayesian model applied to the annual temperature series.")

    model_col1, model_col2 = st.columns(2)

    with model_col1:
        st.markdown("""
        **Model**

        - **Model used:** AR(1) model
        - **Input:** Annual temperature series
        - **Analyses:** Annual average and annual maximum temperature
        - **Seasonal features:** Removed before modelling
        """)

    with model_col2:
        st.markdown("""
        **Bayesian Inference**

        - **Residual structure:** First-order autocorrelation
        - **Inference:** Bayesian posterior estimation
        - **Parameter of interest:** Long-term temperature trend
        - **Output:** Trend estimate and uncertainty
        """)

    st.divider()

    # ==========================================================
    # Posterior Distributions
    # ==========================================================
    st.header("Posterior Distributions")
    st.write("The posterior distributions show the range of plausible values for the estimated temperature trend.")

    avg_plot = Path(plot_dir) / "ModelA_average_tempcorener_plot.png"
    ext_plot = Path(plot_dir) / "ModelA_max_tempcorener_plot.png"

    if avg_plot.exists() and ext_plot.exists():
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.subheader("Annual Average Temperature")
            st.image(str(avg_plot), use_container_width=True)

        with plot_col2:
            st.subheader("Annual Extreme Temperature")
            st.image(str(ext_plot), use_container_width=True)
    else:
        st.info("No Bayesian posterior visualization files were found.")
            