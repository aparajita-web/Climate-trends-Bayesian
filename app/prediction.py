#from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

import streamlit as st


import pandas as pd


def prediction_page(forecast_df):

    st.header("Temperature Prediction for January 2026")

    st.write(
        """
        Forecast of the average monthly temperature using three different
        forecasting approaches.
        """
    )

    climatology = forecast_df["Climatology"].iloc[0]
    gradient_boost = forecast_df["Gradient_Boost"].iloc[0]
    sarimax = forecast_df["SARIMA"].iloc[0]

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:

        st.markdown(
            f"""
            <div style="
                background:#E0F2F1;
                padding:25px;
                border-radius:15px;
                text-align:center;
                height:230px;
            ">

            <h4 style="color:#0F766E;">Baseline</h4>

            <h3>Climatology</h3>

            <p style="
                font-size:38px;
                font-weight:bold;
                color:#1F2937;
                margin-top:30px;
            ">
                {climatology:.2f} °C
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div style="
                background:#E8F1FD;
                padding:25px;
                border-radius:15px;
                text-align:center;
                height:230px;
            ">

            <h4 style="color:#2563EB;">Machine Learning</h4>

            <h3>Gradient Boost</h3>

            <p style="
                font-size:38px;
                font-weight:bold;
                color:#1F2937;
                margin-top:30px;
            ">
                {gradient_boost:.2f} °C
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div style="
                background:#FEF3C7;
                padding:25px;
                border-radius:15px;
                text-align:center;
                height:230px;
            ">

            <h4 style="color:#B45309;">State-space</h4>

            <h3>SARIMAX</h3>

            <p style="
                font-size:38px;
                font-weight:bold;
                color:#1F2937;
                margin-top:30px;
            ">
                {sarimax:.2f} °C
            </p>

            </div>
            """,
            unsafe_allow_html=True,
        )
        
        
        
######### Make the table for Evaluation metrics ##########


from numpy.random import default_rng as rng



def evaluation_page(plot_dir):
    """
    Display model evaluation metrics.

    Parameters
    ----------
    plot_dir : pathlib.Path
        Directory containing the evaluation CSV files.
    """



    # --------------------------------------------------------
    # Read CSV files
    # --------------------------------------------------------

    climatology_df = pd.read_csv(
        plot_dir / "climatology_metrics.csv"
    )

    gb_df = pd.read_csv(
        plot_dir / "GradientBoost_metrics.csv"
    )

    sarima_df = pd.read_csv(
        plot_dir / "SARIMA_metrics.csv"
    )

    # --------------------------------------------------------
    # Helper function
    # --------------------------------------------------------

    def get_metric(df, metric):

        return df.loc[
            df["Metric"] == metric,
            "Value"
        ].iloc[0]

    # --------------------------------------------------------
    # Create comparison table
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Climatology",
                "Gradient Boost",
                "SARIMAX",
            ],

            "R²": [
                get_metric(climatology_df, "R2"),
                get_metric(gb_df, "R2"),
                get_metric(sarima_df, "R2"),
            ],

            "RMSE (°C)": [
                get_metric(climatology_df, "RMSE"),
                get_metric(gb_df, "RMSE"),
                get_metric(sarima_df, "RMSE"),
            ],

            "MAE (°C)": [
                get_metric(climatology_df, "MAE"),
                get_metric(gb_df, "MAE"),
                get_metric(sarima_df, "MAE"),
            ],
        }
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True,
        column_config={

            "Model": st.column_config.TextColumn(
                "Model",
                width="medium",
            ),

            "R²": st.column_config.NumberColumn(
                "R²",
                format="%.3f",
            ),

            "RMSE (°C)": st.column_config.NumberColumn(
                "RMSE",
                format="%.3f °C",
            ),

            "MAE (°C)": st.column_config.NumberColumn(
                "MAE",
                format="%.3f °C",
            ),
        },
    )


def warming_trend_card(avg_trend,avg_err,extreme_trend,extreme_err):

    with st.container(border=True):

        st.markdown(
            "<h3 style='text-align:center;'>🌍 Warming Trend per Decade</h3>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <h1 style='text-align:center;
                       color:#0F766E;
                       margin-bottom:0px;'>
                {avg_trend:.2f} ± {avg_err:.2f} °C / decade
            </h1>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='text-align:center;color:gray;'>Annual Mean Temperature</p>",
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            f"""
            <h2 style='text-align:center;
                       color:#B45309;
                       margin-bottom:0px;'>
                {extreme_trend:.2f} ± {extreme_err:.2f} °C / decade
            </h2>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='text-align:center;color:gray;'>Annual Maximum Temperature</p>",
            unsafe_allow_html=True,
        )
