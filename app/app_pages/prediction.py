import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent.parent
APP_COMPONENTS = APP_DIR / "app_components"
sys.path.insert(0, str(APP_COMPONENTS))

from styles import MODEL_COLORS
from styles import load_css



MODELS = {
    "sarima": {
        "name": "SARIMA",
        "api": "sarima",
        "help": "Seasonal ARIMA on the monthly series (statsmodels).",
        "forecast": "SARIMA_forecast.csv",
        "metrics": "SARIMA_metrics.csv",
        "color": MODEL_COLORS["sarima"],
    },
    "gb": {
        "name": "Gradient Boosting",
        "api": "gradient_boosting",
        "help": "HistGradientBoostingRegressor using calendar and seasonal features.",
        "forecast": "gradient_boost_forecast.csv",
        "metrics": "GradientBoost_metrics.csv",
        "color": MODEL_COLORS["gb"],
    },
    "clim": {
        "name": "Climatology",
        "api": "climatology",
        "help": "A baseline based on the historical August climatology.",
        "forecast": "climatology_forecast.csv",
        "metrics": "climatology_metrics.csv",
        "color": MODEL_COLORS["clim"],
    },
}


def prediction_page(plot_dir):
    load_css()

    cities = ["Paris", "Berlin", "Kolkata", "Hanoi"]

    left, right = st.columns([6, 6], gap="large")

    with left:

        st.title("Next month's temperature, and how sure we are.")

        st.write(
            "Pick a city. Decadal forecasts the coming month's mean "
            "2 m air temperature from 76 years of ERA5 data, interval included."
        )

        city = st.segmented_control(
            "City",
            cities,
            default=st.session_state.get("predict_city", "Paris"),
            selection_mode="single",
            width="stretch",
            key="city_selector"
        )

        if city is None:
            city = "Paris"

        city_dir = plot_dir.parent / city

        model_options = list(MODELS.keys())

        model_key = st.segmented_control(
            "Model",
            model_options,
            default="sarima",
            format_func=lambda x: MODELS[x]["name"],
            selection_mode="single",
            width="stretch",)
            

        if model_key is None:
            model_key = "sarima"

        model = MODELS[model_key]

        st.caption(model["help"])

        #st.code(
         #   f"GET /api/forecast?city={city.lower()}&model={model['api']}"
        #)

    with right:
        with st.container(border=True,key="forecast_panel"):
            dataset_path = city_dir / "dataset.csv"

            if not dataset_path.exists():
                st.error(f"Dataset not found:\n{dataset_path}")
                return

            history = pd.read_csv(dataset_path)
            history["date"] = pd.to_datetime(history["date"])
            history["actual"] = pd.to_numeric(history["actual"], errors="coerce")
            history = history.dropna(subset=["date", "actual"]).sort_values("date")

            forecast_path = city_dir / model["forecast"]

            if not forecast_path.exists():
                st.error(f"Forecast file not found:\n{forecast_path}")
                return

            forecast = pd.read_csv(forecast_path)

            metrics_path = city_dir / model["metrics"]

            if not metrics_path.exists():
                st.error(f"Metrics file not found:\n{metrics_path}")
                return

            metrics = pd.read_csv(metrics_path)

            trend_path = city_dir / "warming_trend_per_decade.csv"

            if not trend_path.exists():
                st.error(f"Warming trend file not found:\n{trend_path}")
                return

            trend = pd.read_csv(trend_path)

            forecast_date = history["date"].iloc[-1] + pd.offsets.MonthBegin(1)
            forecast_value = float(forecast["prediction"].iloc[0])
            lower = float(forecast["lower"].iloc[0])
            upper = float(forecast["upper"].iloc[0])
            forecast_month = forecast_date.strftime("%B %Y")

            mae_rows = metrics[metrics["Metric"].astype(str).str.upper() == "MAE"]
            mae = float(mae_rows["Value"].iloc[0]) if not mae_rows.empty else float("nan")

            trend_row = trend.iloc[0]
            trend_value = float(trend_row["Average_Trend"])
            trend_lower = trend_value - float(trend_row["Average_Minus_Error"])
            trend_upper = trend_value + float(trend_row["Average_Plus_Error"])


            trend_value = (trend["Average_Trend"]).iloc[0]
            trend_minus = (trend["Average_Minus_Error"]).iloc[0]
            trend_plus = (trend["Average_Plus_Error"]).iloc[0]

            august_2025 = history[
                (history["date"].dt.year == 2025)
                & (history["date"].dt.month == 8)
                    ]

            observed_2025 = float(august_2025["actual"].iloc[0]) if not august_2025.empty else float("nan")

            august_normal = history[
                (history["date"].dt.year.between(1991, 2020))
                & (history["date"].dt.month == 8)
            ]

            normal = float(august_normal["actual"].mean()) if not august_normal.empty else float("nan")

            model_color = model["color"]

            #st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
            

            top_left, top_right = st.columns([3, 1])

            with top_left:
                st.markdown(f'<div class="forecast-month">{forecast_month}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="forecast-where">{city}, the month after the record ends</div>', unsafe_allow_html=True)

            with top_right:
                st.markdown(
                    f'<div style="text-align:right;"><span style="color:{model_color};">●</span> {model["name"]}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="forecast-value">{forecast_value:.1f}<span class="forecast-unit">°C</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="forecast-interval">{lower:.1f} to {upper:.1f} °C, {model["name"]} prediction interval</div>',
                unsafe_allow_html=True,
            )

            hist = history.tail(24)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=hist["date"],
                    y=hist["actual"],
                    mode="lines",
                    name="Observed",
                    line=dict(width=2.5, color="#F6F3EF"),
                    hovertemplate="%{x|%b %Y}<br>%{y:.1f} °C observed<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[forecast_date, forecast_date],
                    y=[lower, upper],
                    mode="lines",
                    name="Prediction interval",
                    line=dict(width=2.5, color=model_color),
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[forecast_date],
                    y=[forecast_value],
                    mode="markers+text",
                    name="Forecast",
                    marker=dict(size=11, color=model_color),
                    text=[f"{forecast_value:.1f} °C"],
                    textposition="middle right",
                    textfont=dict(size=12, color="#F4F7FA"),
                    hovertemplate=f"<b>{forecast_month}</b><br>{forecast_value:.1f} °C forecast<extra></extra>",
                )
            )

            fig.add_vrect(
                x0=hist["date"].iloc[-1],
                x1=forecast_date + pd.Timedelta(days=20),
                fillcolor="#E6DEE4",
                opacity=0.12,
                line_width=0,
                layer="below",
            )

            fig.update_layout(
                height=214,
                width=560,
                margin=dict(l=46, r=66, t=18, b=28),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(family="Geist Mono, monospace", size=11, color="#EEF2FB"),
                xaxis=dict(showgrid=False, showline=True, linecolor="#E9ECF3", fixedrange=True, tickformat="%b %Y"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.10)", zeroline=False, fixedrange=True, title="°C"),
            )

            st.markdown('<div class="forecast-chart">', unsafe_allow_html=True)
            st.plotly_chart(fig, width=560)

            st.markdown(
                f'<div class="forecast-legend"><span><i class="legend-line"></i>Observed, ERA5, last 24 months</span><span><i class="legend-dot" style="background:{model_color};"></i>Forecast with its interval</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "August 2025, observed",
                    f"{observed_2025:.1f} °C"
                )

            with col2:
                st.metric(
                    "August normal, 1991–2020",
                    f"{normal:.1f} °C"
                )
            #st.markdown("---")
            st.metric(
                "Model error on held-out months",
                f"MAE {mae:.2f} °C"
            )
#
            st.markdown("## Long-term warming")

            st.write(f"{trend_value:.2f} ± {trend_plus:.3f} °C per decade")
            #st.markdown("---")

    footer_left, footer_right = st.columns([4, 1])

    with footer_left:
        st.caption(
            "Built by Aparajita Sen, physicist. "
            "Data: ERA5 monthly means, Copernicus C3S."
        )

        with footer_right:
            st.markdown(
                "[Source on GitHub](https://github.com/aparajita-web/Climate-trends-Bayesian)"
            )