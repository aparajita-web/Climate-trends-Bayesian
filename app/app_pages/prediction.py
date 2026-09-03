from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def _metric(df, name):
    return df.loc[df["Metric"].str.upper() == name.upper(), "Value"].iloc[0]


def _read_forecast(plot_dir, filename):
    path = Path(plot_dir) / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _format_date(value):
    return pd.to_datetime(value).strftime("%B %Y")


def _model_card(title, subtitle, value, lower=None, upper=None):
    interval = ""
    if lower is not None and upper is not None:
        interval = f"<div style='color:#6B7280;font-size:14px;margin-top:8px;'>Interval: {lower:.2f} – {upper:.2f} °C</div>"

    st.markdown(
        f"""
        <div style="border:1px solid #E5E7EB;border-radius:14px;padding:20px;text-align:center;min-height:190px;">
            <div style="font-size:14px;color:#6B7280;">{subtitle}</div>
            <h3 style="margin:8px 0 14px 0;">{title}</h3>
            <div style="font-size:34px;font-weight:700;">{value:.2f} °C</div>
            {interval}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _forecast_value(df):
    if df.empty:
        return None, None, None, None

    row = df.iloc[0]
    prediction = row.get("prediction")
    lower = row.get("lower")
    upper = row.get("upper")
    date = row.get("date", row.get("Forecast Month"))

    return (
        float(prediction) if pd.notna(prediction) else None,
        float(lower) if pd.notna(lower) else None,
        float(upper) if pd.notna(upper) else None,
        date,
    )


def prediction_page(plot_dir):
    plot_dir = Path(plot_dir)

    st.title("Next Month Temperature Prediction")
    st.write("Compare three forecasting approaches, evaluate their historical performance, and examine the forecast for the next month.")

    # --------------------------------------------------------
    # DATA & EVALUATION PERIOD
    # --------------------------------------------------------

    st.subheader("📊 Data & Evaluation Period")

    dataset_path = plot_dir / "dataset.csv"
    info_path = plot_dir / "data_info.csv"

    dataset = pd.read_csv(dataset_path) if dataset_path.exists() else pd.DataFrame()
    info = pd.read_csv(info_path) if info_path.exists() else pd.DataFrame()

    info_dict = {}
    if not info.empty and {"Specification", "Value"}.issubset(info.columns):
        info_dict = dict(zip(info["Specification"], info["Value"]))

    if not dataset.empty and "date" in dataset.columns:
        dataset["date"] = pd.to_datetime(dataset["date"])
        full_start = dataset["date"].min()
        full_end = dataset["date"].max()
        total_obs = len(dataset)
    else:
        full_start = full_end = None
        total_obs = None

    test_files = [
        plot_dir / "climatology_predictions.csv",
        plot_dir / "gradient_boost_predictions.csv",
        plot_dir / "SARIMA_predictions.csv",
    ]

    test_dates = []
    for path in test_files:
        if path.exists():
            temp = pd.read_csv(path)
            if "date" in temp.columns:
                test_dates.extend(pd.to_datetime(temp["date"]).tolist())

    if test_dates:
        test_start = min(test_dates)
        test_end = max(test_dates)
        test_obs = len(set(test_dates))
        train_end = test_start - pd.DateOffset(months=1)
        train_start = full_start
        train_obs = len(dataset[dataset["date"] <= train_end]) if not dataset.empty else None
    else:
        train_start = train_end = test_start = test_end = None
        train_obs = test_obs = None

    data_cols = st.columns(4)

    with data_cols[0]:
        st.markdown(f"**Dataset**  \nERA5")
        st.markdown(f"**Variable**  \n2 m air temperature (`t2m`)")

    with data_cols[1]:
        city = info_dict.get("City", "Selected city")
        lat = info_dict.get("Latitude", "—")
        lon = info_dict.get("Longitude", "—")
        st.markdown(f"**Location**  \n{city}")
        st.markdown(f"**Coordinates**  \n{lat}, {lon}")

    with data_cols[2]:
        st.markdown(f"**Full period**  \n{full_start:%b %Y} – {full_end:%b %Y}" if full_start is not None else "**Full period**  \n—")
        st.markdown(f"**Observations**  \n{total_obs:,}" if total_obs is not None else "**Observations**  \n—")

    with data_cols[3]:
        st.markdown(f"**Training**  \n{train_start:%b %Y} – {train_end:%b %Y}" if train_start is not None else "**Training**  \n—")
        st.markdown(f"**Testing**  \n{test_start:%b %Y} – {test_end:%b %Y}" if test_start is not None else "**Testing**  \n—")

    st.divider()

    # --------------------------------------------------------
    # NEXT MONTH FORECAST
    # --------------------------------------------------------

    st.subheader("🔮 Next Month Forecast")

    climatology_forecast = _read_forecast(plot_dir, "climatology_forecast.csv")
    gb_forecast = _read_forecast(plot_dir, "gradient_boost_forecast.csv")
    sarima_forecast = _read_forecast(plot_dir, "SARIMA_forecast.csv")

    c_pred, c_lower, c_upper, forecast_date = _forecast_value(climatology_forecast)
    g_pred, g_lower, g_upper, _ = _forecast_value(gb_forecast)
    s_pred, s_lower, s_upper, _ = _forecast_value(sarima_forecast)

    if forecast_date is not None:
        st.markdown(f"### Forecast for {_format_date(forecast_date)}")

    if all(v is not None for v in [c_pred, g_pred, s_pred]):
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            _model_card("Climatology", "Baseline", c_pred, c_lower, c_upper)

        with col2:
            _model_card("Gradient Boost", "Machine Learning", g_pred, g_lower, g_upper)

        with col3:
            _model_card("SARIMA", "Time-series model", s_pred, s_lower, s_upper)

        # ----------------------------------------------------
        # FORECAST COMPARISON
        # ----------------------------------------------------

        st.markdown("### Forecast Comparison")

        comparison_fig = go.Figure()
        models = ["Climatology", "Gradient Boost", "SARIMA"]
        predictions = [c_pred, g_pred, s_pred]

        comparison_fig.add_trace(go.Bar(
            x=models,
            y=predictions,
            text=[f"{x:.2f} °C" for x in predictions],
            textposition="auto",
            marker_color=["#7A8B8B", "#6F8F8F", "#B08A62"],
        ))

        comparison_fig.update_layout(
            yaxis_title="Temperature (°C)",
            xaxis_title="",
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(comparison_fig, use_container_width=True)

    else:
        st.warning("Next-month forecast files could not be read.")

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("📈 Model Performance")
    st.write("Performance evaluated on the historical testing period.")

    climatology_metrics = _read_forecast(plot_dir, "climatology_metrics.csv")
    gb_metrics = _read_forecast(plot_dir, "GradientBoost_metrics.csv")
    sarima_metrics = _read_forecast(plot_dir, "SARIMA_metrics.csv")

    if not climatology_metrics.empty and not gb_metrics.empty and not sarima_metrics.empty:
        metrics_df = pd.DataFrame({
            "Model": ["Climatology", "Gradient Boost", "SARIMA"],
            "MAE (°C)": [
                _metric(climatology_metrics, "MAE"),
                _metric(gb_metrics, "MAE"),
                _metric(sarima_metrics, "MAE"),
            ],
            "RMSE (°C)": [
                _metric(climatology_metrics, "RMSE"),
                _metric(gb_metrics, "RMSE"),
                _metric(sarima_metrics, "RMSE"),
            ],
            "R²": [
                _metric(climatology_metrics, "R2"),
                _metric(gb_metrics, "R2"),
                _metric(sarima_metrics, "R2"),
            ],
        })

        st.dataframe(
            metrics_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "MAE (°C)": st.column_config.NumberColumn(format="%.3f °C"),
                "RMSE (°C)": st.column_config.NumberColumn(format="%.3f °C"),
                "R²": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    else:
        st.warning("Model evaluation files could not be read.")

    # --------------------------------------------------------
    # HISTORICAL PREDICTION EXPLORER
    # --------------------------------------------------------

    st.subheader("🔍 Historical Prediction Explorer")
    st.write("Explore how the three models performed during the testing period.")

    prediction_files = {
        "Climatology": "climatology_predictions.csv",
        "Gradient Boost": "gradient_boost_predictions.csv",
        "SARIMA": "SARIMA_predictions.csv",
    }

    predictions = {}

    for model, filename in prediction_files.items():
        path = plot_dir / filename
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            predictions[model] = df

    if predictions:
        common_dates = set.intersection(*[set(df["date"]) for df in predictions.values()])
        common_dates = sorted(common_dates)

        if common_dates:
            selected_date = st.selectbox(
                "Select a test month",
                common_dates,
                format_func=lambda x: x.strftime("%B %Y"),
            )

            fig = go.Figure()

            for model, df in predictions.items():
                row = df[df["date"] == selected_date].iloc[0]

                fig.add_trace(go.Scatter(
                    x=[selected_date],
                    y=[row["prediction"]],
                    mode="markers",
                    name=model,
                    marker=dict(size=12),
                ))

                if pd.notna(row.get("lower")) and pd.notna(row.get("upper")):
                    fig.add_trace(go.Scatter(
                        x=[selected_date, selected_date],
                        y=[row["lower"], row["upper"]],
                        mode="lines",
                        name=f"{model} interval",
                        showlegend=False,
                    ))

            actual = predictions["Climatology"].loc[
                predictions["Climatology"]["date"] == selected_date, "actual"
            ].iloc[0]

            fig.add_trace(go.Scatter(
                x=[selected_date],
                y=[actual],
                mode="markers",
                name="Actual",
                marker=dict(size=14, symbol="diamond"),
            ))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                height=450,
                margin=dict(l=20, r=20, t=20, b=20),
            )

            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # FORECASTING MODELS
    # --------------------------------------------------------

    st.subheader("🧠 Forecasting Models")

    model_cols = st.columns(3)

    with model_cols[0]:
        st.markdown("### 🌡️ Climatology")
        st.write("A baseline forecast based on the historical temperature distribution for the corresponding calendar month.")

    with model_cols[1]:
        st.markdown("### 📈 Gradient Boosting")
        st.write("A machine-learning model that learns the relationship between temporal and seasonal features and temperature.")

    with model_cols[2]:
        st.markdown("### 📊 SARIMA")
        st.write("A time-series model that captures temporal dependence and seasonal structure in the temperature series.")