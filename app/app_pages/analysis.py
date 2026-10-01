from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys
import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
APP_COMPONENTS = APP_DIR / "app_components"
sys.path.insert(0, str(APP_COMPONENTS))

from styles import MODEL_COLORS



CITIES = ["Paris", "Berlin", "Kolkata", "Hanoi"]

MODEL_INFO = {
    "clim": {
        "name": "Climatology",
        "kind": "baseline",
        "color": MODEL_COLORS["clim"],
        "forecast": "climatology_forecast.csv",
        "metrics": "climatology_metrics.csv",
    },
    "gb": {
        "name": "Gradient boosting",
        "kind": "machine learning",
        "color": MODEL_COLORS["gb"],
        "forecast": "gradient_boost_forecast.csv",
        "metrics": "GradientBoost_metrics.csv",
    },
    "sarima": {
        "name": "SARIMA",
        "kind": "time series",
        "color": MODEL_COLORS["sarima"],
        "forecast": "SARIMA_forecast.csv",
        "metrics": "SARIMA_metrics.csv",
    },
}

BAYES_FILES = {
    "mean": {
        "posterior": "ModelA_average_tempposterior_values.csv",
        "rhat": "ModelA_average_temprhat.csv",
        "ess": "ModelA_average_tempess.csv",
    },
    "max": {
        "posterior": "ModelA_max_tempposterior_values.csv",
        "rhat": "ModelA_max_temprhat.csv",
        "ess": "ModelA_max_tempess.csv",
    },
}


def load_history(city_dir):
    path = city_dir / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = {"date", "actual"}

    if not required.issubset(df.columns):
        raise ValueError(f"dataset.csv must contain {required}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df = df.dropna(subset=["date", "actual"]).sort_values("date").copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    return df


def annual_series(df):
    return df.groupby("year")["actual"].agg(["mean", "max"]).reset_index()


def august_normal(df):
    x = df[df["year"].between(1991, 2020) & (df["month"] == 8)]
    return float(x["actual"].mean())


def observed_august_2025(df):
    x = df[(df["year"] == 2025) & (df["month"] == 8)]
    return float(x["actual"].iloc[0]) if not x.empty else None


def load_trend(city_dir):
    path = city_dir / "warming_trend_per_decade.csv"

    if not path.exists():
        alt = city_dir / "warming_trend_per_decade(1).csv"
        if alt.exists():
            path = alt

    if not path.exists():
        raise FileNotFoundError(f"Warming trend file not found in {city_dir}")

    row = pd.read_csv(path).iloc[0]

    return {
        "average": float(row["Average_Trend"]),
        "average_minus": float(row["Average_Minus_Error"]),
        "average_plus": float(row["Average_Plus_Error"]),
        "extreme": float(row["Extreme_Trend"]),
        "extreme_minus": float(row["Extreme_Minus_Error"]),
        "extreme_plus": float(row["Extreme_Plus_Error"]),
    }


def load_bayes(city_dir, mode="mean"):
    files = BAYES_FILES[mode]

    posterior_path = city_dir / files["posterior"]
    rhat_path = city_dir / files["rhat"]
    ess_path = city_dir / files["ess"]

    missing = [
        str(p)
        for p in [posterior_path, rhat_path, ess_path]
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError("Missing Bayesian files:\n" + "\n".join(missing))

    posterior = pd.read_csv(posterior_path).set_index("Parameter")
    rhat = pd.read_csv(rhat_path).set_index("Parameter")["Rhat"].to_dict()
    ess = pd.read_csv(ess_path).set_index("Parameter")["ESS"].to_dict()

    posterior.index = posterior.index.str.lower()

    return {
        "posterior": posterior,
        "rhat": {str(k).lower(): float(v) for k, v in rhat.items()},
        "ess": {str(k).lower(): float(v) for k, v in ess.items()},
    }


def posterior_value(bayes, parameter):
    if parameter not in bayes["posterior"].index:
        return None
    return float(bayes["posterior"].loc[parameter, "Median"])


def posterior_ci(bayes, parameter):
    if parameter not in bayes["posterior"].index:
        return None, None

    row = bayes["posterior"].loc[parameter]

    return float(row["Lower_68CI"]), float(row["Upper_68CI"])


def load_model_forecast(city_dir, model_key):
    cfg = MODEL_INFO[model_key]
    path = city_dir / cfg["forecast"]

    if not path.exists():
        raise FileNotFoundError(f"Forecast file not found: {path}")

    row = pd.read_csv(path).iloc[0]

    return {
        "prediction": float(row["prediction"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
    }


def load_model_metrics(city_dir, model_key):
    cfg = MODEL_INFO[model_key]
    path = city_dir / cfg["metrics"]

    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    df = pd.read_csv(path)
    out = {}

    if {"Metric", "Value"}.issubset(df.columns):
        for _, row in df.iterrows():
            out[str(row["Metric"]).strip().upper()] = float(row["Value"])

    return out


def best_model(city_dir):
    scores = {}

    for key in MODEL_INFO:
        metrics = load_model_metrics(city_dir, key)
        scores[key] = metrics.get("MAE", float("inf"))

    return min(scores, key=scores.get)


def forecast_month(city_dir, df):
    path = city_dir / "forecast_next_month.csv"

    if path.exists():
        x = pd.read_csv(path)

        for col in ["Forecast_Month", "forecast_month"]:
            if col in x.columns:
                return pd.to_datetime(x[col].iloc[0]).strftime("%B %Y")

    last_date = df["date"].max()
    return (last_date + pd.offsets.MonthBegin(1)).strftime("%B %Y")


def build_stripes_chart(df):
    annual = annual_series(df)
    baseline = float(annual.loc[annual["year"] <= 1980, "mean"].mean())
    anomaly = annual["mean"] - baseline
    max_abs = max(abs(anomaly.min()), abs(anomaly.max()))

    fig = go.Figure(
        go.Bar(
            x=annual["year"],
            y=[1] * len(annual),
            marker={
                "color": anomaly,
                "colorscale": [
                    [0.0, "#0B07EB"],
                    [0.5, "#F6F7F9"],
                    [1.0, "#FC0909"],
                ],
                "cmin": -max_abs,
                "cmax": max_abs,
                "showscale": True,
            },
            customdata=anomaly,
            hovertemplate="%{x}<br>%{customdata:+.2f} °C vs 1950–1980<extra></extra>",
        )
    )

    fig.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        bargap=0,
    )

    return fig, baseline, max_abs


def build_record_chart(annual, trend_value, trend_minus, trend_plus, beta0=None, mode="mean"):
    value_col = "max" if mode == "max" else "mean"

    years = annual["year"].to_numpy()
    values = annual[value_col].to_numpy()

    midpoint = (years.min() + years.max()) / 2

    if beta0 is not None:
        trend_line = beta0 + trend_value / 10 * (years - midpoint)
    else:
        center = values.mean()
        trend_line = center + trend_value / 10 * (years - midpoint)

    half_band = (
        (trend_minus + trend_plus) / 2
        + abs(years - midpoint) / 10 * (trend_minus + trend_plus) / 2
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=trend_line + half_band,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=trend_line - half_band,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(54,173,163,0.20)",
            name="68% credible band",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=trend_line,
            mode="lines",
            line=dict(color="#36ADA3", width=2.5),
            name="Bayesian trend",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=values,
            mode="lines",
            line=dict(color="#C8D2E5", width=2.2),
            name="ERA5 annual maximum" if mode == "max" else "ERA5 annual mean",
            hovertemplate="%{x}: %{y:.2f} °C<extra></extra>",
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(l=55, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist Mono, monospace", size=11, color="#9FAAC1"),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor="#7F8BA7",
            dtick=10,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
            title="°C",
        ),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0,
        ),
    )

    return fig


def build_forecast_chart(forecasts, normal, observed):
    ordered = sorted(
        forecasts.items(),
        key=lambda item: item[1]["prediction"]
    )

    all_values = []

    for _, result in ordered:
        all_values.extend(
            [result["lower"], result["prediction"], result["upper"]]
        )

    all_values.append(normal)

    if observed is not None:
        all_values.append(observed)

    xmin = min(all_values) - 0.7
    xmax = max(all_values) + 0.7

    fig = go.Figure()

    fig.add_vline(
        x=normal,
        line_color="#7F8BA7",
        line_width=1,
    )

    if observed is not None:
        fig.add_vline(
            x=observed,
            line_color="#9FAAC1",
            line_width=2,
        )

    row_y = {
        "sarima": 1,
        "gb": 2,
        "clim": 3,
    }

    for key, result in ordered:
        cfg = MODEL_INFO[key]
        y = row_y[key]

        fig.add_trace(
            go.Scatter(
                x=[result["lower"], result["upper"]],
                y=[y, y],
                mode="lines",
                line=dict(color=cfg["color"], width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[result["prediction"]],
                y=[y],
                mode="markers",
                marker=dict(
                    color=cfg["color"],
                    size=10,
                ),
                name=cfg["name"],
                hovertemplate=(
                    f"{cfg['name']}<br>"
                    f"{result['prediction']:.2f} °C<br>"
                    f"{result['lower']:.2f} to {result['upper']:.2f} °C"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=250,
        margin=dict(l=135, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist Mono, monospace", size=11, color="#9FAAC1"),
        xaxis=dict(
            range=[xmin, xmax],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.10)",
            title="°C",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[1, 2, 3],
            ticktext=[
                "SARIMA",
                "Gradient boosting",
                "Climatology",
            ],
            showgrid=False,
            range=[0.5, 3.5],
        ),
    )

    return fig


def analysis_page(plot_dir):

    current_city = plot_dir.name if plot_dir.name in CITIES else "Paris"

    '''city = st.selectbox(
        "City",
        CITIES,
        index=CITIES.index(current_city),
        key="analysis_city",
    )'''
    city = st.segmented_control(
                "City",
                CITIES,
                default=st.session_state.get("predict_city", "Paris"),
                selection_mode="single",
                width="stretch",
                key="predict_city_selector"
            )
    
    if city is None:
                city = "Paris"
    city_dir = plot_dir.parent / city

    try:
        df = load_history(city_dir)
        annual = annual_series(df)
        trend = load_trend(city_dir)
        month = forecast_month(city_dir, df)

    except Exception as exc:
        st.error(f"Could not load analysis data for {city}: {exc}")
        return

    # ==========================================================
    # HERO
    # ==========================================================

    average_trend = trend["average"]
    average_low = average_trend - trend["average_minus"]
    average_high = average_trend + trend["average_plus"]
    extreme_trend = trend["extreme"] 

    best_key = best_model(city_dir)
    best_forecast = load_model_forecast(city_dir, best_key)

    hot = annual.loc[annual["mean"].idxmax()]

    left, right = st.columns([7, 5], gap="large")

    with left:
        st.subheader(
            f"{city} has warmed {average_trend:.2f} °C per decade since 1950."
        )

        st.write(
            f"A Bayesian AR(1) trend fitted to {len(annual)-1} annual means "
            f"from ERA5 reanalysis. The 68% credible interval runs from "
            f"{average_low:.2f} to {average_high:.2f} °C per decade."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            fitted_rise = average_trend * (
                annual["year"].max() - annual["year"].min()
            ) / 10
            st.metric(
                "Fitted rise",
                f"+{fitted_rise:.1f} °C",
                "1950–2025",
            )

        with c2:
            st.metric(
                month,
                f"{best_forecast['prediction']:.1f} °C",
                best_key.upper(),
            )

        with c3:
            st.metric(
                "Warmest year",
                int(hot["year"]),
                f"{hot['mean']:.1f} °C annual mean",
            )

    with right:
        with st.container(border=True,key="analysis_stripes"):
            st.markdown("**Warming stripes**")
            st.caption(
                f"Annual mean, 1950–{int(annual['year'].max())}, "
                "relative to the 1950–1980 baseline."
            )

            fig, baseline, max_abs = build_stripes_chart(df)

            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False},
            )

            st.caption(
                f"Baseline: {baseline:.2f} °C   "
                f"Range: −{max_abs:.1f} to +{max_abs:.1f} °C"
            )

    st.divider()

    left, right = st.columns([7, 5], gap="large")

    with left:

        with st.container(border=True, key="analysis_record"):

            st.subheader("Seventy-six years of annual temperature")

            st.caption(
                "ERA5 annual series with the Bayesian trend "
                "and its 68% credible band."
            )

            mode_label = st.segmented_control(
                "Series",
                ["Annual mean", "Annual maximum"],
                default="Annual mean",
                key="record_mode",
                label_visibility="collapsed",
            )

            mode = "max" if mode_label == "Annual maximum" else "mean"

            try:

                bayes = load_bayes(city_dir, mode)

                trend_value = (
                    trend["extreme"] if mode == "max"
                    else trend["average"]
                )

                trend_minus = (
                    trend["extreme_minus"] if mode == "max"
                    else trend["average_minus"]
                )

                trend_plus = (
                    trend["extreme_plus"] if mode == "max"
                    else trend["average_plus"]
                )

                beta0 = posterior_value(bayes, "beta0")

                fig = build_record_chart(
                    annual,
                    trend_value,
                    trend_minus,
                    trend_plus,
                    beta0=beta0,
                    mode=mode,
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"displayModeBar": False},
                )

                st.caption(
                    "ERA5 annual maximum"
                    if mode == "max"
                    else "ERA5 annual mean"
                )

            except Exception as exc:

                st.warning(
                    f"Bayesian record model is unavailable for "
                    f"{'annual maxima' if mode == 'max' else 'annual means'}: {exc}"
                )

            if st.toggle("Show table", key="record_table"):

                table = annual.rename(
                    columns={
                        "year": "Year",
                        "mean": "Annual mean",
                        "max": "Annual maximum",
                    }
                )

                st.dataframe(
                    table,
                    width="stretch",
                    hide_index=True,
                )


    # ==========================================================
    # RIGHT COLUMN — BAYESIAN MODELS
    # ==========================================================

    with right:

        with st.container(border=True):

            st.subheader("Trend model")

            st.caption(
                "Linear trend with first-order autocorrelated residuals, "
                "sampled with emcee."
            )

            # Keep this exactly as your original code
            st.code(
                "Tₜ = β₀ + β₁·t + εₜ\n"
                "εₜ = φ·εₜ₋₁ + ηₜ\n"
                "ηₜ ~ N(0, σ²),   φ = tanh(ψ)",
                language="text",
            )

            # ==================================================
            # ANNUAL MEAN
            # ==================================================

            try:

                bayes_mean = load_bayes(city_dir, "mean")

                beta0_mean = posterior_value(
                    bayes_mean, "beta0"
                )

                beta0_mean_low, beta0_mean_high = posterior_ci(
                    bayes_mean, "beta0"
                )

                psi_mean = posterior_value(
                    bayes_mean, "psi"
                )

                sigma_mean = posterior_value(
                    bayes_mean, "sigma"
                )

                phi_mean = (
                    None
                    if psi_mean is None
                    else float(np.tanh(psi_mean))
                )

                #st.write("**Annual mean temperature**")
                st.markdown(
                    """
                    <div style="
                        background-color: #E6F4F1;
                        border-left: 4px solid #0F766E;
                        padding: 7px 12px;
                        border-radius: 6px;
                        margin: 8px 0 12px 0;
                        font-weight: 600;
                        color: #0F766E;
                    ">
                        Annual mean temperature
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.metric(
                        "Trend β₁",
                        f"{average_trend:.3f} °C/decade",
                        help="Bayesian warming trend of the annual mean temperature.",
                    )

                    if beta0_mean is not None:

                        st.metric(
                            "Level β₀",
                            f"{beta0_mean:.3f} °C",
                            "record midpoint",
                            help=(
                                "Estimated temperature level at the midpoint "
                                "of the time series."
                            ),
                        )

                with c2:

                    if phi_mean is not None:

                        st.metric(
                            "Autocorrelation φ",
                            f"{phi_mean:.3f}",
                            help=(
                                "First-order autocorrelation of the residuals."
                            ),
                        )

                    if sigma_mean is not None:

                        st.metric(
                            "Residual σ",
                            f"{sigma_mean:.3f} °C",
                            help=(
                                "Estimated standard deviation of the "
                                "innovation/residual noise."
                            ),
                        )

            except Exception as exc:

                st.warning(
                    f"Annual mean Bayesian model unavailable: {exc}"
                )


            #st.divider()


            # ==================================================
            # ANNUAL MAXIMUM
            # ==================================================

            try:

                bayes_max = load_bayes(city_dir, "max")

                beta0_max = posterior_value(
                    bayes_max, "beta0"
                )

                beta0_max_low, beta0_max_high = posterior_ci(
                    bayes_max, "beta0"
                )

                psi_max = posterior_value(
                    bayes_max, "psi"
                )

                sigma_max = posterior_value(
                    bayes_max, "sigma"
                )

                phi_max = (
                    None
                    if psi_max is None
                    else float(np.tanh(psi_max))
                )

                #st.write("**Annual maximum temperature**")
                st.markdown(
                    """
                    <div style="
                        background-color: #FFF4E5;
                        border-left: 4px solid #B45309;
                        padding: 7px 12px;
                        border-radius: 6px;
                        margin: 8px 0 12px 0;
                        font-weight: 600;
                        color: #B45309;
                    ">
                        Annual maximum temperature
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)
                
                with c1:

                    st.metric(
                        "Trend β₁",
                        f"{extreme_trend:.3f} °C/decade",
                        help="Bayesian warming trend of the annual maximum temperature.",
                    )

                    if beta0_max is not None:

                        st.metric(
                            "Level β₀",
                            f"{beta0_max:.3f} °C",
                            "record midpoint",
                            help=(
                                "Estimated temperature level at the midpoint "
                                "of the time series."
                            ),
                        )

                with c2:

                    if phi_max is not None:

                        st.metric(
                            "Autocorrelation φ",
                            f"{phi_max:.3f}",
                            help=(
                                "First-order autocorrelation of the residuals."
                            ),
                        )

                    if sigma_max is not None:

                        st.metric(
                            "Residual σ",
                            f"{sigma_max:.3f} °C",
                            help=(
                                "Estimated standard deviation of the "
                                "innovation/residual noise."
                            ),
                        )

            except Exception as exc:

                st.warning(
                    f"Annual maximum Bayesian model unavailable: {exc}"
                )


            st.divider()


            # ==================================================
            # CONVERGENCE DIAGNOSTICS
            # ==================================================

            st.write("**Convergence diagnostics**")

            try:

                max_rhat_mean = max(bayes_mean["rhat"].values())
                max_rhat_max = max(bayes_max["rhat"].values())

                min_ess_mean = min(bayes_mean["ess"].values())
                min_ess_max = min(bayes_max["ess"].values())

                max_rhat = max(
                    max_rhat_mean,
                    max_rhat_max,
                )

                min_ess = min(
                    min_ess_mean,
                    min_ess_max,
                )

                n_parameters = len(
                    bayes_mean["posterior"]
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Max R-hat",
                        f"{max_rhat:.3f}",
                        help=(
                            "Maximum R-hat across both Bayesian models. "
                            "Values close to 1 indicate good convergence."
                        ),
                    )

                with c2:

                    st.metric(
                        "Min ESS",
                        f"{min_ess:,.0f}",
                        help=(
                            "Minimum effective sample size across "
                            "both Bayesian models."
                        ),
                    )

                with c3:

                    st.metric(
                        "Parameters",
                        n_parameters,
                        "per model",
                    )

                st.caption(
                    "Intervals shown above are 68% posterior credible intervals."
                )

            except Exception as exc:

                st.warning(
                    f"Convergence diagnostics unavailable: {exc}"
                )

    # ==========================================================
    # FORECAST + PERFORMANCE
    # ==========================================================

    left, right = st.columns([7, 5], gap="large")

    forecasts = {
        key: load_model_forecast(city_dir, key)
        for key in MODEL_INFO
    }

    normal = august_normal(df)
    observed = observed_august_2025(df)

    with left:
        with st.container(border=True,key="analysis_forecast"):
            st.subheader(f"{month}, three ways")
            st.caption(
                f"Each model forecasts next month's mean 2 m temperature "
                f"from the record to {df['date'].max():%B %Y}. "
                "Interval as reported by the model."
            )

            fig = build_forecast_chart(
                forecasts,
                normal,
                observed,
            )

            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False},
            )

            st.caption(
                f"August normal, 1991–2020: {normal:.2f} °C"
            )

            if observed is not None:
                st.caption(
                    f"August 2025 observed: {observed:.2f} °C"
                )

    with right:
        with st.container(border=True,key="analysis_performance"):
            st.subheader("Test-period performance")
            st.caption(
                "Metrics are read directly from the model output CSVs."
            )

            rows = []

            for key in ["clim", "gb", "sarima"]:
                m = load_model_metrics(city_dir, key)

                rows.append(
                    {
                        "Model": MODEL_INFO[key]["name"],
                        "MAE (°C)": m.get("MAE"),
                        "RMSE (°C)": m.get("RMSE"),
                        "R²": m.get("R2", m.get("R²")),
                    }
                )

            metrics_df = pd.DataFrame(rows)

            st.dataframe(
                metrics_df,
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Climatology is the baseline. "
                "Gradient boosting uses calendar and seasonal features. "
                "SARIMA models the monthly time series."
            )

    st.divider()

    # ==========================================================
    # DATA + PIPELINE
    # ==========================================================

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True,key="analysis_data"):
            st.subheader("The data")
            st.caption(
                "One reanalysis grid box per city, monthly, no gaps."
            )

            data_info_path = city_dir / "data_info.csv"

            if data_info_path.exists():
                info_df = pd.read_csv(data_info_path)

                if {"Specification", "Value"}.issubset(info_df.columns):
                    info = dict(
                        zip(
                            info_df["Specification"],
                            info_df["Value"],
                        )
                    )

                    specs = pd.DataFrame(
                        {
                            "": [
                                "Source",
                                "Variable",
                                "Grid box",
                                "Period",
                                "Observations",
                                "Split",
                            ],
                            "Value": [
                                "ERA5 monthly means — Copernicus C3S",
                                info.get(
                                    "Variable",
                                    "2 m air temperature (t2m)",
                                ),
                                (
                                    f"{info.get('Latitude', 'N/A')}, "
                                    f"{info.get('Longitude', 'N/A')}"
                                ),
                                (
                                    f"{df['date'].min():%B %Y} to "
                                    f"{df['date'].max():%B %Y}"
                                ),
                                f"{len(df)} months",
                                "735 train, 184 test",
                            ],
                        }
                    )

                    st.dataframe(
                        specs,
                        width="stretch",
                        hide_index=True,
                    )

                else:
                    st.info("data_info.csv has an unexpected format.")

            else:
                st.info("data_info.csv not found.")

    with right:
        with st.container(border=True,key="analysis_pipeline"):
            st.subheader("How a city gets processed")
            st.caption(
                "One YAML configuration in, one folder of results out."
            )

            pipeline = [
                ("1. Download", "cdsapi, xarray", "Pull ERA5 data."),
                ("2. Explore", "pandas, STL", "Annual means, anomalies and decomposition."),
                ("3. Fit trend", "emcee, arviz", "AR(1) trend and diagnostics."),
                ("4. Forecast", "scikit-learn, statsmodels", "Three forecasting approaches."),
                ("5. Track", "MLflow", "Metrics and artefacts per city."),
            ]

            for title, library, description in pipeline:
                st.markdown(f"**{title}**")
                st.caption(f"{library} — {description}")

    # ==========================================================
    # FOOTER
    # ==========================================================

    st.divider()

    left, right = st.columns([7, 5])

    with left:
        st.caption(
            "Built by Dr. Aparajita Sen. "
            "Data: ERA5 monthly means, Copernicus C3S."
        )

    with right:
        st.markdown(
            "[Source on GitHub](https://github.com/aparajita-web/Climate-trends-Bayesian)"
        )
