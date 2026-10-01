import streamlit as st
from styles import MODEL_COLORS

from charts import forecast_chart




def render_forecast_card(
    city,
    month,
    model_key,
    model,
    history,
    forecast,
    observed,
    normal,
    mae,
    trend,
):
    def signed(value):
        sign = "+" if value >= 0 else "-"
        return f"{sign}{abs(value):.1f}"

    # --------------------------------------------------
    # Forecast card container
    # --------------------------------------------------

    st.markdown(
        '<div class="forecast-card">',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    st.markdown(
        f"""
        <div class="forecast-top">

            <div class="forecast-heading">

                <div class="forecast-month">
                    {month}
                </div>

                <div class="forecast-where">
                    {city}, the month after the record ends
                </div>

            </div>

            <div class="decadal-chip">

                <span class="decadal-chip-dot {model_key}"></span>

                {model["name"]}

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Main forecast value
    # --------------------------------------------------

    st.markdown(
        f"""
        <div class="forecast-main">

            <div class="decadal-hero-value">
                {forecast["value"]:.1f}
                <span class="forecast-unit">°C</span>
            </div>

            <div class="decadal-interval">
                {forecast["lower"]:.1f}
                to
                {forecast["upper"]:.1f}
                °C · {model["level"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Forecast chart
    # --------------------------------------------------

    forecast_chart(
        history=history,
        forecast=forecast,
        forecast_month=month,
        model_key=model_key,
        model_name=model["name"],
    )

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    st.markdown(
        f"""
        <div class="decadal-legend">

            <span>
                <i class="legend-line"></i>
                Observed, ERA5, last 24 months
            </span>

            <span>
                <i class="legend-dot {model_key}"></i>
                Forecast with its interval
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Context rows
    # --------------------------------------------------

    st.markdown(
        f"""
        <div class="decadal-context">

            <div class="context-row">

                <span>
                    August 2025, observed
                </span>

                <strong>
                    {observed:.1f} °C

                    <small>
                        forecast is
                        {signed(forecast["value"] - observed)}
                    </small>
                </strong>

            </div>


            <div class="context-row">

                <span>
                    August normal, 1991 to 2020
                </span>

                <strong>
                    {normal:.1f} °C

                    <small>
                        forecast is
                        {signed(forecast["value"] - normal)}
                    </small>
                </strong>

            </div>


            <div class="context-row">

                <span>
                    Model error on 184 held-out months
                </span>

                <strong>
                    MAE {mae:.2f} °C
                </strong>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Bottom trend
    # --------------------------------------------------

    st.markdown(
        f"""
        <div class="forecast-more">

            <span>
                {city} is warming
                {trend["value"]:.2f} °C per decade
                ({trend["lower"]:.2f} to {trend["upper"]:.2f}).
            </span>

            <a href="#">
                See the full analysis →
            </a>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Close card
    # --------------------------------------------------

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )
