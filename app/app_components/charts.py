import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles import MODEL_COLORS


PLOT_COLORS = {
    "historical": "#C8D2E5",
    "axis": "#7F8BA7",
    "grid": "rgba(255,255,255,0.10)",
    "forecast_zone": "#36ADA3",
    "text": "#34405F",
}


def forecast_chart(
    history,
    forecast,
    forecast_month,
    model_key,
    model_name,
):
    """
    Reproduce the compact forecast chart from the HTML page.
    """

    # Last 24 observations
    hist = history.tail(24).copy()

    hist["date"] = pd.to_datetime(
        hist["date"]
    )

    forecast_date = pd.to_datetime(
        forecast["date"]
    )

    fig = go.Figure()

    # --------------------------------------------------
    # Observed temperature
    # --------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["actual"],
            mode="lines",

            line=dict(
                color="#C8D2E5",
                width=2,
            ),

            hovertemplate=(
                "%{x|%b %Y}"
                "<br>%{y:.1f} °C observed"
                "<extra></extra>"
            ),

            showlegend=False,
        )
    )

    # --------------------------------------------------
    # Prediction interval
    # --------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[
                forecast_date,
                forecast_date,
            ],

            y=[
                forecast["lower"],
                forecast["upper"],
            ],

            mode="lines",

            line=dict(
                color=MODEL_COLORS[model_key],
                width=2,
            ),

            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --------------------------------------------------
    # Forecast point
    # --------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[forecast_date],

            y=[
                forecast["value"]
            ],

            mode="markers+text",

            marker=dict(
                size=10,
                color=MODEL_COLORS[model_key],
                line=dict(
                    width=2,
                    color="#121358",
                ),
            ),

            text=[
                f'{forecast["value"]:.1f} °C'
            ],

            textposition="middle right",

            textfont=dict(
                size=12,
                color="#C8D2E5",
            ),

            hovertemplate=(
                f"<b>{forecast_month}</b>"
                f"<br>{forecast['value']:.1f} °C forecast"
                f", {model_name}"
                f"<br>{forecast['lower']:.1f}"
                f" to {forecast['upper']:.1f} °C"
                "<extra></extra>"
            ),

            showlegend=False,
        )
    )

    # --------------------------------------------------
    # Forecast zone
    # --------------------------------------------------

    fig.add_vrect(
        x0=hist["date"].iloc[-1],
        x1=forecast_date + pd.Timedelta(days=20),

        fillcolor="#36ADA3",
        opacity=0.65,

        line_width=0,

        layer="below",
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    fig.update_layout(

        height=214,

        margin=dict(
            l=46,
            r=66,
            t=18,
            b=28,
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        showlegend=False,

        font=dict(
            family="Geist Mono, monospace",
            size=11,
            color="#9FAAC1",
        ),

        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor="#7F8BA7",
            fixedrange=True,
            tickformat="%b %Y",
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
            fixedrange=True,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "scrollZoom": False,
        },
    )
