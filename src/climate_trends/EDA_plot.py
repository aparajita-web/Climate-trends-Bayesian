
import numpy as np
import matplotlib.pyplot as plt
import sys
import xarray
sys.path.append("../src/climate_trends/")

import load as ld
import xarray as xr
import pandas as pd
from statsmodels.tsa.seasonal import STL



def plot_raw_monthly(t2m_p,output_path):
    plt.figure(figsize=(12,5))

    plt.plot(t2m_p["valid_time"], t2m_p, color="black", alpha=0.6, label="Paris",marker='o')
    #plt.plot(t2m_b["valid_time"], t2m_b, color="red", alpha=0.6, label="Berlin")

    plt.title("Monthly Temperature (Raw) for Paris")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid()
    ############

    plt.figure(figsize=(12,5))
    plt.plot(t2m_p["valid_time"], t2m_p, color="red", alpha=0.6,marker='o')

    plt.title("Monthly Temperature (Raw)")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)

def plot_yearly_trend(t2m_p,output_path):
    t2m_p_year = t2m_p.resample({"valid_time": "1YE"}).mean()
    t2m_p_5year = t2m_p.resample({"valid_time": "5YE"}).mean()

    plt.figure(figsize=(12,5))

    plt.plot(t2m_p_year["valid_time"], t2m_p_year, marker="o", color="darkblue", label="Yearly Average")
    plt.plot(t2m_p_5year["valid_time"], t2m_p_5year, marker="o", color="black", label="5 Yearly Average")
    plt.title("Temperature Trend")
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid()

    
    plt.savefig(output_path)
    #########################

def plot_anomaly_trend(t2m_p,output_path):
    baseline_p = t2m_p.sel(valid_time=slice("1950", "1980"))
    climatology_p = baseline_p.groupby("valid_time.month").mean()
    anomaly_p = t2m_p.groupby("valid_time.month") - climatology_p
    
    anomaly_5yearly = anomaly_p.resample(valid_time="5YE").mean()
    anomaly_yearly = anomaly_p.resample(valid_time="1YE").mean()
    plt.figure(figsize=(12,5))

    plt.plot(anomaly_yearly["valid_time"], anomaly_yearly, marker="o",label="Averaged over 1Y ",color='darkblue',alpha=0.5)
    plt.plot(anomaly_5yearly["valid_time"], anomaly_5yearly, marker="o",label="Averaged over 5Y",color='red')
    plt.axhline(0, color="black", linestyle="--")
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Temperature anomaly (°C)")
    plt.title("Temperature Anomaly [Mean over 1 year and 5 years ]")
    plt.grid()

    
    plt.savefig(output_path)

def plot_stl_decomposition(t2m_p,output_path):
    series = t2m_p.to_series().dropna()
    series = series.sort_index()


    stl_p = STL(series, period=12, robust=True)  # monthly data → 12
    result = stl_p.fit()
    fig, axes = plt.subplots(4, 1, figsize=(10,8), sharex=True)

    axes[0].plot(series, color="black")
    axes[0].set_title("Original")
    axes[0].set_ylabel("Temp (°C)")

    axes[1].plot(result.trend, color="blue")
    axes[1].set_title("Trend")
    axes[1].set_ylabel("Deviation (°C)")

    axes[2].plot(result.seasonal, color="green")
    axes[2].set_title("Seasonal")
    axes[2].set_ylabel("Deviation (°C)")

    axes[3].plot(result.resid, color="red")
    axes[3].set_title("Residual")
    axes[3].set_ylabel("Deviation (°C)")

    plt.tight_layout()

    plt.savefig(output_path)
    
    
###########################################################################################
#### Generate Interactive plots for dash board ############################################################
import plotly.graph_objects as go
from plotly.subplots import make_subplots

## define the dasboard theme #######
##### define the colors
# =====================================================
# Plot Theme
# =====================================================

PRIMARY = "#2563EB"      # Blue
SECONDARY = "#64748B"    # Slate
ORANGE = "#F97316"       # Orange

TEAL = "#14B8A6"
GREEN = "#16A34A"
RED = "#DC2626"

BLACK = "#374151"
GREY = "#94A3B8"
GRID = "#E5E7EB"

FONT = "Inter"
################################################

def apply_dashboard_theme(fig, title, ylabel):

    fig.update_layout(

        template="plotly_white",

        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(
                size=22,
                family="Inter"
            ),
        ),

        font=dict(
            family="Inter",
            size=15,
            color=BLACK,
        ),

        hovermode="x unified",

        transition_duration=400,

        paper_bgcolor="white",
        plot_bgcolor="white",

        height=520,

        margin=dict(
            l=30,
            r=20,
            t=70,
            b=30,
        ),

        legend=dict(
            orientation="h",
            y=1.05,
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),

        xaxis=dict(
            title="Year",
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            showline=False,
        ),

        yaxis=dict(
            title=ylabel,
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            showline=False,
        ),
    )

    return fig
    
def save_plotly(fig, output_path):

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
    )
 
###### Monthly Plot #####################################
def plot_raw_monthly_int(t2m_p, output_path):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t2m_p["valid_time"],
            y=t2m_p.values,
            mode="lines",
            name="Monthly",
            line=dict(
                color=PRIMARY,
                width=2,
            ),
            hovertemplate=
            "<b>%{x|%b %Y}</b><br>"
            "Temperature: %{y:.2f} °C"
            "<extra></extra>",
        )
    )

    apply_dashboard_theme(
        fig,
        "Monthly Temperature",
        "Temperature (°C)",
    )

    save_plotly(fig, output_path)
 
#### Yearly Plot #################################################
def plot_yearly_trend_int(t2m_p, output_path):

    yearly = t2m_p.resample(valid_time="1YE").mean()
    five_year = t2m_p.resample(valid_time="5YE").mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=yearly["valid_time"],
            y=yearly.values,
            mode="lines",
            name="1-Year Average",
            line=dict(
                color=GREY,
                width=1.5,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=five_year["valid_time"],
            y=five_year.values,
            mode="lines",
            name="5-Year Average",
            line=dict(
                color=PRIMARY,
                width=3,
            ),
        )
    )

    apply_dashboard_theme(
        fig,
        "Temperature Trend",
        "Temperature (°C)",
    )

    save_plotly(fig, output_path)

##### Anomaly Plot #######################################################
def plot_anomaly_trend_int(t2m_p, output_path):

    baseline = t2m_p.sel(valid_time=slice("1950", "1980"))

    climatology = baseline.groupby("valid_time.month").mean()

    anomaly = t2m_p.groupby("valid_time.month") - climatology

    yearly = anomaly.resample(valid_time="1YE").mean()

    five_year = anomaly.resample(valid_time="5YE").mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=yearly["valid_time"],
            y=yearly.values,
            mode="lines",
            name="1-Year Average",
            line=dict(
                color=GREY,
                width=1.5,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=five_year["valid_time"],
            y=five_year.values,
            mode="lines",
            name="5-Year Average",
            line=dict(
                color=ORANGE,
                width=3,
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=SECONDARY,
    )

    apply_dashboard_theme(
        fig,
        "Temperature Anomaly",
        "Anomaly (°C)",
    )

    save_plotly(fig, output_path)
    
################## STL decompostion #######################
def plot_stl_decomposition_int(t2m_p, output_path):

    series = t2m_p.to_series().dropna().sort_index()

    result = STL(
        series,
        period=12,
        robust=True,
    ).fit()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Original",
            "Trend",
            "Seasonal",
            "Residual",
        ),
    )

    colours = [
        BLACK,
        PRIMARY,
        TEAL,
        ORANGE,
    ]

    data = [
        series,
        result.trend,
        result.seasonal,
        result.resid,
    ]

    for i, (d, c) in enumerate(zip(data, colours), start=1):

        fig.add_trace(
            go.Scatter(
                x=d.index,
                y=d.values,
                mode="lines",
                line=dict(
                    color=c,
                    width=2,
                ),
            ),
            row=i,
            col=1,
        )

    fig.update_layout(

        template="plotly_white",

        title=dict(
            text="STL Decomposition",
            x=0.02,
            font=dict(
                size=22,
                family="Inter",
            ),
        ),

        height=850,

        hovermode="x unified",

        showlegend=False,

        paper_bgcolor="white",

        plot_bgcolor="white",

        font=dict(
            family="Inter",
            size=15,
        ),
    )

    for r in range(1, 5):

        fig.update_xaxes(
            showgrid=True,
            gridcolor=GRID,
            row=r,
            col=1,
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor=GRID,
            row=r,
            col=1,
        )

    save_plotly(fig, output_path)


