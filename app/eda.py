#from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components


## Helper function

def display_html_plot(plot_path, height=550):

    with open(plot_path, "r", encoding="utf-8") as f:
        html = f.read()

    components.html(
        html,
        height=height,
        scrolling=False,
    )
    
 
### Define the EDA page layout ########
def eda_page(plot_dir):

    st.header("Exploratory Data Analysis")

    st.write(
        """
        Explore the temporal evolution and variability of the ERA5 monthly
        2 m air temperature dataset. Select one of the visualizations below.
        """
    )

    plot_choice = st.selectbox(
        "Choose a visualization",
        [
            "Monthly Temperature",
            "Temperature Trend",
            "Temperature Anomaly",
            "STL Decomposition",
        ],
    )

    #plot_dir = Path("plots") / city.lower()

    descriptions = {

        "Monthly Temperature":
        "Monthly mean 2 m air temperature from the ERA5 reanalysis dataset.",

        "Temperature Trend":
        "Comparison of annual and 5-year averaged temperatures highlighting long-term warming.",

        "Temperature Anomaly":
        "Temperature anomalies relative to the 1950–1980 climatological baseline.",

        "STL Decomposition":
        "Seasonal-Trend decomposition separates the temperature signal into trend, seasonal and residual components.",
    }

    if plot_choice == "Monthly Temperature":

        display_html_plot(
            plot_dir / "raw_monthly.html",
            height=550,
        )

    elif plot_choice == "Temperature Trend":

        display_html_plot(
            plot_dir / "yearly_average.html",
            height=550,
        )

    elif plot_choice == "Temperature Anomaly":

        display_html_plot(
            plot_dir / "anomaly_yearly.html",
            height=550,
        )

    elif plot_choice == "STL Decomposition":

        display_html_plot(
            plot_dir / "stl_decomposition.html",
            height=900,
        )

    st.caption(descriptions[plot_choice])

