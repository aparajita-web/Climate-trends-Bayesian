from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys
APP_DIR = Path(__file__).resolve().parent.parent
APP_COMPONENTS = APP_DIR / "app_components"
sys.path.insert(0, str(APP_COMPONENTS))


CITIES = ["Paris", "Berlin", "Hanoi", "Kolkata"]

INSTRUMENTS = {
"CMB-Bharat[Satellite Instrument], PhD": [
("Model", "Ten multi-spectra cosmological simulations, spanning noise and model-complexity regimes."),
("Data", "Realistic Simulations "),
("Fit", "Wavelet-based and Bayesian parameter estimation to separate signal from noise."),
("Uncertainty", "Parameter-recovery bias quantified under noise and signal mixing; fed the systematic-error budget of two published papers."),
("Deliver", "Three PRD and MNRAS papers, including published D-statistics and Minkowski Functional analysis of polarised CMB data."),
],
"Planck and WMAP maps, PostDoc": [
("Model", "A non-linear model with a 7-dimensional posterior for multi-source satellite data."),
("Data", "Observational data from six legacy instruments including Planck and WMAP"),
("Fit", "5 multisource components through iterative regression"),
("Uncertainty", "Automated fit-quality and consistency checks across 18 channels, with five diagnostic plots per run."),
("Deliver", "A Git-versioned, python package and scientific report of the analysis"),
],
"ERA5 cities, Decadal": [
("Model", "A linear trend with AR(1) residuals, four parameters and explicit priors."),
("Data", "919 months of ERA5 2 m temperature for each of four cities, with no gaps."),
("Fit", "emcee MCMC with R-hat and effective sample-size checks."),
("Uncertainty", "68% credible intervals on every trend and 184 held-out months for three forecasters."),
("Deliver", "One result folder per city, MLflow tracking, and this Streamlit application. FastAPI and Docker are next."),
],
}

def load_trend(city_dir):
    path = city_dir / "warming_trend_per_decade.csv"


    if not path.exists():
        alt = city_dir / "warming_trend_per_decade(1).csv"
        if alt.exists():
            path = alt

    if not path.exists():
        return None

    row = pd.read_csv(path).iloc[0]

    return {
        "trend": float(row["Average_Trend"]),
        "minus": float(row["Average_Minus_Error"]),
        "plus": float(row["Average_Plus_Error"]),
    }


def build_city_trend_chart(output_dir):
    rows = []


    for city in CITIES:
        result = load_trend(output_dir / city)

        if result is not None:
            rows.append(
                {
                    "City": city,
                    "Trend": result["trend"],
                    "Lower": result["trend"] - result["minus"],
                    "Upper": result["trend"] + result["plus"],
                }
            )

    if not rows:
        return None

    df = pd.DataFrame(rows)

    fig = go.Figure()

    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Lower"], row["Upper"]],
                y=[row["City"], row["City"]],
                mode="lines",
                line=dict(color="#36ADA3", width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[row["Trend"]],
                y=[row["City"]],
                mode="markers+text",
                marker=dict(color="#36ADA3", size=9),
                text=[f'{row["Trend"]:.2f}'],
                textposition="middle right",
                textfont=dict(size=12),
                showlegend=False,
                hovertemplate=(
                    f'{row["City"]}<br>'
                    f'{row["Trend"]:.2f} °C/decade<br>'
                    f'68% interval: {row["Lower"]:.2f}–{row["Upper"]:.2f}'
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=230,
        margin=dict(l=70, r=60, t=10, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist Mono, monospace", size=11, color="#9FAAC1"),
        xaxis=dict(
            title="°C / decade",
            range=[0, 0.4],
            dtick=0.1,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            autorange="reversed",
        ),
    )

    return fig


def about_page(plot_dir):


    output_dir = plot_dir.parent

    # ==========================================================
    # WHO
    # ==========================================================

    st.markdown("---")

    left, right = st.columns([4, 8], gap="small")

    with left:

        with st.container(border=True):

            # Profile picture
            profile_path = Path(__file__).resolve().parent / "photo_profile.jpg"
            st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)    
            img_left, img_center, img_right = st.columns([0.2, 4.5, 0.2])

            with img_center:
                st.image(
                    str(profile_path),
                    width=1000,
                )

    '''with left:
        with st.container(border=True):
            st.markdown(
                "<div style='text-align:center;font-family:\"Bricolage Grotesque\";"
                "font-size:76px;font-weight:600;letter-spacing:-0.04em;'>AS</div>",
                unsafe_allow_html=True,
            )

        f1, f2 = st.columns(2)

        with f1:
            st.metric("Papers", "3")
            st.metric("Research teams", "6")

        with f2:
            st.metric("Citations", "~60")
            st.metric("Graduate teaching", "8 semesters")'''

    with right:
        st.title("I fit physical models to noisy data and say how sure I am.")

        st.markdown(
            """
            <div style="margin-bottom: 22px;">
                Hi there, and welcome to my portfolio.
            </div>
            I’m a computational physicist with 7 years of experience in developing numerical models, 
            building data analysis pipelines, and communicating scientific results. 
            I completed my PhD in Cosmology at IISER-TVM, India, followed by two years of postdoctoral research at CNRS in the same field. 
            I’m looking to bring this experience to applied research and industry. Decadal is a toolkit pointed 
            at  76 years of city temperature and I built it to showcase the skills I’ve developed throughout my research journey.
            """,
            unsafe_allow_html=True,
        )


       # ==========================================================
    # TOOLS AND METHODS — BELOW BOTH COLUMNS
    # ==========================================================

    tags = [
        "Bayesian inference, MCMC",
        "Uncertainty quantification",
        "Time series",
        "Wavelet and Fourier analysis",
        "Python",
        "NumPy, SciPy, pandas, xarray",
        "emcee, arviz",
        "statsmodels, scikit-learn",
        "pytest, Git, MLflow",
        "Fortran, SLURM, HPC",
    ]

    st.markdown(
        """
        <div style="
            display:inline-block;
            background:var(--teal);
            color:#000000;
            padding:7px 14px;
            border-radius:7px;
            font-family:'Bricolage Grotesque', sans-serif;
            font-size:16px;
            font-weight:600;
            margin-top:24px;
            margin-bottom:10px;
        ">
            Tools and methods
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(" · ".join(tags))


    st.divider()

    # ==========================================================
    # ONE LOOP, THREE INSTRUMENTS
    # ==========================================================

    st.header("One workflow, different problems")
    st.caption(
        "The work has had the same shape for seven years: write the model "
        "down, get the data, fit, quantify the uncertainty, and deliver something others can use "
        "What changes are the questions I am trying to answer."
    )

    '''stages = [
        ("Model", "what I think produces the signal"),
        ("Data", "what was actually measured"),
        ("Fit", "posterior, not point estimate"),
        ("Uncertainty", "validation and the error budget"),
        ("Deliver", "code other people run"),
    ]

    cols = st.columns(5)

    for col, (name, description) in zip(cols, stages):
        with col:
            st.markdown(f"**{name}**")
            st.caption(description)'''

    instrument = st.segmented_control(
        "Problems",
        list(INSTRUMENTS.keys()),
        default="ERA5 cities, Decadal",
        key="about_instrument",
    )

    instrument = instrument or "ERA5 cities, Decadal"

    cells = st.columns(5)

    for col, (stage, text) in zip(cells, INSTRUMENTS[instrument]):
        with col:
            st.markdown(f"**{stage}**")
            st.caption(text)

    st.divider()

    # ==========================================================
    # HOW DECadal IS BUILT
    # ==========================================================

    st.header("How Decadal is built")
    st.caption(
        "One YAML file per city goes in. A folder of results, an MLflow run "
        "and the pages you just used come out. The same chain ran for all four cities."
    )

    left, right = st.columns([7, 5], gap="large")

    with left:
        pipeline = [
            (
                "config/config_<City>.yaml",
                "one file per city",
                "City, variable, years, ERA5 grid box and output path.",
            ),
            (
                "main.py",
                "argparse, mlflow",
                "Runs the whole chain for one city and logs parameters, metrics and artefacts.",
            ),
            (
                "download.py",
                "cdsapi",
                "Pulls monthly 2 m temperature from the Copernicus Climate Data Store.",
            ),
            (
                "load.py",
                "xarray, pandas",
                "NetCDF in, one clean monthly series out.",
            ),
        ]

        for name, library, description in pipeline:
            with st.container(border=True):
                st.markdown(f"**`{name}`** · {library}")
                st.write(description)

        st.markdown("**Three analyses, same series**")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown("**EDA_plot.py**")
            st.caption(
                "Annual means, anomalies against 1950–1980 and STL decomposition."
            )

        with p2:
            st.markdown("**Model_A.py**")
            st.caption(
                "Bayesian AR(1) trend sampled with emcee. Posterior, R-hat and ESS to CSV."
            )

        with p3:
            st.markdown("**Model_C_ML.py**")
            st.caption(
                "Climatology, gradient boosting and SARIMA on a fixed time split."
            )

        with st.container(border=True):
            st.markdown("**`output/<City>/` · results")
            st.caption(
                "Everything the pages show, plus the MLflow run that records "
                "which code and configuration produced it."
            )

        with st.container(border=True):
            st.markdown("**`app/api.py` · planned**")
            st.caption(
                "FastAPI endpoint and Docker deployment. One JSON per city."
            )

    with right:
        st.code(
            """Climate-trends-Bayesian/
    ```

    ├── config/
    ├── main.py
    ├── src/climate_trends/
    │   ├── download.py
    │   ├── load.py
    │   ├── EDA_plot.py
    │   ├── Model_A.py
    │   └── Model_C_ML.py
    ├── notebooks/
    ├── data/raw/
    ├── data/processed/
    ├── output/<City>/
    ├── tests/
    ├── app/
    ├── pyproject.toml
    └── Makefile""",
    language="text",
    )


    st.divider()

    # ==========================================================
    # FIVE DECISIONS
    # ==========================================================

    st.header("Five decisions, and what they cost")
    st.caption(
        "Each one closed off an alternative. The alternative is named, "
        "and so is the evidence that the choice was worth it."
    )

    decisions = [
        (
            "AR(1) residuals, not independent years",
            "Instead of ordinary least squares on the annual means.",
            "Warm years follow warm years. Treating them as independent gives a trend interval narrower than the data justify.",
            "φ = 0.20 to 0.30",
            "year-to-year autocorrelation, all four cities",
        ),
        (
            "emcee with the likelihood written out",
            "Instead of a probabilistic programming language with NUTS.",
            "The whole model is a few dozen lines I can defend line by line. Convergence is checked with ArviZ on every run.",
            "R-hat ≤ 1.006",
            "annual-mean model, all four cities",
        ),
        (
            "Annual means for the trend, monthly data for the forecast",
            "Instead of one harmonic model on the monthly series.",
            "The trend question uses a deseasonalised annual series. The forecast keeps the seasonal cycle.",
            "76 years, 919 months",
            "two data paths per city",
        ),
        (
            "A climatology baseline every model has to beat",
            "Instead of comparing the two learned models with each other.",
            "The median August is a forecast too. The baseline tells us whether a learned model is actually useful.",
            "MAE benchmark",
            "forecast baseline",
        ),
        (
            "A fixed time split, no shuffling",
            "Instead of k-fold cross-validation.",
            "Nothing from the future leaks into training, so the score means what a forecaster's score should mean.",
            "735 train, 184 test",
            "cut at April 2011",
        ),
    ]

    for title, alternative, explanation, value, note in decisions:
        left, right = st.columns([5, 7], gap="large")

        with left:
            st.subheader(title)
            st.caption(alternative)

        with right:
            st.write(explanation)
            st.caption(f"**{value}** · {note}")

        st.divider()

    # ==========================================================
    # LESSONS
    # ==========================================================

    st.header("What the numbers taught me")

    lessons = [
        (
            "Autocorrelation is not a detail",
            "All four cities show positive year-to-year autocorrelation. "
            "The AR(1) term is the difference between an honest trend interval "
            "and an optimistic one.",
        ),
        (
            "The baseline is the model to beat, not a formality",
            "SARIMA beat the median August in every city. Gradient boosting "
            "did not in Berlin. A model that loses to a lookup table is a "
            "result worth reporting, not hiding.",
        ),
        (
            "Convergence is a number, not a feeling",
            "The annual-mean model converged in all four cities. The annual-maximum "
            "model did not. Its posterior diagnostics belong in the analysis rather "
            "than being hidden from the reader.",
        ),
        (
            "One method, four different answers",
            "The same Bayesian machinery produces different warming rates in "
            "different climates. The method did not change between cities. "
            "The climate did.",
        ),
    ]

    left, right = st.columns([7, 5], gap="large")

    with left:
        for title, text in lessons:
            st.subheader(title)
            st.write(text)
            st.divider()

    with right:
        with st.container(border=True):
            st.subheader("Warming per decade, four cities")
            st.caption(
                "Posterior median with the 68% credible interval. "
                "Bayesian AR(1) trend on annual means."
            )

            fig = build_city_trend_chart(output_dir)

            if fig is not None:
                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={"displayModeBar": False},
                )
            else:
                st.info("Warming-trend CSV files were not found.")

            st.caption(
                "The same numbers are loaded by the Analysis page."
            )

    st.divider()

    # ==========================================================
    # NOT FINISHED
    # ==========================================================

    st.header("Not finished")
    st.caption("What a reviewer would find, said first.")

    open_items = [
        (
            "The annual-maximum model",
            "Its σ posterior collapsed and the chains did not mix. "
            "The likelihood for the extreme series needs a fix or the toggle comes out.",
        ),
        (
            "Interval coverage",
            "Climatology and SARIMA report 95% intervals. "
            "The gradient-boosting interval's coverage is not documented yet.",
        ),
        (
            "Deployment",
            "The FastAPI endpoint, Docker image and live URL are not finished yet. "
            "The JSON contract the pages need is written.",
        ),
        (
            "More cities",
            "A fifth configuration already exists. Each new city needs its YAML file, "
            "Copernicus credentials and one pipeline run.",
        ),
    ]

    c1, c2 = st.columns(2, gap="large")

    for i, (title, text) in enumerate(open_items):
        col = c1 if i % 2 == 0 else c2

        with col:
            with st.container(border=True):
                st.subheader(title)
                st.write(text)

    # ==========================================================
    # FOOTER
    # ==========================================================

    st.divider()

    left, right = st.columns([7, 5])

    with left:
        st.caption(
            "Aparajita Sen · Grenoble, France"
        )

    with right:
        st.markdown(
            "[GitHub](https://github.com/aparajita-web/Climate-trends-Bayesian)"
        )


