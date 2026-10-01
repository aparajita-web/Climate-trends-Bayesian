# Temperature Warming Trend Analysis

A Python package for analysing long-term urban warming trends using ERA5 temperature data. Two types of models are used: Bayesian inference and state space model.

## Overview

This project estimates warming trends for individual cities using historical temperature records from 1950–2025 from the ERA5 reanalysis dataset. Given the geographic coordinates of a city, the workflow automatically retrieves the corresponding temperature data, performs exploratory data analysis, and fits statistical models to quantify long-term warming trends and their uncertainties.

The main goal is not only to estimate a warming rate, but also to provide robust uncertainty estimates.

## Workflow

The analysis consists of four main steps:

### 1. Data Acquisition

Temperature data are extracted from ERA5 at the specified city coordinates. Annual mean temperatures are computed from the monthly data to create a long-term temperature time series.

### 2. Exploratory Data Analysis

The time series is explored through visualisation and descriptive statistics, including:

* Annual temperature evolution
* Distribution of temperatures
* STL decomposition to separate seasonality
* Preliminary trend inspection

### 3. Bayesian Trend Estimation

A Bayesian parametric model is fitted to the temperature time series to estimate:

* Long-term warming trend
* Model parameters and uncertainties
* Credible intervals for the warming rate

The model uses an AR(1) residual structure to account for temporal correlation in the temperature time series.

Posterior distributions are sampled using Markov Chain Monte Carlo (MCMC) methods, providing a probabilistic estimate of the warming signal.

### 4. Next-Month Temperature Forecasting

The project predicts the next month's mean 2 m air temperature using three approaches:

* **Climatology** — a historical monthly baseline
* **SARIMA** — a seasonal autoregressive time-series model
* **Gradient Boosting** — a machine-learning model using lagged and seasonal information

Model performance is evaluated on held-out historical months using metrics such as MAE, RMSE and R².

## Models

### Bayesian Trend Model

The long-term warming analysis uses a Bayesian parametric model with correlated residuals. MCMC sampling is used to estimate posterior distributions for the model parameters and quantify uncertainty in the inferred warming trend.

### Climatology

The climatological forecast provides a baseline estimate based on historical monthly temperatures.

### SARIMA

A seasonal autoregressive integrated moving-average model is used to capture temporal dependence and seasonal structure in the monthly temperature series.

### Gradient Boosting

A gradient-boosting machine-learning model is used for next-month forecasting using temporal, lagged and seasonal features.

## Outputs

For each city, the workflow generates processed temperature data, exploratory plots, Bayesian posterior results, warming-trend estimates, model evaluation metrics, and next-month forecasts.

Results are stored in city-specific output directories and can be directly used by the Streamlit dashboard.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring the temperature record, visualising warming trends, comparing forecasting models, and displaying the next-month temperature prediction together with uncertainty information.

The dashboard currently supports:

* Paris
* Berlin
* Kolkata
* Hanoi

## Example Application

The workflow can be applied to any city or region for which geographic coordinates are available in a YAML configuration file.

Example run:

```bash
python main.py config_paris
```

## Technologies

The project uses:

* Python
* NumPy
* pandas
* SciPy
* xarray
* statsmodels
* scikit-learn
* emcee
* ArviZ
* Plotly
* Streamlit

The workflow is configuration-driven, with city-specific geographic coordinates and analysis settings defined through YAML configuration files.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd Climate-trends-Bayesian
```

Create the required Python environment and install the project dependencies according to the environment or requirements file provided in the repository.

ERA5 data acquisition requires access to the Copernicus Climate Data Store (CDS) API and the corresponding API credentials.

## Repository Structure

```text
Climate-trends-Bayesian/
│
├── src/            Source code for models and plotting
├── notebooks/      Analysis notebooks
├── data/           Input and processed data
├── output/         City-specific analysis and forecasting results
├── app/            Streamlit dashboard
└── tests/          Unit tests
```

## References

* Castilla Valdez et al. (2026) — *Long-Term Performance Assessment of Statistical and Machine Learning Models for Temperature Forecasting in Gulf of Mexico and Atlantic-Transition Coastal Cities*
* Dashboard inspiration: https://dashboard.theclimatebrink.com/#global

