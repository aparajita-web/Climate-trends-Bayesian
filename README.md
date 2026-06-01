# Temperature Warming Trend Analysis

A Python package for analysing long-term urban warming trends using ERA5 temperature data. Two types of models are used: Bayesian inference and state space model.

## Overview

This project estimates warming trends for individual cities using historical temperature records from 1950-2025 from the ERA5 reanalysis dataset. Given the geographic coordinates of a city, the workflow automatically retrieves the corresponding temperature data, performs exploratory data analysis, and fits statistical models to quantify long-term warming trends and their uncertainties.

The main goal is not only to estimate a warming rate, but also to provide robust uncertainty estimates.

## Workflow

The analysis consists of three main steps:

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

Posterior distributions are sampled using Markov Chain Monte Carlo (MCMC) methods, providing a probabilistic estimate of the warming signal.

## Example Application

The workflow can be applied to any city/region for which geographic coordinates are available. Example analyses include:

* Paris


## Installation

....

## Repository Structure


src/            Source code for Models and plotting
notebooks/      Analysis notebooks
data/           Input and processed data

tests/          Unit tests
```

