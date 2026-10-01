import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pathlib import Path
import sys
sys.path.append("../src/climate_trends/")

from Model_C_ML import climatology, gradient_boost, SARIMA


# ==========================================================
# FIXTURES
# ==========================================================

@pytest.fixture
def temperature_data():
    """
    Synthetic monthly temperature data covering 10 years.

    The data contain:
    - a simple warming trend
    - seasonal variation
    - small deterministic variation
    """

    dates = pd.date_range(
        start="2015-01-01",
        periods=120,
        freq="MS",
    )

    months = np.arange(1, 121)

    seasonal = 5 * np.sin(2 * np.pi * (months % 12) / 12)
    trend = 0.02 * months
    noise = 0.2 * np.sin(months)

    temperatures = 15 + trend + seasonal + noise

    return xr.DataArray(
        temperatures,
        coords={"valid_time": dates},
        dims=["valid_time"],
        name="t2m",
    )


@pytest.fixture
def output_path(tmp_path):
    """
    Temporary output directory required by the model functions.
    """

    return str(tmp_path) + "/"


# ==========================================================
# CLIMATOLOGY TEST
# ==========================================================

def test_climatology_returns_forecast(
    temperature_data,
    output_path,
):
    """
    Check that the climatology model returns a finite
    next-month temperature forecast.
    """

    result = climatology(
        temperature_data,
        output_path,
    )

    assert np.isfinite(result)


# ==========================================================
# GRADIENT BOOSTING TEST
# ==========================================================

def test_gradient_boost_returns_forecast(
    temperature_data,
    output_path,
):
    """
    Check that Gradient Boosting returns a finite
    next-month temperature forecast.
    """

    result = gradient_boost(
        temperature_data,
        output_path,
    )

    assert np.isfinite(result)


# ==========================================================
# SARIMA TEST
# ==========================================================

def test_sarima_returns_forecast(
    temperature_data,
    output_path,
):
    """
    Check that SARIMA returns a finite
    next-month temperature forecast.
    """

    result = SARIMA(
        temperature_data,
        output_path,
    )

    assert np.isfinite(result)



