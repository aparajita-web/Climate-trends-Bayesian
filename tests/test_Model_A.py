import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.append("../src/climate_trends/")

from Model_A import climate_model, log_prior, log_likelihood, log_probability



# ==========================================================
# FIXTURES
# ==========================================================

@pytest.fixture
def theta():
    """
    Valid parameter vector for Model A:

    beta0 = intercept
    beta1 = trend
    psi   = AR(1) parameter before tanh transformation
    sigma = noise standard deviation
    """
    return np.array([10.0, 0.2, 0.3, 1.0])


@pytest.fixture
def time():
    """
    Small standardized time array for testing.
    """
    return np.array([0.0, 1.0, 2.0, 3.0])


@pytest.fixture
def temperature_data():
    """
    Small synthetic temperature dataset.
    """
    return np.array([10.1, 10.4, 10.2, 10.7])


# ==========================================================
# CLIMATE MODEL TESTS
# ==========================================================

def test_climate_model_returns_expected_linear_trend(theta, time):
    """
    Check that climate_model() returns:

        beta0 + beta1 * t
    """

    result = climate_model(theta, time)

    expected = theta[0] + theta[1] * time

    np.testing.assert_allclose(result, expected)


def test_climate_model_output_shape(theta, time):
    """
    Check that the model output has the same shape as the
    input time array.
    """

    result = climate_model(theta, time)

    assert result.shape == time.shape


# ==========================================================
# PRIOR TESTS
# ==========================================================

def test_log_prior_rejects_non_positive_sigma(theta):
    """
    sigma must be strictly positive.
    """

    theta_invalid = theta.copy()
    theta_invalid[3] = 0.0

    assert log_prior(theta_invalid) == -np.inf


@pytest.mark.parametrize("sigma", [-1.0, 0.0])
def test_log_prior_rejects_invalid_sigma_values(theta, sigma):
    """
    Check that both negative and zero sigma values are rejected.
    """

    theta_invalid = theta.copy()
    theta_invalid[3] = sigma

    assert log_prior(theta_invalid) == -np.inf


def test_log_prior_returns_finite_value_for_valid_parameters(theta):
    """
    A valid parameter vector should produce a finite log prior.
    """

    result = log_prior(theta)

    assert np.isfinite(result)


# ==========================================================
# LIKELIHOOD TESTS
# ==========================================================

def test_log_likelihood_returns_finite_value(
    theta,
    time,
    temperature_data,
):
    """
    Check that the AR(1) log-likelihood returns a finite value
    for valid parameters and data.
    """

    result = log_likelihood(
        theta,
        time,
        temperature_data,
    )

    assert np.isfinite(result)


def test_log_likelihood_changes_with_sigma(
    theta,
    time,
    temperature_data,
):
    """
    Changing sigma should change the likelihood because sigma
    controls the assumed residual noise level.
    """

    theta_sigma_1 = theta.copy()
    theta_sigma_1[3] = 1.0

    theta_sigma_2 = theta.copy()
    theta_sigma_2[3] = 2.0

    likelihood_1 = log_likelihood(
        theta_sigma_1,
        time,
        temperature_data,
    )

    likelihood_2 = log_likelihood(
        theta_sigma_2,
        time,
        temperature_data,
    )

    assert likelihood_1 != likelihood_2


# ==========================================================
# POSTERIOR TEST
# ==========================================================

def test_log_probability_combines_prior_and_likelihood(
    theta,
    time,
    temperature_data,
):
    """
    Check that:

        log_probability =
            log_prior + log_likelihood

    for valid parameters.
    """

    prior = log_prior(theta)

    likelihood = log_likelihood(
        theta,
        time,
        temperature_data,
    )

    posterior = log_probability(
        theta,
        time,
        temperature_data,
    )

    expected = prior + likelihood

    np.testing.assert_allclose(
        posterior,
        expected,
    )

