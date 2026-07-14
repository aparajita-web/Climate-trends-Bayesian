

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import xarray
import xarray as xr
import pandas as pd
import emcee


def climate_model(theta, t):

    beta0, beta1, psi, sigma = theta

    phi=np.tanh(psi)
    model = (beta0 + beta1 * t )

    return model


def log_prior(theta):

    beta0, beta1, psi, sigma = theta

    # sigma must remain positive
    if sigma <= 0:
        return -np.inf

    lp = 0.0

    # Intercept prior
    lp += -0.5 * (beta0 / 10.0)**2

    # Trend prior
    # Assumes standardized time axis
    lp += -0.5 * (beta1 / 0.3)**2

   
  
    # prior on tanh(psi parameters)
    lp += -0.5 * (psi / 0.3)**2

    # Weakly informative prior on sigma
    lp += -0.5 * (sigma / 2.0)**2

    return lp
    

##########################################
# AR(1) LOG-LIKELIHOOD
############################################

def log_likelihood(theta, t, y):

    beta0, beta1, psi, sigma = theta
    phi=np.tanh(psi)
    
    # Deterministic model
    model = climate_model(theta, t)

    # Residuals
    r = y - model

    # AR(1) residuals
    ar_resid = r[1:] - phi * r[:-1]

    loglike = -0.5 * np.sum(
        (ar_resid / sigma)**2
        + np.log(2*np.pi*sigma**2)
    )

    return loglike

# ==========================================================
# POSTERIOR
# ==========================================================

def log_probability(theta, t, y):

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, t, y)


#####################################################################
# Climate model with harmonics
#####################################################################




def climate_model_harmonics(theta, t_trend, t_harmonic, nharm):

    beta0 = theta[0]
    beta1 = theta[1]

    # Harmonic coefficients
    a = theta[2:2+nharm]
    b = theta[2+nharm:2+2*nharm]

    psi   = theta[-2]
    sigma = theta[-1]

    phi = np.tanh(psi)

    
    model = beta0 + beta1 * t_trend

    # Harmonic seasonal terms
    for k in range(1, nharm + 1):

        model += (
            a[k-1] * np.cos(2*np.pi*k*t_harmonic)
            + b[k-1] * np.sin(2*np.pi*k*t_harmonic)
        )

    return model


# ==========================================================
# PRIORS
# ==========================================================

def log_prior_harmonics(theta, nharm):

    beta0 = theta[0]
    beta1 = theta[1]

    a = theta[2:2+nharm]
    b = theta[2+nharm:2+2*nharm]

    psi   = theta[-2]
    sigma = theta[-1]

    # sigma must be positive
    if sigma <= 0:
        return -np.inf

    lp = 0.0

    # Intercept prior
    lp += -0.5 * (beta0 / 10.0)**2

    # Trend prior
    lp += -0.5 * (beta1 / 0.3)**2

    # Harmonic priors
    for ak in a:
        lp += -0.5 * (ak / 5.0)**2

    for bk in b:
        lp += -0.5 * (bk / 5.0)**2

    # AR1 tanh(psi) prior
    lp += -0.5 * (psi / 0.3)**2

    # Noise prior
    lp += -0.5 * (sigma / 2.0)**2

    return lp


# ==========================================================
# AR(1) LOG-LIKELIHOOD
# ==========================================================

def log_likelihood_harmonics(theta, t_trend, t_harmonic, y, nharm):

    psi   = theta[-2]
    sigma = theta[-1]

    phi = np.tanh(psi)

    # Deterministic model
    model = climate_model_harmonics(
        theta,
        t_trend,
        t_harmonic,
        nharm
    )

    # Residuals
    r = y - model

    # AR1 residuals
    ar_resid = r[1:] - phi * r[:-1]

    loglike = -0.5 * np.sum(
        (ar_resid / sigma)**2
        + np.log(2*np.pi*sigma**2)
    )

    return loglike


# ==========================================================
# POSTERIOR
# ==========================================================

def log_probability_harmonics(theta,
                    t_trend,
                    t_harmonic,
                    y,
                    nharm):

    lp = log_prior_harmonics(theta, nharm)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood_harmonics(
        theta,
        t_trend,
        t_harmonic,
        y,
        nharm
    )