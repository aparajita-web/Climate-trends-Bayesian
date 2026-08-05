

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import xarray
import xarray as xr
import pandas as pd
import emcee
import corner

import arviz as az


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


def run_Bayesian(time_norm,t2m_trend_values,output_dir,run_name):
    ## Define the number of walkers and initial parameters
    ndim = 4
    nwalkers = 32

    initial = np.array([
        t2m_trend_values.mean(),   # beta0
        0.01,          # beta1
        0.5,           # psi
        2.0            # sigma
    ])

    # Initialize walkers
    pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

    ## Run the Sampler

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(time_norm, t2m_trend_values)
    )

    sampler.run_mcmc(pos, 10000, progress=True)

    ## Extract samples

    samples = sampler.get_chain(
        discard=2000,
        thin=10,
        flat=True
    )

    ## Parameter estimates from chain

    labels = [
        "beta0",
        "beta1",
        "psi",
        "sigma"
    ]
    

    posterior_results = []

    for i, label in enumerate(labels):

        q16, q50, q84 = np.percentile(
            samples[:, i],
            [16, 50, 84]
        )

        posterior_results.append({
            "Parameter": label,
            "Median": q50,
            "Lower_68CI": q16,
            "Upper_68CI": q84,
            "Minus_Error": q50 - q16,
            "Plus_Error": q84 - q50
        })

    # Convert to DataFrame
    posterior_df = pd.DataFrame(posterior_results)

  

    # Save
    csv_filename=output_dir+run_name+"posterior_values.csv"
    posterior_df.to_csv(
        csv_filename,
        index=False
    )
    #### Plot Sample chains ########################
    schainplot_name=output_dir+run_name+"sample_chain_plot.png"
    chains = sampler.get_chain()

    fig, axes = plt.subplots(ndim, figsize=(10,12), sharex=True)

    for i in range(ndim):

        axes[i].plot(chains[:, :, i], alpha=0.5)

        axes[i].set_ylabel(labels[i])

    axes[-1].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(schainplot_name)
    ##### Convergence Checks #########################
    idata = az.from_emcee(
        sampler,
        var_names=[
            "beta0",
            "beta1",
            "phi",
            "sigma"
        ]
    )

    # Compute diagnostics
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    print(type(rhat))
    print(rhat)
    print(rhat.dims)
    print(rhat.data_vars)
 	
    # Convert to DataFrames
    #rhat_df = rhat.to_dataframe().reset_index()
    #ess_df = ess.to_dataframe().reset_index()
    
    def dataset_to_dataframe(ds, value_name):
	    return pd.DataFrame({
		"Parameter": list(ds.data_vars),
		value_name: [ds[var].item() for var in ds.data_vars],
	    })

    rhat_df = dataset_to_dataframe(az.rhat(idata), "Rhat")
    ess_df = dataset_to_dataframe(az.ess(idata), "ESS")
    rhat_name=output_dir+run_name+"rhat.csv"
    ## Save in .csv
    rhat_df.to_csv(
        rhat_name,
        index=False
    )
    print(rhat_name)
    print(rhat_df)

    ess_df.to_csv(
        output_dir+run_name+"ess.csv",
        index=False
    )
    
    #### Corner Plot ##############################
    cornerplot_name=output_dir+run_name+"corener_plot.png"
    flat_samples = sampler.get_chain(discard=2000, thin=10, flat=True)

    # Convert psi -> phi
    flat_samples[:, 2] = np.tanh(flat_samples[:, 2])

    # Parameter labels
    labels = [r"$\beta_0$",
            r"$\beta_1$",
            
            r"$\phi$",
            r"$\sigma$"]

    # Corner plot
    fig = corner.corner(
        flat_samples,
        labels=labels,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        truths=None
    )

    plt.savefig(cornerplot_name)
