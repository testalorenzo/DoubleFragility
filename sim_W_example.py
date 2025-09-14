import numpy as np
import pandas as pd
import warnings
from statsmodels.api import Logit, OLS, add_constant
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress convergence warnings from statsmodels for a cleaner simulation output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def generate_data(n):
    """
    Generates a dataset based on the simulation parameters in the paper.

    Args:
        n (int): The sample size.

    Returns:
        pd.DataFrame: A DataFrame containing the generated data.
    """
    z = np.random.normal(0, 1, size=(n, 4))
    x1 = np.exp(z[:, 0] / 2)
    x2 = z[:, 1] / (1 + np.exp(z[:, 0])) + 10
    x3 = (z[:, 0] * z[:, 2] / 25 + 0.6)**3
    x4 = (z[:, 1] + z[:, 3] + 20)**2

    y = 210 + 27.4 * z[:, 0] + 13.7 * z[:, 1] + 13.7 * z[:, 2] + 13.7 * z[:, 3] + np.random.normal(0, 1, n)

    true_propensity_linear = -z[:, 0] + 0.5 * z[:, 1] - 0.25 * z[:, 2] - 0.1 * z[:, 3]
    true_propensity = 1 / (1 + np.exp(-true_propensity_linear))

    t_indicator = np.random.binomial(1, true_propensity)

    return pd.DataFrame({
        'z1': z[:, 0], 'z2': z[:, 1], 'z3': z[:, 2], 'z4': z[:, 3],
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
        'y': y, 't': t_indicator
    })

def estimate_propensity_scores(data, use_correct_model):
    """
    Estimates propensity scores using a logistic regression model.
    """
    X = add_constant(data[['z1', 'z2', 'z3', 'z4']]) if use_correct_model else add_constant(data[['x1', 'x2', 'x3', 'x4']])
    try:
        logit_model = Logit(data['t'], X).fit(disp=0)
        return logit_model.predict(X)
    except Exception:
        # In rare cases of perfect separation, return a default
        return pd.Series(np.full(len(data), 0.5), index=data.index)


def get_y_model_predictions(data, use_correct_model):
    """
    Fits a y-model on respondents and predicts for the entire sample.
    """
    respondents = data['t'] == 1
    if respondents.sum() < 2: # Not enough data to fit model
        m_hat_all = pd.Series(data.loc[respondents, 'y'].mean(), index=data.index).fillna(np.mean(data['y']))
        eps_hat_resp = pd.Series(0, index=respondents[respondents].index)
        return m_hat_all, eps_hat_resp

    X_resp = add_constant(data.loc[respondents, ['z1', 'z2', 'z3', 'z4']]) if use_correct_model else add_constant(data.loc[respondents, ['x1', 'x2', 'x3', 'x4']])
    X_all = add_constant(data[['z1', 'z2', 'z3', 'z4']]) if use_correct_model else add_constant(data[['x1', 'x2', 'x3', 'x4']])

    ols_model = OLS(data.loc[respondents, 'y'], X_resp).fit()
    m_hat_all = ols_model.predict(X_all)
    eps_hat_resp = data.loc[respondents, 'y'] - ols_model.predict(X_resp)
    return m_hat_all, eps_hat_resp

def ipw_estimator(data, pi_hat, pop_weighting):
    """Calculates the Inverse-Propensity Weighted (IPW) estimate."""
    respondents = data['t'] == 1
    if respondents.sum() == 0:
        return np.nan
    pi_hat_resp = pi_hat[respondents].clip(lower=1e-6) # Clip weights to avoid division by zero

    if pop_weighting:
        weights = 1 / pi_hat_resp
        return np.sum(weights * data.loc[respondents, 'y']) / np.sum(weights)
    else:  # NR weighting
        weights = (1 - pi_hat_resp) / pi_hat_resp
        mu0_hat = np.sum(weights * data.loc[respondents, 'y']) / np.sum(weights)
        r1_hat = np.mean(data['t'])
        y_bar_1 = data.loc[respondents, 'y'].mean()
        return r1_hat * y_bar_1 + (1 - r1_hat) * mu0_hat

def bc_ols_estimator(data, m_hat_all, eps_hat_resp, pi_hat):
    """Calculates the bias-corrected regression estimate (AIPW)."""
    respondents = data['t'] == 1
    n = len(respondents)
    if respondents.sum() == 0:
        return np.nan
    mu_ols = m_hat_all.mean()
    pi_hat_resp = pi_hat[respondents].clip(lower=1e-6)
    bias_correction = np.sum((data.loc[respondents, 't'] / pi_hat_resp) * eps_hat_resp) / n
    return mu_ols + bias_correction

def clipped_dr_estimator(data, pi_hat, m_hat_all):
    """
    Calculates the clipped doubly robust estimate.
    This is a simplified version of the bias-corrected OLS estimator.
    """
    respondents = data['t'] == 1
    n = len(respondents)
    if respondents.sum() == 0:
        return np.nan
    outcome_reg = m_hat_all.mean()
    pi_hat_resp = pi_hat[respondents].clip(lower=1e-6)
    weights = 1 / pi_hat_resp
    # Hajek style
    ipw = np.sum(weights * data.loc[respondents, 'y']) / np.sum(weights)
    correction = np.sum(weights * m_hat_all[respondents]) / np.sum(weights)

    # Standard
    ipw = np.sum(weights * data.loc[respondents, 'y']) / n
    correction = np.sum(weights * m_hat_all[respondents]) / n

    # Ensure correction is within outcome_reg and ipw bounds
    correction = np.clip(correction, min(outcome_reg, ipw), max(outcome_reg, ipw))
    # Return the clipped doubly robust estimate
    return outcome_reg + ipw - correction

def set_size(width, fraction=1, subplots=(3, 3)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

if __name__ == '__main__':

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=0.6)
    # sns.set_context("notebook")
    width = 426.79135  # width in pts

    # Set a seed for reproducibility
    np.random.seed(12)

    n_sims = 1
    ns = [100, 200, 1000]
    true_mu = 210.0
    n_simulations = 10000
    alpha = 0.05
 
    use_correct_model_mu = True
    use_correct_model_pi = True

    axd = plt.figure(figsize=set_size(width, subplots=(1,3))).subplot_mosaic(
            [['100', '200', '1000']],
            sharey=True
            )

    for i in range(n_sims):
        for n in ns:
            data = generate_data(n)

            pi_correct = estimate_propensity_scores(data, use_correct_model=use_correct_model_pi)
            m_correct, eps_correct = get_y_model_predictions(data, use_correct_model=use_correct_model_mu)

            sim_results = [
                ipw_estimator(data, pi_correct, pop_weighting=True),
                m_correct.mean(),
                bc_ols_estimator(data, m_correct, eps_correct, pi_correct),
                clipped_dr_estimator(data, pi_correct, m_correct)
            ]

            pi_correct = pi_correct.clip(lower=1e-6, upper=1-1e-6)  # Avoid extreme weights
            
            # --- Influence Functions (Needed for both CIs) ---
            phi_or = m_correct - sim_results[1]
            phi_ipw = (data['t'] * data['y'] / pi_correct) - sim_results[0]
            phi_correction = (data['t'] * m_correct / pi_correct) - np.mean((data['t'] * m_correct / pi_correct))
            
            # --- CI for Standard DR Estimator (Normal Approximation) ---
            # Step 1: DR Influence Function
            phi_dr = phi_or + phi_ipw - phi_correction
            
            # Step 2: Calculate Standard Error from Influence Function
            var_dr_infl = np.mean(phi_dr**2)
            se_dr = np.sqrt(var_dr_infl / n)
            
            # --- CI for Clipped Estimator (Parametric Bootstrap) ---
            # Step 1: Covariance Matrix
            phi_matrix = np.vstack([phi_or, phi_ipw, phi_correction]).T
            cov_matrix = (phi_matrix.T @ phi_matrix) / n
            
            # Step 2: Simulate W
            Z_samples = np.random.multivariate_normal([0, 0, 0], cov_matrix, size=n_simulations)
            Z_or, Z_ipw, Z_c = Z_samples[:, 0], Z_samples[:, 1], Z_samples[:, 2]
            W_samples = Z_or + Z_ipw - np.clip(Z_c, np.minimum(Z_or, Z_ipw), np.maximum(Z_or, Z_ipw))

            # Plot
            ax = axd[str(n)]
            sns.histplot(W_samples, bins=30, stat='density', kde=False, ax=ax)   
            mu_w, std_w = 0, var_dr_infl**0.5
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = np.exp(-0.5 * ((x - mu_w) / std_w)**2) / (std_w * np.sqrt(2 * np.pi))
            ax.plot(x, p, color='black', linestyle='--', label='Normal Density')
            ax.set_title(f'n={n}')
            if n == ns[0]:
                ax.set_ylabel('Density')
            ax.set_xlabel('W')
            ax.tick_params(pad=-3)
            ax.label_outer()
    plt.savefig('sim_W_example.pdf',  bbox_inches='tight')