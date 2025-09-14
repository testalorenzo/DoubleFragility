import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
import warnings
from statsmodels.api import Logit, OLS, WLS, add_constant
import matplotlib.pyplot as plt

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


def run_simulation(n, n_sims=1000, true_mu=210.0, n_simulations=10000, alpha=0.05, use_correct_model_mu=False, use_correct_model_pi=False):
    """Runs the full simulation study."""
    estimator_names = ['IPW (C-Pi) lower', 'IPW (C-Pi) upper', 'width IPW (C-Pi)', 'covered IPW (C-Pi)',
                       'OR (C-Y) lower', 'OR (C-Y) upper', 'width OR (C-Y)', 'covered OR (C-Y)',
                       'BC-OLS (C-Pi, C-Y) lower', 'BC-OLS (C-Pi, C-Y) upper', 'width BC-OLS (C-Pi, C-Y)', 'covered BC-OLS (C-Pi, C-Y)',
                       'DR-Clipped (C-Pi, C-Y) lower', 'DR-Clipped (C-Pi, C-Y) upper', 'width DR-Clipped (C-Pi, C-Y)', 'covered DR-Clipped (C-Pi, C-Y)']
    results = []
    
    for i in range(n_sims):
        if (i + 1) % 100 == 0: print(f"  Sim {i+1}/{n_sims}")
        data = generate_data(n)

        pi_correct = estimate_propensity_scores(data, use_correct_model=use_correct_model_pi)
        m_correct, eps_correct = get_y_model_predictions(data, use_correct_model=use_correct_model_mu)

        sim_results = [
            ipw_estimator(data, pi_correct, pop_weighting=True),
            m_correct.mean(),
            bc_ols_estimator(data, m_correct, eps_correct, pi_correct),
            clipped_dr_estimator(data, pi_correct, m_correct)
        ]

        # Confidence intervals

        pi_correct = pi_correct.clip(lower=1e-6, upper=1-1e-6)  # Avoid extreme weights
        
        # --- Influence Functions (Needed for both CIs) ---
        phi_or = m_correct - sim_results[1]
        phi_ipw = (data['t'] * data['y'] / pi_correct) - sim_results[0]
        phi_correction = (data['t'] * m_correct / pi_correct) - np.mean((data['t'] * m_correct / pi_correct))
        
        # --- CI for Standard IPW Estimator (Normal Approximation) ---
        var_ipw_infl = np.mean(phi_ipw**2)
        se_ipw = np.sqrt(var_ipw_infl / n)
        lower_ci_ipw = sim_results[0] - 1.96 * se_ipw
        upper_ci_ipw = sim_results[0] + 1.96 * se_ipw
        width_ipw = upper_ci_ipw - lower_ci_ipw
        covered_ipw = (lower_ci_ipw <= true_mu) and (upper_ci_ipw >= true_mu)

        # --- CI for Standard OR Estimator (Normal Approximation) ---
        var_or_infl = np.mean(phi_or**2)
        se_or = np.sqrt(var_or_infl / n)
        lower_ci_or = sim_results[1] - 1.96 * se_or
        upper_ci_or = sim_results[1] + 1.96 * se_or
        width_or = upper_ci_or - lower_ci_or
        covered_or = (lower_ci_or <= true_mu) and (upper_ci_or >= true_mu)

        # --- CI for Standard DR Estimator (Normal Approximation) ---
        # Step 1: DR Influence Function
        phi_dr = phi_or + phi_ipw - phi_correction
        
        # Step 2: Calculate Standard Error from Influence Function
        var_dr_infl = np.mean(phi_dr**2)
        se_dr = np.sqrt(var_dr_infl / n)
        
        # Step 3: Construct DR CI
        lower_ci_dr = sim_results[2] - 1.96 * se_dr
        upper_ci_dr = sim_results[2] + 1.96 * se_dr
        width_dr = upper_ci_dr - lower_ci_dr
        covered_dr = (lower_ci_dr <= true_mu) and (upper_ci_dr >= true_mu)

        # --- CI for Clipped Estimator (Parametric Bootstrap) ---
        # Step 1: Covariance Matrix
        phi_matrix = np.vstack([phi_or, phi_ipw, phi_correction]).T
        cov_matrix = (phi_matrix.T @ phi_matrix) / n
        
        # Step 2: Simulate W
        Z_samples = np.random.multivariate_normal([0, 0, 0], cov_matrix, size=n_simulations)
        Z_or, Z_ipw, Z_c = Z_samples[:, 0], Z_samples[:, 1], Z_samples[:, 2]
        W_samples = Z_or + Z_ipw - np.clip(Z_c, np.minimum(Z_or, Z_ipw), np.maximum(Z_or, Z_ipw))

        # plt.hist(W_samples, bins=30, density=True)
        # # plot normal density for comparison
        # mu_w, std_w = np.mean(W_samples), np.std(W_samples)
        # xmin, xmax = plt.xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = np.exp(-0.5 * ((x - mu_w) / std_w)**2) / (std_w * np.sqrt(2 * np.pi))
        # plt.plot(x, p, 'k', linewidth=2, label='Normal Density')
        # plt.title('Histogram of W samples')
        # plt.xlabel('W')
        # plt.ylabel('Density')
        # plt.axvline(x=np.percentile(W_samples, 2.5), color='r', linestyle='--', label='2.5th Percentile')
        # plt.axvline(x=np.percentile(W_samples, 97.5), color='g', linestyle='--', label='97.5th Percentile')
        # plt.legend()
        # plt.show()
        
        # Step 3: Construct Clipped CI
        q_lower, q_upper = np.percentile(W_samples, [100 * (alpha/2), 100 * (1 - alpha/2)])
        lower_ci_clipped = sim_results[3] - q_upper / np.sqrt(n)
        upper_ci_clipped = sim_results[3] - q_lower / np.sqrt(n)
        width_clipped = upper_ci_clipped - lower_ci_clipped
        covered_clipped = (lower_ci_clipped <= true_mu) and (upper_ci_clipped >= true_mu)

        results.append([lower_ci_ipw, upper_ci_ipw, width_ipw, covered_ipw,
                        lower_ci_or, upper_ci_or, width_or, covered_or,
                        lower_ci_dr, upper_ci_dr, width_dr, covered_dr,
                        lower_ci_clipped, upper_ci_clipped, width_clipped, covered_clipped])

    results_df = pd.DataFrame(results, columns=estimator_names).dropna()
    print(f"\nSimulation for n={n} finished. Used {len(results_df)}/{n_sims} valid runs for metrics.")

    # Export results to CSV
    results_df.to_csv(f'simulation_results_coverage_n{n}.csv', index=False)
    return results_df

if __name__ == '__main__':
    # Set a seed for reproducibility
    np.random.seed(42)

    # For a quicker test run, reduce n_sims (e.g., n_sims=100)
    num_simulations = 1000

    mu_model_correct = False
    pi_model_correct = False
    
    print(f"Running simulation for n=200 with {num_simulations} iterations...")
    results_200 = run_simulation(n=200, n_sims=num_simulations, use_correct_model_mu=mu_model_correct, use_correct_model_pi=pi_model_correct)
    # print avg covered and avg width for each method
    covered = results_200.mean()[results_200.columns.str.contains('covered')]
    print("\nAverage Coverage Probabilities:")
    print(covered)
    width = results_200.mean()[results_200.columns.str.contains('width')]
    print("\nAverage Confidence Interval Widths:")
    print(width)

    print("\n" + "="*60 + "\n")

    print(f"Running simulation for n=1000 with {num_simulations} iterations...")
    results_1000 = run_simulation(n=1000, n_sims=num_simulations, use_correct_model_mu=mu_model_correct, use_correct_model_pi=pi_model_correct)
    covered = results_1000.mean()[results_1000.columns.str.contains('covered')]
    print("\nAverage Coverage Probabilities:")
    print(covered)
    width = results_1000.mean()[results_1000.columns.str.contains('width')]
    print("\nAverage Confidence Interval Widths:")
    print(width)

    print("\n" + "="*60 + "\n")

    print(f"Running simulation for n=100 with {num_simulations} iterations...")
    results_100 = run_simulation(n=100, n_sims=num_simulations, use_correct_model_mu=mu_model_correct, use_correct_model_pi=pi_model_correct)
    covered = results_100.mean()[results_100.columns.str.contains('covered')]
    print("\nAverage Coverage Probabilities:")
    print(covered)
    width = results_100.mean()[results_100.columns.str.contains('width')]
    print("\nAverage Confidence Interval Widths:")
    print(width)