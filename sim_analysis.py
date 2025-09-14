import numpy as np
import pandas as pd
from statsmodels.api import Logit, OLS, WLS, add_constant
import warnings

# Suppress convergence and future warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
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


def stratification_estimator(data, pi_hat):
    """Calculates the propensity-stratified estimate."""
    respondents = data['t'] == 1
    if respondents.sum() == 0: return np.nan
    
    # Use temporary DataFrame to avoid modifying original in loop
    temp_data = pd.DataFrame({'pi_hat': pi_hat, 't': data['t'], 'y': data['y']})
    temp_data['pi_strata'] = pd.qcut(temp_data['pi_hat'], 5, labels=False, duplicates='drop')

    strata_means = temp_data[respondents].groupby('pi_strata')['y'].mean()
    strata_props = temp_data.groupby('pi_strata').size() / len(temp_data)
    
    # Align means and props for multiplication, fill missing strata means
    mu_hat_df = pd.DataFrame({'props': strata_props, 'means': strata_means}).fillna(temp_data.loc[respondents, 'y'].mean())
    return (mu_hat_df['props'] * mu_hat_df['means']).sum()


def dual_stratification_estimator(data, pi_hat, m_hat):
    """Calculates DR estimate by stratifying on propensity and predicted values."""
    if data['t'].sum() == 0: return np.nan
    temp_data = data.copy()
    temp_data['pi_strata'] = pd.qcut(pi_hat, 5, labels=False, duplicates='drop')
    temp_data['m_strata'] = pd.qcut(m_hat, 5, labels=False, duplicates='drop')

    respondent_data = temp_data[temp_data['t'] == 1]
    
    # Define the complete grid from the full sample
    cell_props = temp_data.groupby(['pi_strata', 'm_strata']).size().unstack().fillna(0) / len(data)
    
    # Create cell means from respondents
    cell_means = respondent_data.groupby(['pi_strata', 'm_strata'])['y'].mean().unstack()

    # *** FIX: Reindex cell_means to match the full grid before imputation ***
    cell_means = cell_means.reindex(index=cell_props.index, columns=cell_props.columns)

    # Impute missing cells (those with no respondents)
    imputed_means = cell_means.apply(lambda row: row.fillna(row.mean()), axis=1).apply(lambda col: col.fillna(col.mean()), axis=0).fillna(respondent_data['y'].mean())
    
    # Now imputed_means and cell_props have the same shape
    return np.sum(imputed_means.values * cell_props.values)


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


def wls_estimator(data, pi_hat, use_correct_y_model):
    """Calculates regression estimate with inverse-propensity weighted coefficients."""
    respondents = data['t'] == 1
    if respondents.sum() < 2: return np.nan
    weights = 1 / pi_hat[respondents].clip(lower=1e-6)

    X_resp = add_constant(data.loc[respondents, ['z1', 'z2', 'z3', 'z4']]) if use_correct_y_model else add_constant(data.loc[respondents, ['x1', 'x2', 'x3', 'x4']])
    X_all = add_constant(data[['z1', 'z2', 'z3', 'z4']]) if use_correct_y_model else add_constant(data[['x1', 'x2', 'x3', 'x4']])

    wls_model = WLS(data.loc[respondents, 'y'], X_resp, weights=weights).fit()
    return wls_model.predict(X_all).mean()


def pi_cov_estimator(data, pi_hat, use_correct_y_model, use_inverse_pi):
    """Calculates the regression estimate with propensity-based covariates."""
    respondents = data['t'] == 1
    if respondents.sum() < 2: return np.nan

    X_all_base = add_constant(data[['z1', 'z2', 'z3', 'z4']]) if use_correct_y_model else add_constant(data[['x1', 'x2', 'x3', 'x4']])
    
    if use_inverse_pi:
        pi_hat_clipped = pi_hat.clip(lower=1e-6)
        X_all_aug = X_all_base.assign(inv_pi=1 / pi_hat_clipped)
    else:  # Use quintile dummies
        pi_strata = pd.qcut(pi_hat, 5, labels=False, duplicates='drop')
        pi_dummies = pd.get_dummies(pi_strata, prefix='pi_q', drop_first=True)
        if pi_dummies.empty:
            X_all_aug = X_all_base
        else:
            X_all_aug = X_all_base.join(pi_dummies)
    
    # Convert the final DataFrame to float to prevent casting errors
    X_all_aug = X_all_aug.astype(float).fillna(0)
    X_resp_aug = X_all_aug.loc[respondents]
    
    # Check for perfect collinearity before fitting
    if np.linalg.matrix_rank(X_resp_aug) < X_resp_aug.shape[1]:
        ols_fallback = OLS(data.loc[respondents, 'y'], X_all_base.loc[respondents]).fit()
        return ols_fallback.predict(X_all_base).mean()

    ols_aug_model = OLS(data.loc[respondents, 'y'], X_resp_aug).fit()
    return ols_aug_model.predict(X_all_aug).mean()


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


def run_simulation(n, n_sims=1000, true_mu=210.0):
    """Runs the full simulation study."""
    estimator_names = [
        'IPW-POP (Correct Pi)', 'IPW-NR (Correct Pi)', 'IPW-POP (Incorrect Pi)', 'IPW-NR (Incorrect Pi)',
        'Strat-Pi (Correct Pi)', 'Strat-Pi (Incorrect Pi)', 'OLS (Correct Y)', 'OLS (Incorrect Y)',
        'Strat-Pi-M (C-Pi, C-Y)', 'Strat-Pi-M (C-Pi, I-Y)', 'Strat-Pi-M (I-Pi, C-Y)', 'Strat-Pi-M (I-Pi, I-Y)',
        'BC-OLS (C-Pi, C-Y)', 'BC-OLS (C-Pi, I-Y)', 'BC-OLS (I-Pi, C-Y)', 'BC-OLS (I-Pi, I-Y)',
        'WLS (C-Pi, C-Y)', 'WLS (C-Pi, I-Y)', 'WLS (I-Pi, C-Y)', 'WLS (I-Pi, I-Y)',
        'Pi-Cov (C-Pi, C-Y)', 'Pi-Cov (C-Pi, I-Y)', 'Pi-Cov (I-Pi, C-Y)', 'Pi-Cov (I-Pi, I-Y)',
        '1/Pi-Cov (C-Pi, C-Y)', '1/Pi-Cov (C-Pi, I-Y)', '1/Pi-Cov (I-Pi, C-Y)', '1/Pi-Cov (I-Pi, I-Y)', 
        'DR-Clipped (C-Pi, C-Y)', 'DR-Clipped (C-Pi, I-Y)', 'DR-Clipped (I-Pi, C-Y)', 'DR-Clipped (I-Pi, I-Y)'
    ]
    results = []
    
    for i in range(n_sims):
        if (i + 1) % 100 == 0: print(f"  Sim {i+1}/{n_sims}")
        data = generate_data(n)

        pi_correct = estimate_propensity_scores(data, use_correct_model=True)
        pi_incorrect = estimate_propensity_scores(data, use_correct_model=False)
        m_correct, eps_correct = get_y_model_predictions(data, use_correct_model=True)
        m_incorrect, eps_incorrect = get_y_model_predictions(data, use_correct_model=False)

        sim_results = [
            ipw_estimator(data, pi_correct, pop_weighting=True),
            ipw_estimator(data, pi_correct, pop_weighting=False),
            ipw_estimator(data, pi_incorrect, pop_weighting=True),
            ipw_estimator(data, pi_incorrect, pop_weighting=False),
            stratification_estimator(data, pi_correct),
            stratification_estimator(data, pi_incorrect),
            m_correct.mean(),
            m_incorrect.mean(),
            dual_stratification_estimator(data, pi_correct, m_correct),
            dual_stratification_estimator(data, pi_correct, m_incorrect),
            dual_stratification_estimator(data, pi_incorrect, m_correct),
            dual_stratification_estimator(data, pi_incorrect, m_incorrect),
            bc_ols_estimator(data, m_correct, eps_correct, pi_correct),
            bc_ols_estimator(data, m_incorrect, eps_incorrect, pi_correct),
            bc_ols_estimator(data, m_correct, eps_correct, pi_incorrect),
            bc_ols_estimator(data, m_incorrect, eps_incorrect, pi_incorrect),
            wls_estimator(data, pi_correct, use_correct_y_model=True),
            wls_estimator(data, pi_correct, use_correct_y_model=False),
            wls_estimator(data, pi_incorrect, use_correct_y_model=True),
            wls_estimator(data, pi_incorrect, use_correct_y_model=False),
            pi_cov_estimator(data, pi_correct, use_correct_y_model=True, use_inverse_pi=False),
            pi_cov_estimator(data, pi_correct, use_correct_y_model=False, use_inverse_pi=False),
            pi_cov_estimator(data, pi_incorrect, use_correct_y_model=True, use_inverse_pi=False),
            pi_cov_estimator(data, pi_incorrect, use_correct_y_model=False, use_inverse_pi=False),
            pi_cov_estimator(data, pi_correct, use_correct_y_model=True, use_inverse_pi=True),
            pi_cov_estimator(data, pi_correct, use_correct_y_model=False, use_inverse_pi=True),
            pi_cov_estimator(data, pi_incorrect, use_correct_y_model=True, use_inverse_pi=True),
            pi_cov_estimator(data, pi_incorrect, use_correct_y_model=False, use_inverse_pi=True),
            clipped_dr_estimator(data, pi_correct, m_correct),
            clipped_dr_estimator(data, pi_correct, m_incorrect),
            clipped_dr_estimator(data, pi_incorrect, m_correct),
            clipped_dr_estimator(data, pi_incorrect, m_incorrect)
        ]
        results.append(sim_results)

    results_df = pd.DataFrame(results, columns=estimator_names).dropna()
    print(f"\nSimulation for n={n} finished. Used {len(results_df)}/{n_sims} valid runs for metrics.")

    # Export results to CSV
    results_df.to_csv(f'simulation_results_n{n}.csv', index=False)

    # Calculate metrics for each estimator
    metrics = {}
    for col in results_df.columns:
        bias = (results_df[col] - true_mu).mean()
        std_dev = results_df[col].std()
        percent_bias = (bias / std_dev) * 100 if std_dev != 0 else 0
        rmse = np.sqrt(((results_df[col] - true_mu)**2).mean())
        mae = np.abs(results_df[col] - true_mu).median()
        metrics[col] = {'Bias': bias, '% Bias': percent_bias, 'RMSE': rmse, 'MAE': mae}

    metrics = pd.DataFrame(metrics).T
    metrics.to_csv(f'simulation_metrics_n{n}.csv')
    return metrics

if __name__ == '__main__':
    # Set a seed for reproducibility
    np.random.seed(42)

    # For a quicker test run, reduce n_sims (e.g., n_sims=100)
    num_simulations = 1000
    
    print(f"Running simulation for n=200 with {num_simulations} iterations... (This may take several minutes)")
    results_200 = run_simulation(n=200, n_sims=num_simulations)
    print("\n" + "="*60)
    print("Combined Simulation Results for n=200")
    print("="*60)
    # Widen pandas display to show all columns
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 10)
    print(results_200)

    print("\n" + "="*60 + "\n")

    print(f"Running simulation for n=1000 with {num_simulations} iterations... (This may take a long time)")
    results_1000 = run_simulation(n=1000, n_sims=num_simulations)
    print("\n" + "="*60)
    print("Combined Simulation Results for n=1000")
    print("="*60)
    print(results_1000)

    print("\n" + "="*60 + "\n")

    print(f"Running simulation for n=100 with {num_simulations} iterations... (This may take a long time)")
    results_100 = run_simulation(n=100, n_sims=num_simulations)
    print("\n" + "="*60)
    print("Combined Simulation Results for n=100")
    print("="*60)
    print(results_100)