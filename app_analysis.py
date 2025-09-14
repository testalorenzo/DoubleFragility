#
# Peptide data analysis script
#

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
from missingforest import MissForest
import matplotlib.pyplot as plt

def clip(value, L, U):
    if value < L:
        return L
    elif value > U:
        return U
    else:
        return value

if __name__ == '__main__':

    # Load data
    covariates = pd.read_csv('./peptides/covariates.csv', index_col=0)
    peptides = pd.read_csv('./peptides/peptides.csv', index_col=0)

    missing = peptides.isna().sum()
    peptides = peptides.loc[:, missing <= 22]  # keep peptides with less than 10% missing values

    imputer = MissForest(n_jobs=-1, random_state=42, verbose=0)
    peptides_imputed = imputer.fit_transform(peptides)
    peptides = pd.DataFrame(peptides_imputed, columns=peptides.columns, index=peptides.index)
    peptides.to_csv('./peptides/peptides_imputed.csv', index=True)

    # Load imputed data computed above
    peptides = pd.read_csv('./peptides/peptides_imputed.csv', index_col=0)

    A = covariates['condition']
    A = (A == 'ADD').astype(int)

    X = covariates.drop(columns=['condition'])
    # make region categorical
    X = pd.get_dummies(X, columns=['region', 'sex'])
    X = X.applymap(lambda x: 1 if x is True else (0 if x is False else x))

    # order rows
    X = X.loc[peptides.index, :]
    A = A.loc[peptides.index]

    # For each transcript, compute ATE using Naive, OR, IPW, DR
    results = pd.DataFrame(columns=['peptide', 'missing', 'naive', 'or_ate', 'ipw_ate', 'dr', 'dr_clip'])
    
    for peptide in tqdm(peptides.columns):

        #
        # Estimation
        #

        peptides2 = peptides.loc[:, [peptide]]
        data = pd.concat([peptides2, X], axis=1)
        
        A_complete = A.loc[data.index]
        
        Y_treatment = data.loc[A_complete==1, peptide]
        Y_control = data.loc[A_complete==0, peptide]

        X_complete = data.drop(peptide, axis=1)
        X_treatment = X_complete.loc[A_complete==1, :]
        X_control = X_complete.loc[A_complete==0, :]

        # Naive
        naive = Y_treatment.mean() - Y_control.mean()

        # OR
        # model_treatment = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_treatment, Y_treatment)
        # model_control = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_control, Y_control)
        model_treatment = LinearRegression().fit(X_treatment, Y_treatment)
        model_control = LinearRegression().fit(X_control, Y_control)
        Y_treatment_pred = model_treatment.predict(X_complete)
        Y_control_pred = model_control.predict(X_complete)

        Y_treatment_pred = np.repeat(Y_treatment.mean(), len(X_complete))
        Y_control_pred = np.repeat(Y_control.mean(), len(X_complete))

        Y_treatment_pred = pd.Series(Y_treatment_pred, index=X_complete.index)
        Y_control_pred = pd.Series(Y_control_pred, index=X_complete.index)

        or_treatment = Y_treatment_pred.mean()
        or_control = Y_control_pred.mean()
        or_ate = or_treatment - or_control

        # IPW
        n = len(A_complete)
        model_ipw = LogisticRegression(max_iter=1000).fit(X_complete, A_complete)
        # model_ipw = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_complete, A_complete)
        propensities = model_ipw.predict_proba(X_complete)[:,1]
        # propensities = np.clip(propensities, 1e-6, 1-1e-6)  # avoid extreme weights
        
        scaling_treatment = (1 / propensities[A_complete == 1]).sum() / n
        scaling_control = (1 / (1 - propensities[A_complete == 0])).sum() / n

        propensities_treatment = propensities[A_complete == 1] * scaling_treatment
        propensities_control = (1 - propensities[A_complete == 0]) * scaling_control

        propensities_treatment = np.clip(propensities_treatment, 1e-6, 1-1e-6)  # avoid extreme weights
        propensities_control = np.clip(propensities_control, 1e-6, 1-1e-6)

        weights_treatment = 1 / propensities_treatment
        weights_control = 1 / propensities_control
        weights_all = np.zeros(n)
        weights_all[A_complete == 1] = weights_treatment
        weights_all[A_complete == 0] = weights_control

        # ipw_treatment = ((A_complete * Y_treatment).fillna(0)[data.index] / propensities).mean()
        # ipw_control = (((1 - A_complete) * Y_control).fillna(0)[data.index] / (1 - propensities)).mean()
        
        ipw_treatment = np.sum(weights_treatment * Y_treatment) / n
        ipw_control = np.sum(weights_control * Y_control) / n
        ipw_ate = ipw_treatment - ipw_control

        # Hajek
        ipw_treatment_h = np.sum(Y_treatment / propensities[A_complete == 1]) / (1 / propensities[A_complete == 1]).sum()
        ipw_control_h = np.sum(Y_control / (1 - propensities[A_complete == 0])) / (1 / (1 - propensities[A_complete == 0])).sum()
        ipw_ate_h = ipw_treatment_h - ipw_control_h

        # DR
        correction_treatment = weights_all *(A_complete * Y_treatment_pred)[data.index]
        correction_control = weights_all * ((1 - A_complete) * Y_control_pred)[data.index]

        dr = or_ate + ipw_ate - (correction_treatment.mean() - correction_control.mean())

        # just Check dr is right computation
        # pseudo1 = Y_treatment_pred + ((A_complete * Y_treatment).fillna(0)[data.index])  / propensities - ((A_complete * Y_treatment_pred)[data.index]) / propensities
        # pseudo0 = Y_control_pred + (((1 - A_complete) * Y_control).fillna(0)[data.index] - ((1 - A_complete) * Y_control_pred)[data.index]) / (1 - propensities)
        # pseudo = pseudo1 - pseudo0
        # dr = pseudo.mean()

        # DR clip
        L_treatment = min(or_treatment, ipw_treatment)
        U_treatment = max(or_treatment, ipw_treatment)
        L_control = min(or_control, ipw_control)
        U_control = max(or_control, ipw_control)
        correction_treatment_clip = clip(correction_treatment.mean(), L_treatment, U_treatment)
        correction_control_clip = clip(correction_control.mean(), L_control, U_control)
         
        dr_clip = or_ate + ipw_ate - (correction_treatment_clip - correction_control_clip)

        # Store results
        results = pd.concat([results, pd.DataFrame({'peptide': peptide,
                                                    'missing': len(A_complete),
                                                    'naive': naive,
                                                    'or_ate': or_ate,
                                                    'ipw_ate': ipw_ate,
                                                    'ipw_ate_h': ipw_ate_h,
                                                    'dr': dr,
                                                    'dr_clip': dr_clip},
                                                    index=[0])], ignore_index=True)
        
        #
        # Inference
        #

        phi_or = Y_treatment_pred - Y_control_pred - or_ate
        phi_ipw = A_complete * (weights_all * data[peptide]) - (1 - A_complete) * (weights_all * data[peptide]) - ipw_ate
        phi_correction = correction_treatment - correction_control - (correction_treatment.mean() - correction_control.mean())
        
        phi_dr = phi_or + phi_ipw - phi_correction
        
        # Step 2: Calculate Standard Error from Influence Function
        var_dr_infl = np.mean(phi_dr**2)
        se_dr = np.sqrt(var_dr_infl / n)
        
        # Step 3: Construct DR CI
        lower_ci_dr = dr - 1.96 * se_dr
        upper_ci_dr = dr + 1.96 * se_dr

        phi_matrix = np.vstack([phi_or, phi_ipw[data.index], phi_correction[data.index]]).T
        cov_matrix = (phi_matrix.T @ phi_matrix) / n
        
        # Step 2: Simulate W
        n_simulations = 10000
        alpha = 0.05 
        Z_samples = np.random.multivariate_normal([0, 0, 0], cov_matrix, size=n_simulations)
        Z_or, Z_ipw, Z_c = Z_samples[:, 0], Z_samples[:, 1], Z_samples[:, 2]
        W_samples = Z_or + Z_ipw - np.clip(Z_c, np.minimum(Z_or, Z_ipw), np.maximum(Z_or, Z_ipw))
        
        q_lower, q_upper = np.percentile(W_samples, [100 * (alpha/2), 100 * (1 - alpha/2)])
        lower_ci_clipped = dr_clip - q_upper / np.sqrt(n)
        upper_ci_clipped = dr_clip - q_lower / np.sqrt(n)

        # Store confidence intervals
        results.loc[results['peptide'] == peptide, 'lower_ci_dr'] = lower_ci_dr
        results.loc[results['peptide'] == peptide, 'upper_ci_dr'] = upper_ci_dr
        results.loc[results['peptide'] == peptide, 'lower_ci_clipped'] = lower_ci_clipped
        results.loc[results['peptide'] == peptide, 'upper_ci_clipped'] = upper_ci_clipped

    # Save results
    results.to_csv('./peptides/peptides_ATE_results.csv', index=False)