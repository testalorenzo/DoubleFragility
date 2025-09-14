#
# Motivating plots
#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

if __name__ == "__main__":
    
    sns.set_theme(style="whitegrid", palette="pastel", font_scale=0.6)
    # sns.set_context("notebook")
    width = 426.79135

    n = 1000

    data = pd.read_csv(f'simulation_results_n{n}.csv')

    # # Plot DR estimator results
    # subset = data[['BC-OLS (I-Pi, I-Y)', 'DR-Clipped (I-Pi, I-Y)', 'IPW-POP (Incorrect Pi)', 'OLS (Incorrect Y)']]
    # # Reshape for seaborn
    # subset = subset.melt(var_name='Estimator', value_name='Value')

    # # Plot kdeplot for the DR estimators

    # axd = plt.figure(figsize=set_size(width, subplots=(1,1))).subplot_mosaic(
    #     [['kde']]
    #     )
    
    # sns.kdeplot(data=subset, x='Value', hue='Estimator', fill=True, common_norm=False, ax=axd['kde'])
    # # Add vertical line for true value
    # true_value = 210
    # plt.axvline(true_value, color='black', linestyle='--', label='True Value')

    # # Add labels and title
    # plt.title(f'Distribution of estimates, n={n}')
    # plt.xlabel('Estimate')
    # plt.ylabel('Density')

    # # Only x axis after 0
    # plt.xlim(left=100)

    # # Show the plot
    # plt.savefig(f'kde_plot_n{n}.pdf', bbox_inches='tight')


    # subset = data[['BC-OLS (C-Pi, C-Y)', 'DR-Clipped (C-Pi, C-Y)']]
    # # Reshape for seaborn
    # subset = subset.melt(var_name='Estimator', value_name='Value')

    # # Plot kdeplot for the DR estimators

    # axd = plt.figure(figsize=set_size(width, subplots=(1,1))).subplot_mosaic(
    #     [['kde']]
    #     )
    
    # sns.kdeplot(data=subset, x='Value', hue='Estimator', fill=True, common_norm=False, ax=axd['kde'])
    # # Add vertical line for true value
    # true_value = 210
    # plt.axvline(true_value, color='black', linestyle='--', label='True Value')

    # # Add labels and title
    # plt.title(f'Distribution of estimates, n={n}')
    # plt.xlabel('Estimate')
    # plt.ylabel('Density')

    # # Only x axis after 0
    # # plt.xlim(left=100)

    # # Show the plot
    # plt.savefig(f'kde_plot_C_n{n}.pdf', bbox_inches='tight')


    # subset = data[['BC-OLS (I-Pi, C-Y)', 'DR-Clipped (I-Pi, C-Y)']]
    # # Reshape for seaborn
    # subset = subset.melt(var_name='Estimator', value_name='Value')

    # # Plot kdeplot for the DR estimators

    # axd = plt.figure(figsize=set_size(width, subplots=(1,1))).subplot_mosaic(
    #     [['kde']]
    #     )
    
    # sns.kdeplot(data=subset, x='Value', hue='Estimator', fill=True, common_norm=False, ax=axd['kde'])
    # # Add vertical line for true value
    # true_value = 210
    # plt.axvline(true_value, color='black', linestyle='--', label='True Value')

    # # Add labels and title
    # plt.title(f'Distribution of estimates, n={n}')
    # plt.xlabel('Estimate')
    # plt.ylabel('Density')

    # # Only x axis after 0
    # # plt.xlim(left=100)

    # # Show the plot
    # plt.savefig(f'kde_plot_CI_n{n}.pdf', bbox_inches='tight')

    # n = 1000

    # data = pd.read_csv(f'simulation_results_n{n}.csv')

    # Plot DR estimator results
    axd = plt.figure(figsize=set_size(width, subplots=(2,2))).subplot_mosaic(
        [['cc', 'ci'],
         ['ic', 'ii']],
        sharex=True, sharey=True,
        gridspec_kw = {'wspace':0.1}
        )

    subset = data[['BC-OLS (C-Pi, C-Y)', 'DR-Clipped (C-Pi, C-Y)', 'IPW-POP (Correct Pi)', 'OLS (Correct Y)']]
    subset = subset.melt(var_name='Estimator', value_name='Value')
    sns.kdeplot(data=subset, x='Value', hue='Estimator', fill=True, common_norm=False, ax=axd['cc'], legend=False)

    subset = data[['BC-OLS (C-Pi, I-Y)', 'DR-Clipped (C-Pi, I-Y)', 'IPW-POP (Correct Pi)', 'OLS (Incorrect Y)']]
    subset = subset.melt(var_name='Estimator', value_name='Value')
    sns.kdeplot(data=subset, x='Value', hue='Estimator', fill=True, common_norm=False, ax=axd['ci'], legend=False)

    subset = data[['BC-OLS (I-Pi, C-Y)', 'DR-Clipped (I-Pi, C-Y)', 'IPW-POP (Incorrect Pi)', 'OLS (Correct Y)']]
    subset = subset.melt(var_name='Estimator', value_name='Value')
    sns.kdeplot(data=subset, x='Value', hue='Estimator', fill=True, common_norm=False, ax=axd['ic'], legend=False)

    subset = data[['BC-OLS (I-Pi, I-Y)', 'DR-Clipped (I-Pi, I-Y)', 'IPW-POP (Incorrect Pi)', 'OLS (Incorrect Y)']]
    # rename columns to remove spaces
    subset.columns = ['DR', 'DR+ACC', 'IPW', 'OR']    
    subset = subset.melt(var_name='Estimator', value_name='Value')
    sns.kdeplot(data=subset, x='Value', hue='Estimator', fill=True, common_norm=False, ax=axd['ii'], legend=True)

    # range between 100 and 300
    axd['cc'].set_xlim(190, 230)

    # Add vertical line for true value
    true_value = 210
    axd['cc'].axvline(true_value, color='black', linestyle='--', label='True Value')
    axd['ci'].axvline(true_value, color='black', linestyle='--', label='True Value')
    axd['ic'].axvline(true_value, color='black', linestyle='--', label='True Value')
    axd['ii'].axvline(true_value, color='black', linestyle='--', label='True Value')

    # Title for each subplot
    axd['cc'].set_title('Correct $\hat\mu$, correct $\hat\pi$')
    axd['ci'].set_title('Incorrect $\hat\mu$, correct $\hat\pi$')
    axd['ic'].set_title('Correct $\hat\mu$, incorrect $\hat\pi$')
    axd['ii'].set_title('Incorrect $\hat\mu$, incorrect $\hat\pi$')

    axd['cc'].tick_params(pad=-3)
    axd['ci'].tick_params(pad=-3)
    axd['ic'].tick_params(pad=-3)
    axd['ii'].tick_params(pad=-3)

    axd['cc'].set_ylabel('Density')
    axd['ic'].set_ylabel('Density')
    axd['ic'].set_xlabel('Estimate')
    axd['ii'].set_xlabel('Estimate')

    axd['cc'].label_outer()
    axd['ci'].label_outer()
    axd['ic'].label_outer()
    axd['ii'].label_outer()

    sns.move_legend(axd['ii'], "upper left", bbox_to_anchor=(-0.63, -0.2), ncol=4)

    # Add labels and title
    plt.title(f'Distribution of estimates, n={n}')
    plt.xlabel('Estimate')
    plt.ylabel('Density')

    # Only x axis after 0
    axd['cc'].xlim(left=100)
    axd['ci'].xlim(left=100)
    axd['ic'].xlim(left=100)
    axd['ii'].xlim(left=100)

    # Show the plot
    plt.savefig(f'intro_plot_n{n}.pdf', bbox_inches='tight')

    #
    # Scatterplots
    #

    n = 1000

    data = pd.read_csv(f'simulation_results_n{n}.csv')

    subset = data[['BC-OLS (I-Pi, I-Y)', 'DR-Clipped (I-Pi, I-Y)', 'IPW-POP (Incorrect Pi)', 'OLS (Incorrect Y)']]
    subset.columns = ['DR', 'DR+ACC', 'IPW', 'OR'] 

    axd = plt.figure(figsize=set_size(width, subplots=(4,3))).subplot_mosaic(
        [['acc_or_cc', 'acc_ipw_cc', 'acc_dr_cc'],
         ['acc_or_ci', 'acc_ipw_ci', 'acc_dr_ci'],
         ['acc_or_ic', 'acc_ipw_ic', 'acc_dr_ic'],
         ['acc_or_ii', 'acc_ipw_ii', 'acc_dr_ii']],
        sharey=True,
        sharex=False
        )
    
    subset = data[['BC-OLS (C-Pi, C-Y)', 'DR-Clipped (C-Pi, C-Y)', 'IPW-POP (Correct Pi)', 'OLS (Correct Y)']]
    subset.columns = ['DR', 'DR+ACC', 'IPW', 'OR']
    sns.scatterplot(data=subset, x='OR', y='DR+ACC', ax=axd['acc_or_cc'], s=5)
    axd['acc_or_cc'].set_title('DR+ACC vs OR')
    axd['acc_or_cc'].set_xlabel('OR estimate')
    axd['acc_or_cc'].set_ylabel('DR+ACC estimate')
    axd['acc_or_cc'].tick_params(pad=-3)
    sns.scatterplot(data=subset, x='IPW', y='DR+ACC', ax=axd['acc_ipw_cc'], s=5)
    axd['acc_ipw_cc'].set_title('DR+ACC vs IPW')
    axd['acc_ipw_cc'].set_xlabel('IPW estimate')
    axd['acc_ipw_cc'].tick_params(pad=-3)
    sns.scatterplot(data=subset, x='DR', y='DR+ACC', ax=axd['acc_dr_cc'], s=5)
    axd['acc_dr_cc'].set_title('DR+ACC vs DR')
    axd['acc_dr_cc'].set_xlabel('DR estimate')
    axd['acc_dr_cc'].tick_params(pad=-3)
    
    subset = data[['BC-OLS (I-Pi, C-Y)', 'DR-Clipped (I-Pi, C-Y)', 'IPW-POP (Incorrect Pi)', 'OLS (Correct Y)']]
    subset.columns = ['DR', 'DR+ACC', 'IPW', 'OR']
    sns.scatterplot(data=subset, x='OR', y='DR+ACC', ax=axd['acc_or_ci'], s=5)
    axd['acc_or_ci'].set_title('')
    axd['acc_or_ci'].set_xlabel('OR estimate')
    axd['acc_or_ci'].set_ylabel('DR+ACC estimate')
    axd['acc_or_ci'].tick_params(pad=-3)
    sns.scatterplot(data=subset, x='IPW', y='DR+ACC', ax=axd['acc_ipw_ci'], s=5)
    axd['acc_ipw_ci'].set_title('')
    axd['acc_ipw_ci'].set_xlabel('IPW estimate')
    axd['acc_ipw_ci'].tick_params(pad=-3)
    sns.scatterplot(data=subset, x='DR', y='DR+ACC', ax=axd['acc_dr_ci'], s=5)
    axd['acc_dr_ci'].set_title('')
    axd['acc_dr_ci'].set_xlabel('DR estimate')
    axd['acc_dr_ci'].tick_params(pad=-3)

    subset = data[['BC-OLS (C-Pi, I-Y)', 'DR-Clipped (C-Pi, I-Y)', 'IPW-POP (Correct Pi)', 'OLS (Incorrect Y)']]  
    subset.columns = ['DR', 'DR+ACC', 'IPW', 'OR']
    sns.scatterplot(data=subset, x='OR', y='DR+ACC', ax=axd['acc_or_ic'], s=5)
    axd['acc_or_ic'].set_title('')
    axd['acc_or_ic'].set_xlabel('OR estimate')
    axd['acc_or_ic'].set_ylabel('DR+ACC estimate')
    axd['acc_or_ic'].tick_params(pad=-3)
    sns.scatterplot(data=subset, x='IPW', y='DR+ACC', ax=axd['acc_ipw_ic'], s=5)
    axd['acc_ipw_ic'].set_title('')
    axd['acc_ipw_ic'].set_xlabel('IPW estimate')
    axd['acc_ipw_ic'].tick_params(pad=-3)
    sns.scatterplot(data=subset, x='DR', y='DR+ACC', ax=axd['acc_dr_ic'], s=5)
    axd['acc_dr_ic'].set_title('')
    axd['acc_dr_ic'].set_xlabel('DR estimate')
    axd['acc_dr_ic'].tick_params(pad=-3)

    subset = data[['BC-OLS (I-Pi, I-Y)', 'DR-Clipped (I-Pi, I-Y)', 'IPW-POP (Incorrect Pi)', 'OLS (Incorrect Y)']]
    subset.columns = ['DR', 'DR+ACC', 'IPW', 'OR'] 
    sns.scatterplot(data=subset, x='OR', y='DR+ACC', ax=axd['acc_or_ii'], s=5)
    axd['acc_or_ii'].set_title('')
    axd['acc_or_ii'].set_xlabel('OR estimate')
    axd['acc_or_ii'].set_ylabel('DR+ACC estimate')
    axd['acc_or_ii'].tick_params(pad=-3)
    sns.scatterplot(data=subset, x='IPW', y='DR+ACC', ax=axd['acc_ipw_ii'], s=5)
    axd['acc_ipw_ii'].set_title('')
    axd['acc_ipw_ii'].set_xlabel('IPW estimate')
    axd['acc_ipw_ii'].tick_params(pad=-3)
    sns.scatterplot(data=subset, x='DR', y='DR+ACC', ax=axd['acc_dr_ii'], s=5)
    axd['acc_dr_ii'].set_title('')
    axd['acc_dr_ii'].set_xlabel('DR estimate')
    axd['acc_dr_ii'].tick_params(pad=-3)


    # Show the plot
    plt.savefig(f'scatterplot_n{n}.pdf', bbox_inches='tight')
