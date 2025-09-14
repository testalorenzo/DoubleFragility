#
# Plot app results
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
    width = 426.79135  # width in pts

    #
    # Densities
    #

    data = pd.read_csv('./peptides/peptides_ATE_results.csv')

    data.drop('missing', axis=1, inplace=True)
    data.drop('naive', axis=1, inplace=True)
    data.drop('ipw_ate_h', axis=1, inplace=True)
    data.drop('lower_ci_clipped', axis=1, inplace=True)
    data.drop('upper_ci_clipped', axis=1, inplace=True)
    data.drop('lower_ci_dr', axis=1, inplace=True)
    data.drop('upper_ci_dr', axis=1, inplace=True)

    data.columns = ['peptide', 'OR', 'IPW', 'DR', 'DR+ACC']

    data = data.melt(id_vars='peptide', var_name='Estimator', value_name='Value')

    # order the estimators alphabetically
    data.sort_values(by='Estimator', inplace=True)

    axd = plt.figure(figsize=set_size(width, subplots=(1,2))).subplot_mosaic(
            [['kde', 'diff']],
            width_ratios=[0.75, 0.25]
            )
        
    sns.kdeplot(data=data, x='Value', hue='Estimator', fill=True, common_norm=False, ax=axd['kde'])
    # Add vertical line for true value

    axd['kde'].axvline(0, color='black', linestyle='--', label='True Value')
    axd['kde'].tick_params(pad=-3)
    axd['kde'].label_outer()
    sns.move_legend(axd['kde'], "upper right", ncol=1)

    # Add labels and title
    axd['kde'].set_title(f'Estimated ATEs across peptides')
    axd['kde'].set_xlabel('Estimate')
    axd['kde'].set_ylabel('Density')

    # set x axis limits to -0.5, 0.5
    axd['kde'].set_xlim(-1.5, 1.5)

    # Plot difference between estimators DR and clip
    data = pd.read_csv('./peptides/peptides_ATE_results.csv')

    data.drop('missing', axis=1, inplace=True)
    data.drop('naive', axis=1, inplace=True)
    data.drop('ipw_ate_h', axis=1, inplace=True)
    data.drop('lower_ci_clipped', axis=1, inplace=True)
    data.drop('upper_ci_clipped', axis=1, inplace=True)
    data.drop('lower_ci_dr', axis=1, inplace=True)
    data.drop('upper_ci_dr', axis=1, inplace=True)
    # data.drop('Naive', axis=1, inplace=True)

    data.columns = ['peptide', 'OR', 'IPW', 'DR', 'DR+ACC']
    
    data['diff'] = data['DR'] - data['DR+ACC']

    sns.kdeplot(data=data, x='diff', fill=True, color='gray', ax=axd['diff'])
    # Add vertical line for true value
    axd['diff'].axvline(0, color='black', linestyle='--', label='True Value')
    axd['diff'].tick_params(pad=-3)
    axd['diff'].label_outer()
    axd['diff'].set_title(f'Estimated DR - DR+ACC')
    axd['diff'].set_xlabel('Difference')
    axd['diff'].set_ylabel('Density')

    # Show the plot
    plt.savefig('app_results.pdf', bbox_inches='tight')

    #
    # Scatterplots
    #

    data = pd.read_csv('./peptides/peptides_ATE_results.csv')

    data.drop('missing', axis=1, inplace=True)
    data.drop('naive', axis=1, inplace=True)
    data.drop('ipw_ate_h', axis=1, inplace=True)
    data.drop('lower_ci_clipped', axis=1, inplace=True)
    data.drop('upper_ci_clipped', axis=1, inplace=True)
    data.drop('lower_ci_dr', axis=1, inplace=True)
    data.drop('upper_ci_dr', axis=1, inplace=True)

    data.columns = ['peptide', 'OR', 'IPW', 'DR', 'DR+ACC']

    axd = plt.figure(figsize=set_size(width, subplots=(1,3))).subplot_mosaic(
            [['acc_or', 'acc_ipw', 'acc_dr']],
            sharey=True
            )
    
    sns.scatterplot(data=data, x='OR', y='DR+ACC', ax=axd['acc_or'], s=5)
    axd['acc_or'].set_title('DR+ACC vs OR')
    axd['acc_or'].set_xlabel('OR estimate')
    axd['acc_or'].set_ylabel('DR+ACC estimate')
    axd['acc_or'].tick_params(pad=-3)
    axd['acc_or'].label_outer()

    sns.scatterplot(data=data, x='IPW', y='DR+ACC', ax=axd['acc_ipw'], s=5)
    axd['acc_ipw'].set_title('DR+ACC vs IPW')
    axd['acc_ipw'].set_xlabel('IPW estimate')
    axd['acc_ipw'].tick_params(pad=-3)
    axd['acc_ipw'].label_outer()

    sns.scatterplot(data=data, x='DR', y='DR+ACC', ax=axd['acc_dr'], s=5)
    axd['acc_dr'].set_title('DR+ACC vs DR')
    axd['acc_dr'].set_xlabel('DR estimate')
    axd['acc_dr'].tick_params(pad=-3)
    axd['acc_dr'].label_outer()

    # Show the plot
    plt.savefig('app_results_scatter.pdf', bbox_inches='tight')