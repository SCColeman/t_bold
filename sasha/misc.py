from dipy.io.image import load_nifti

import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats


def load_image(image_file):
    image_data, image_affine = load_nifti(image_file)
    return image_data


def slr_plot(X, Y, plot=True):
    results = sm.OLS(Y, sm.add_constant(X)).fit()
    c, m = results.params
    y_model = (m * X) + c
    p = results.f_pvalue
    rsq = results.rsquared

    if plot:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        df = pd.DataFrame()
        df['X'] = X
        df['Y'] = Y
        df['y_model'] = y_model
        sns.scatterplot(df, x='X', y='Y')
        sns.lineplot(df, x='X', y='y_model', color='r')
        ax.set(title='R^2 = ' + str("%.2f" % rsq) + ' , p = ' + str("%.2f" % p))
        plt.show()
    else:
        fig = []
        ax = []
    return rsq, p, fig, ax


def slr_params(X, Y):
    results = sm.OLS(Y, sm.add_constant(X)).fit()
    c, m = results.params
    y_model = (m * X) + c
    return y_model


def ttest(group1, group2):
    # shapiro test for normality
    norm_test = stats.shapiro(group1)
    if norm_test[1] < 0.05:
        print('Data is likely not normally distributed,'
              ' \nconsider different statistical test.')
    norm_test = stats.shapiro(group2)
    if norm_test[1] < 0.05:
        print('Data is likely not normally distributed,'
              ' \nconsider different statistical test.')

    print(stats.ttest_rel(group1, group2))
    return stats.ttest_rel(group1, group2)[1]
