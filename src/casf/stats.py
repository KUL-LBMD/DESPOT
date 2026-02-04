import numpy as np
from scipy.stats import bootstrap, pearsonr
import scikit_posthocs as sp
import pandas as pd

### Helper functions ###
def mean_stat(data):
    return np.mean(data)

def pearson_stat(scores, affinities):
    """
    scores, affinities: arrays with identical shape
    axis: bootstrap axis (ignored here because pearsonr is 1D)
    """
    return pearsonr(scores, affinities)[0]

###

def bca_pearson_ci(scores, affinities, confidence_level = 0.9, n_resamples = 10000):
    """
    Bias-corrected and accelerated bootstrap of Pearson correlation.

    Parameters
    ----------
    scores [L]: 1D array, value per target
    affinities [L]: 1D array, affinity per target
    confidence_level [float (0-1)]: set interval
    n_resamples [int]: How many bootstrap resamples to do

    Returns
    -------
    results: Dictionary containing
        - estimate [float]: Pearson correlation of sample values
        - ci_low [float]: lower edge of confidence interval
        - ci_high [float]: upper edge of confidence interval
        - dist [N]: mean statistic, resampled N times 
    """

    res = bootstrap(
        data = (scores, affinities),
        statistic = pearson_stat,
        n_resamples = n_resamples,
        confidence_level = confidence_level,
        method = 'BCa',
        random_state = 42,
        vectorized = False,
        paired = True
    )

    return {
        'estimate': pearsonr(scores, affinities)[0],
        'ci_low': res.confidence_interval.low,
        'ci_high': res.confidence_interval.high,
        'dist': res.bootstrap_distribution
    }


def bca_mean_ci(values, confidence_level = 0.9, n_resamples = 10000):
    """
    Bias-corrected and accelerated bootstrap of the mean.

    Parameters
    ----------
    values [L]: 1D array, value per target
    confidence_level [float (0-1)]: set interval
    n_resamples [int]: How many bootstrap resamples to do

    Returns
    -------
    results: Dictionary containing
        - mean [float]: mean of sample values
        - ci_low [float]: lower edge of confidence interval
        - ci_high [float]: upper edge of confidence interval
        - dist [N]: mean statistic, resampled N times 
    """
    
    res = bootstrap(
        data = (values,),
        statistic = mean_stat,
        n_resamples = n_resamples,
        confidence_level = confidence_level,
        method = 'BCa',
        random_state = 42
    )

    return {
        'mean': np.mean(values),
        'ci_low': res.confidence_interval.low,
        'ci_high': res.confidence_interval.high,
        'dist': res.bootstrap_distribution
    }

def friedman_nemenyi(stats_dict):
    """
    Non-parametric statistical tests. 
    See whether ranking of scoring functions significantly differs from each other.

    Parameters
    ----------
    stats_dict: Dictionary of dictionary
        For each scoring function, contains:
            - mean [float]: mean of sample values
            - ci_low [float]: lower edge of confidence interval
            - ci_high [float]: upper edge of confidence interval
            - dist [N]: mean statistic, resampled N times

    Returns
    -------
    pvals:
        pd.DataFrame:
            Data = symmetric [S,S] array: p-value of whether 2 mean values come from the same distribution
            Columns, Indices = names of scoring functions
    """

    methods = list(stats_dict.keys())
    X = np.vstack([stats_dict[m]['dist'] for m in methods]).T # [N, S]
    df = pd.DataFrame(X, columns = methods)
    pvals = sp.posthoc_nemenyi_friedman(df) # [S, S]

    return pvals
