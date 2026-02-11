"""
Exponential Rank Consensus (ERC) for combining scoring functions.

Reference: Palacio-Rodríguez et al., J. Chem. Inf. Model. 2019

The ERC score for compound i is:
    S_i = (1/σ) * Σ_k exp(-r_ik / σ)

where r_ik is the rank of compound i according to scoring function k,
and σ controls the exponential decay (higher σ → more weight to lower ranks).
"""

import numpy as np
import pandas as pd

def compute_zscore(df, score_col_a, score_col_b):
    """
    Compute ERC consensus score from two scoring functions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain both score columns. Higher score = better.
    score_col_a, score_col_b : str
        Column names for the two scoring functions.

    Returns
    -------
    np.ndarray
        Z-normalized consensus scores (higher = better), aligned with df index.
    """

    a = df[score_col_a].values.astype(float)
    b = df[score_col_b].values.astype(float)

    std_a = a.std()
    std_b = b.std()

    z_a = (a - a.mean()) / std_a
    z_b = (b - b.mean()) / std_b

    return z_a + z_b

def add_zscore_columns(df, base, partners):
    """
    Add Z-score consensus columns to df for each (base, partner) pair.

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain '{base}_score' and '{partner}_score' columns.
    base : str
        Base scoring function name (e.g. 'despot').
    partners : list of str
        Partner scoring function names.
    group_col : str or None
        Passed to compute_erc.
    sigma_frac : float
        Passed to compute_erc.

    Returns
    -------
    zscore_names : list of str
        Names of the added zscore methods (without '_score' suffix),
        e.g. ['despot_z_glide', 'despot_z_chemplp', ...]
    """
    z_names = []
    base_col = f'{base}_score'

    for partner in partners:
        partner_col = f'{partner}_score'
        z_name = f'{base}_z_{partner}'
        z_col = f'{z_name}_score'

        df[z_col] = compute_zscore(
            df, base_col, partner_col
        )
        z_names.append(z_name)

    return z_names

def compute_erc(df, score_col_a, score_col_b, group_col=None, sigma_frac=0.05):
    """
    Compute ERC consensus score from two scoring functions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain both score columns. Higher score = better.
    score_col_a, score_col_b : str
        Column names for the two scoring functions.
    group_col : str or None
        Column to group by before ranking. If None, ranks globally.
        Use 'pdb_id' for docking/screening, None for scoring power.
    sigma_frac : float
        σ as a fraction of group size. Default 0.05 (top 5%).

    Returns
    -------
    np.ndarray
        ERC consensus scores (higher = better), aligned with df index.
    """
    erc = np.zeros(len(df))

    if group_col is None:
        groups = [('_global', df.index)]
    else:
        groups = df.groupby(group_col).groups.items()

    for _, idx in groups:
        n = len(idx)
        sigma = max(sigma_frac * n, 1.0)

        ranks_a = df.loc[idx, score_col_a].rank(ascending=False, method='min').values
        ranks_b = df.loc[idx, score_col_b].rank(ascending=False, method='min').values

        erc[idx] = (np.exp(-ranks_a / sigma) + np.exp(-ranks_b / sigma)) / sigma

    return erc

def add_erc_columns(df, base, partners, group_col=None, sigma_frac=0.05):
    """
    Add ERC consensus columns to df for each (base, partner) pair.

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain '{base}_score' and '{partner}_score' columns.
    base : str
        Base scoring function name (e.g. 'despot').
    partners : list of str
        Partner scoring function names.
    group_col : str or None
        Passed to compute_erc.
    sigma_frac : float
        Passed to compute_erc.

    Returns
    -------
    erc_names : list of str
        Names of the added ERC methods (without '_score' suffix),
        e.g. ['despot_erc_glide', 'despot_erc_chemplp', ...]
    """
    erc_names = []
    base_col = f'{base}_score'

    for partner in partners:
        partner_col = f'{partner}_score'
        erc_name = f'{base}_erc_{partner}'
        erc_col = f'{erc_name}_score'

        df[erc_col] = compute_erc(
            df, base_col, partner_col,
            group_col=group_col, sigma_frac=sigma_frac,
        )
        erc_names.append(erc_name)

    return erc_names
