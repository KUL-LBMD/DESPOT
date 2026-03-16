"""
Exponential Rank Consensus (ERC) for combining scoring functions.

Reference: Palacio-Rodríguez et al., J. Chem. Inf. Model. 2019

The ERC score for compound i is:
    S_i = (1/σ) * Σ_k exp(-r_ik / σ)

where r_ik is the rank of compound i according to scoring function k,
and σ controls the exponential decay (higher σ → more weight to lower ranks).

Supports combining 2 or more scoring functions.
"""

import numpy as np
import pandas as pd

# ============================================================================
# ERC consensus
# ============================================================================

def compute_erc(df, score_cols, group_col=None, sigma_frac=0.05):
    """
    Compute ERC consensus score from multiple scoring functions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all score columns. Higher score = better.
    score_cols : list of str
        Column names for the scoring functions (2 or more).
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

        erc_group = np.zeros(len(idx))
        for col in score_cols:
            ranks = df.loc[idx, col].rank(ascending=False, method='min').values
            erc_group += np.exp(-ranks / sigma)

        erc[idx] = erc_group / sigma

    return erc


def add_erc_columns(df, partner_combos, group_col=None, sigma_frac=0.05):
    """
    Add ERC consensus columns to df for each combination.

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain '{base}_score' and '{partner}_score' columns.
    base : str
        Base scoring function name (e.g. 'despot').
    partner_combos : list of list of str
        Each element is a list of partner scoring function names to combine
        with base. E.g. [['deltavina'], ['deltavina', 'chemplp']].
    group_col : str or None
        Passed to compute_erc.
    sigma_frac : float
        Passed to compute_erc.

    Returns
    -------
    erc_names : list of str
        Names of the added ERC methods (without '_score' suffix).
    """
    erc_names = []

    for partners in partner_combos:
        p0 = partners[0]
        base_col = f'{p0}_score'
        partner_cols = [f'{p}_score' for p in partners[1:]]
        all_cols = [base_col] + partner_cols

        erc_name = f'{p0}_erc_{"_".join(partners[1:])}'
        erc_col = f'{erc_name}_score'

        df[erc_col] = compute_erc(
            df, all_cols,
            group_col=group_col, sigma_frac=sigma_frac,
        )
        erc_names.append(erc_name)

    return erc_names
