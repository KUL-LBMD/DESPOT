"""
This script runs the 3 DESPOT variants on CASF-2016, and computes and plots the statistics
(with comparison against some competitors).

ERC (Exponential Rank Consensus) combinations are computed for DESPOT paired
with DeltaVina, GlideScore, ChemPLP, and AutoDockVina.
"""

from src.config import DATA_DIR
from src.casf.run_despot import run_scoring_despot, run_docking_despot, run_screening_despot
from src.casf.run_korp import run_scoring_korp, run_docking_korp, run_screening_korp
from src.casf.metrics import (
    get_scoring_values, get_ranking_values, get_docking_values,
    get_screening_values, get_enrichment_factors,
)
from src.casf.plot import generate_combined_figure

import numpy as np
import pandas as pd
import os
import argparse

# ============================================================================
# Name lists
# ============================================================================

NAME_LIST = [
    'despot_crown', 'despot_hiqbind_xtal', 'korp_crown', 'korp_hiqbind_xtal', 'drugscore_crown', 'drugscore_hiqbind_xtal',
    'despot_crown_xtal', 'korp_crown_xtal', 'drugscore_crown_xtal'
]

NAME_LIST_CLEAN = [
    'DESPOT (C)', 'DESPOT (H)', 'DESPOT-screen (C)', 'DESPOT-screen (H)', 'DESPOT-iso (C)', 'DESPOT-iso (H)',
    'DESPOT (C-xtal)', 'DESPOT-screen (C-xtal)', 'DESPOT-iso (C-xtal)'
]

NAME_MAP = {k: v for k, v in zip(NAME_LIST, NAME_LIST_CLEAN)}

CATEGORY_COLORS = {
    'empirical': '#E63946',
    'physical': '#457B9D',
    'kbp': '#2A9D8F',
}

SCORE_CATEGORY = {
    'DrugScoreX': 'kbp',
    'ASP': 'kbp',
    'AutoDockVina': 'empirical',
    'DrugScoreCSD': 'kbp',
    'DrugScore2018': 'kbp',
    'GlideScore-SP': 'empirical',
    'GoldScore': 'physical',
    'PMF04': 'kbp',
    'ChemScore': 'empirical',
    'ChemPLP': 'empirical',
    'GBVI-WSA-dG': 'physical',
    'ΔVinaRF20': 'empirical',
    'KORP-PL': 'kbp',
    'GNINA-score': 'empirical',
    'GNINA-affinity': 'empirical',
    'GNINA-screening': 'empirical'
}

# ============================================================================
# ERC configuration
# ============================================================================

#ERC_PARTNERS = ['dsx', 'asp', 'drugscore_csd', 'drugscore2018', 'gold', 'pmf', 'chemscore', 'gbvi_wsa', 'deltavina', 'glide', 'chemplp', 'autodockvina']
ERC_PARTNERS = []
ERC_CONFIG = {
    'base': 'despot_crown_leaky',
    'partner_combos': ERC_PARTNERS,
    'sigma_frac': 0.05,
}

# Clean display names for ERC combinations
ERC_PARTNER_CLEAN = {k: NAME_MAP[k] for k in ERC_PARTNERS}

ERC_NAMES_CLEAN = [
    f'DESPOT+{ERC_PARTNER_CLEAN[p]} (ERC)'
    for p in ERC_PARTNERS
]

Z_PARTNERS = []

Z_CONFIG = {
    'base': 'despot',
    'partners': Z_PARTNERS,
}

# Clean display names for ERC combinations
Z_PARTNER_CLEAN = {k: NAME_MAP[k] for k in Z_PARTNERS}

Z_NAMES_CLEAN = [
    f'DESPOT+{Z_PARTNER_CLEAN[p]} (Z)'
    for p in Z_PARTNERS
]

### Step 2: Get benchmark metrics (with ERC for docking & screening) ###
score_df, score_names_ext = get_scoring_values(NAME_LIST, z_config = None)
rank_spearman_arr = get_ranking_values(NAME_LIST, z_config = None)
dock_top_arr, dock_spearman_thresholds, dock_names_ext = get_docking_values(NAME_LIST, erc_config=ERC_CONFIG)
screen_df, forward_top_arr, reverse_top_arr, screen_names_ext = get_screening_values(NAME_LIST, erc_config=ERC_CONFIG)
ef_arr = get_enrichment_factors(screen_df, screen_names_ext)

### Step 3: Compute and plot statistics ###

# Scoring & ranking use the original name lists (no ERC).
# Docking, screening & enrichment use the extended lists.
generate_combined_figure(
    'casf_combined.pdf',
    # scoring / ranking (rows 1-2 left panels)
    score_name_list = NAME_LIST,
    score_name_list_clean=NAME_LIST_CLEAN,
    score_df=score_df,
    spearman_arr=rank_spearman_arr,
    # docking (row 2)
    dock_name_list=dock_names_ext,
    dock_name_list_clean=NAME_LIST_CLEAN + ERC_NAMES_CLEAN,
    dock_top_arr=dock_top_arr,
    dock_spearman_thresholds=dock_spearman_thresholds,
    # screening & enrichment (rows 3-4)
    forward_top_arr=forward_top_arr,
    reverse_top_arr=reverse_top_arr,
    ef_arr=ef_arr,
)
