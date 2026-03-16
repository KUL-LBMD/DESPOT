"""
ERC Consensus Benchmark on CASF-2016
=====================================

Evaluates multi-way ERC combinations of DESPOT-min with DeltaVina, ChemPLP,
and GlideScore-SP on docking power, binding funnel, and screening metrics.

Shown methods:
  - 4 individual: DESPOT-min, ΔVinaRF20, ChemPLP, GlideScore-SP
  - 3 ERC combos: DESPOT-min+ΔVinaRF20,
                   DESPOT-min+ΔVinaRF20+ChemPLP,
                   DESPOT-min+ΔVinaRF20+ChemPLP+GlideScore-SP

No scoring or ranking power analysis is performed.
"""

from src.config import DATA_DIR
from src.casf.run_despot import run_docking, run_screening
from src.casf.metrics import (
    get_docking_values, get_screening_values, get_enrichment_factors,
)
from src.casf.plot_ecr import generate_erc_figure

import numpy as np
import pandas as pd
import os

# ============================================================================
# Name lists – only the 4 individual scoring functions
# ============================================================================

NAME_LIST = [
    'despot_crown_druglike_min',
    'deltavina',
    'chemplp',
    'glide',
]

NAME_LIST_CLEAN = [
    'DESPOT',
    'ΔVinaRF20',
    'ChemPLP',
    'GlideScore-SP',
]

NAME_MAP = {k: v for k, v in zip(NAME_LIST, NAME_LIST_CLEAN)}

# ============================================================================
# ERC configuration – multi-way combinations
# ============================================================================

# Each entry is a list of partners to combine with the base score.
# This produces 2-way, 3-way, and 4-way ERC consensus scores.
ERC_PARTNER_COMBOS = [
    ['despot_crown_druglike_min', 'deltavina'],
    ['despot_crown_druglike_min', 'glide'],
    ['despot_crown_druglike_min', 'chemplp']
]

ERC_CONFIG = {
    'partner_combos': ERC_PARTNER_COMBOS,
    'sigma_frac': 0.05,
}

# Clean display names for ERC combinations
ERC_NAMES_CLEAN = [
    'DESPOT + ΔVinaRF20 (ECR)',
    'DESPOT + GlideScore-SP (ECR)',
    'DESPOT + ChemPLP (ECR)',
    'DESPOT + GlideScore-SP + ChemPLP (ECR)',
    'DESPOT + ΔVinaRF20 + ChemPLP + GlideScore-SP (ECR)',
    'GlideScore-SP + ChemPLP (ECR)',
    'ΔVinaRF20 + ChemPLP + GlideScore-SP (ECR)'
]

DATABASE = 'CROWN_druglike_min'

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':

    ### Step 1: run DESPOT on all CASF entries (uncomment if needed) ###
    # run_docking(DATABASE)
    # run_screening(n_jobs=8, database=DATABASE)

    ### Step 2: Get benchmark metrics (docking & screening only, with ERC) ###

    dock_top_arr, dock_spearman_thresholds, dock_names_ext = get_docking_values(
        NAME_LIST, erc_config=ERC_CONFIG,
    )
    screen_df, forward_top_arr, reverse_top_arr, screen_names_ext = get_screening_values(
        NAME_LIST, erc_config=ERC_CONFIG,
    )
    ef_arr = get_enrichment_factors(screen_df, screen_names_ext)

    ### Step 3: Plot ###

    generate_erc_figure(
        'casf_erc_combined.pdf',
        # docking
        dock_name_list=dock_names_ext,
        dock_name_list_clean=NAME_LIST_CLEAN + ERC_NAMES_CLEAN,
        dock_top_arr=dock_top_arr,
        dock_spearman_thresholds=dock_spearman_thresholds,
        # screening & enrichment
        forward_top_arr=forward_top_arr,
        reverse_top_arr=reverse_top_arr,
        ef_arr=ef_arr,
    )
