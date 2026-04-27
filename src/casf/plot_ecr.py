"""
CASF ERC Benchmark Combined Figure
====================================

Creates a publication-quality 3-row combined figure (no scoring/ranking):
- Row 1: Docking success rate | Binding funnel heatmap
- Row 2: Forward screening | Reverse screening
- Row 3: EF@1% | EF@2% | EF@5%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.casf.stats import bca_mean_ci, friedman_nemenyi

# ============================================================================
# Configuration
# ============================================================================

CI_COLOR = '#2166AC'
POINT_COLOR = '#1A1A1A'
STRIPE_COLOR = '#E5E5E5'
BAR_COLORS = ['#4393C3', '#92C5DE', '#D1E5F0']
HEATMAP_CMAP = 'mako_r'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 5,
    'axes.titlesize': 10,
    'axes.labelsize': 7,
    'xtick.labelsize': 4,
    'ytick.labelsize': 3,
    'legend.fontsize': 5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

CATEGORY_COLORS = {
    'empirical': '#E63946',
    'physical': '#457B9D',
    'kbp': '#2A9D8F',
    'consensus': 'black',
}

SCORE_CATEGORY = {
    'DESPOT-leaky': 'consensus',
    'ΔVinaRF20': 'consensus',
    'ChemPLP': 'consensus',
    'GlideScore-SP': 'consensus',
    # ERC combinations
    'DESPOT-leaky + ΔVinaRF20 (ERC)': 'consensus',
    'DESPOT-leaky + ΔVinaRF20 + ChemPLP (ERC)': 'consensus',
    'DESPOT-leaky + ΔVinaRF20 + ChemPLP + GlideScore-SP (ERC)': 'consensus',
}

# ============================================================================
# Helper Functions
# ============================================================================

def build_significance_groups(stats_dict, pvals, stat_key='mean'):
    """Build groups of methods that are not significantly different."""
    methods_sorted = sorted(
        stats_dict.keys(),
        key=lambda m: stats_dict[m]['mean'] if stat_key == 'mean' else stats_dict[m]['estimate'],
        reverse=True,
    )

    groups = []
    current_group = [methods_sorted[0]]

    for i in range(1, len(methods_sorted)):
        curr = methods_sorted[i]
        if pvals.loc[curr, current_group[0]] > 0.1:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]

    groups.append(current_group)
    return methods_sorted, groups


def plot_ci_horizontal(name_map, ax, stats_dict, methods_sorted, groups, xlabel,
                       stat_key='mean', clean_names=True):
    """Plot horizontal confidence interval plot with significance stripes."""
    y_positions = {m: i for i, m in enumerate(methods_sorted)}
    y = np.arange(len(methods_sorted))

    for g_idx, group in enumerate(groups):
        if g_idx % 2 == 0:
            ys = [y_positions[m] for m in group]
            y_min = min(ys) - 0.5
            y_max = max(ys) + 0.5
            ax.axhspan(y_min, y_max, color=STRIPE_COLOR, zorder=0, alpha=0.9)

    for m in methods_sorted:
        i = y_positions[m]
        point = stats_dict[m].get('mean' if stat_key == 'mean' else 'estimate')
        lo = stats_dict[m]['ci_low']
        hi = stats_dict[m]['ci_high']

        ax.plot([lo, hi], [i, i], color=CI_COLOR, lw=1.4, zorder=2, solid_capstyle='round')
        ax.plot(point, i, 'o', color=POINT_COLOR, zorder=2.5, markersize=2)

    ax.set_yticks(y)
    if clean_names:
        ax.set_yticklabels([name_map.get(m, m) for m in methods_sorted])
    else:
        ax.set_yticklabels(methods_sorted)

    for tick_label in ax.get_yticklabels():
        text = tick_label.get_text()
        cat = SCORE_CATEGORY.get(text)
        if cat:
            tick_label.set_color(CATEGORY_COLORS[cat])

    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_axisbelow(True)
    ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_stacked_bars(name_list, name_map, ax, data_arr, labels, bar_labels, title,
                      ylabel=None, show_legend=False, legend_title=None):
    """Plot stacked bar chart sorted by bottom (dark blue) values descending."""
    S = len(labels)

    sort_idx = np.argsort(data_arr[:, 0])[::-1]
    data_arr_sorted = data_arr[sort_idx]
    labels_sorted = [name_list[i] for i in sort_idx]

    x = np.arange(S)
    width = 0.65

    ax.bar(x, data_arr_sorted[:, 2], width, label=bar_labels[2],
           color=BAR_COLORS[2], edgecolor='white', linewidth=0.5)
    ax.bar(x, data_arr_sorted[:, 1], width, label=bar_labels[1],
           color=BAR_COLORS[1], edgecolor='white', linewidth=0.5)
    ax.bar(x, data_arr_sorted[:, 0], width, label=bar_labels[0],
           color=BAR_COLORS[0], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [name_map.get(f'{n}_score', n) for n in labels_sorted],
        rotation=45, ha='right',
    )

    for tick_label in ax.get_xticklabels():
        text = tick_label.get_text()
        cat = SCORE_CATEGORY.get(text)
        if cat:
            tick_label.set_color(CATEGORY_COLORS[cat])

    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight='medium', pad=8)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)

    if show_legend:
        ax.legend(title=legend_title, loc='upper left', framealpha=0.9,
                  edgecolor='none', bbox_to_anchor=(1.0, 1.05),
                  title_fontproperties={'weight': 'bold', 'size': 5})

    return ax.get_legend_handles_labels()


# ============================================================================
# Main Figure
# ============================================================================

def generate_erc_figure(
    output_path,
    dock_name_list, dock_name_list_clean,
    dock_top_arr, dock_spearman_thresholds,
    forward_top_arr, reverse_top_arr, ef_arr,
):
    """Generate the combined ERC figure (docking + screening only)."""

    print("Loading data and computing statistics...")

    score_names_ext = [f'{x}_score' for x in dock_name_list]
    name_map_ext = dict(zip(score_names_ext, dock_name_list_clean))

    # Enrichment factor statistics for 1%, 2%, 5%
    ef_dicts = []
    for ef_idx in [0, 1, 2]:
        ef_slice = ef_arr[:, :, ef_idx]
        ef_dict = {}
        for i, score in enumerate(score_names_ext):
            ef_dict[score] = bca_mean_ci(ef_slice[i, :])
        ef_dicts.append(ef_dict)

    print("Computing significance tests...")

    pvals_ef = [friedman_nemenyi(ef_dict) for ef_dict in ef_dicts]

    ef_methods_groups = []
    for ef_dict, pvals in zip(ef_dicts, pvals_ef):
        methods, groups = build_significance_groups(ef_dict, pvals, stat_key='mean')
        ef_methods_groups.append((methods, groups))

    # -------------------------------------------------------------------------
    # Step 1b: Print requested statistics
    # -------------------------------------------------------------------------
 
    despot_variants = ['GlideScore-SP', 'DESPOT + GlideScore-SP (ECR)']
 
    # Docking power: top-1, top-2, top-3
    print("\n=== Docking Power (Success Rate) ===")
    for variant in despot_variants:
        if variant in dock_name_list_clean:
            idx = dock_name_list_clean.index(variant)
            print(f"  {variant}: Top-1={dock_top_arr[idx, 0]:.3f}, "
                  f"Top-2={dock_top_arr[idx, 1]:.3f}, "
                  f"Top-3={dock_top_arr[idx, 2]:.3f}")
 
    # Forward screening power: 1%, 2%, 5%
    print("\n=== Forward Screening Power (Success Rate) ===")
    for variant in despot_variants:
        if variant in dock_name_list_clean:
            idx = dock_name_list_clean.index(variant)
            print(f"  {variant}: Top 1%={forward_top_arr[idx, 0]:.3f}, "
                  f"Top 2%={forward_top_arr[idx, 1]:.3f}, "
                  f"Top 5%={forward_top_arr[idx, 2]:.3f}")
 
    # Enrichment factor p-values between DESPOT and DESPOT-DS
    print("\n=== Enrichment Factor: DESPOT vs DESPOT-Xtal p-values ===")
    ef_labels = ['EF@1%', 'EF@2%', 'EF@5%']
    despot_score = 'glide_score'
    despot_ds_score = 'despot_crown_druglike_min_erc_glide_score'
    for ef_label, pvals in zip(ef_labels, pvals_ef):
        print(pvals.columns)
        if despot_score in pvals.index and despot_ds_score in pvals.columns:
            p = pvals.loc[despot_score, despot_ds_score]
            print(f"  {ef_label}: p={p}")

    # -------------------------------------------------------------------------
    # Create figure: 3 rows
    # -------------------------------------------------------------------------

    print("Creating figure...")

    fig = plt.figure(figsize=(7.5, 8.5))

    gs_main = gridspec.GridSpec(
        3, 1, figure=fig, height_ratios=[1.1, 1, 1], hspace=0.7,
    )

    # Row 1: Docking | Binding funnel
    gs_row1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[0], wspace=0.6, width_ratios=[1, 1],
    )
    ax_docking = fig.add_subplot(gs_row1[0])
    ax_heatmap = fig.add_subplot(gs_row1[1])

    # Row 2: Forward | Reverse screening
    gs_row2 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[1], wspace=0.6,
    )
    ax_forward = fig.add_subplot(gs_row2[0])
    ax_reverse = fig.add_subplot(gs_row2[1])

    # Row 3: EF@1% | EF@2% | EF@5%
    gs_row3 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[2], wspace=0.6,
    )
    ax_ef1 = fig.add_subplot(gs_row3[0])
    ax_ef2 = fig.add_subplot(gs_row3[1])
    ax_ef5 = fig.add_subplot(gs_row3[2])

    # -------------------------------------------------------------------------
    # Plot panels
    # -------------------------------------------------------------------------

    # Row 1: Docking success rate
    docking_labels = ['Top-1', 'Top-2', 'Top-3']
    plot_stacked_bars(
        dock_name_list, name_map_ext, ax_docking, dock_top_arr,
        score_names_ext, docking_labels, title='',
        ylabel='Success rate', show_legend=True, legend_title='Recovery',
    )
    ax_docking.set_title('(A) Docking power', loc='left', fontweight='bold', fontsize=9)
    ax_docking.set_ylim(0.4, 1.0)

    # Row 1: Binding funnel heatmap
    threshold_list = list(range(2, 11))
    row_means = dock_spearman_thresholds.mean(axis=1)
    sort_idx = np.argsort(row_means)[::-1]
    spearman_sorted = dock_spearman_thresholds[sort_idx]
    names_sorted = [score_names_ext[i] for i in sort_idx]

    sns.heatmap(
        spearman_sorted,
        xticklabels=threshold_list,
        yticklabels=[name_map_ext.get(s, s) for s in names_sorted],
        cmap=HEATMAP_CMAP, ax=ax_heatmap,
        vmin=0.3, vmax=0.8,
        cbar_kws={'label': 'Mean −ρ(RMSD, score)', 'shrink': 0.8},
    )
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0)

    for tick_label in ax_heatmap.get_yticklabels():
        text = tick_label.get_text()
        cat = SCORE_CATEGORY.get(text)
        if cat:
            tick_label.set_color(CATEGORY_COLORS[cat])

    ax_heatmap.set_xlabel('RMSD threshold (Å)')
    ax_heatmap.set_title('(B) Binding funnel', loc='left', fontweight='bold', fontsize=9)

    # Row 2: Forward screening
    forward_labels = ['Top 1%', 'Top 2%', 'Top 5%']
    plot_stacked_bars(
        dock_name_list, name_map_ext, ax_forward, forward_top_arr,
        score_names_ext, forward_labels, title='',
        ylabel='Success rate', show_legend=False,
    )
    ax_forward.set_title('(C) Forward screening', loc='left', fontweight='bold', fontsize=9)

    # Row 2: Reverse screening
    reverse_labels = ['Top 2%', 'Top 5%', 'Top 10%']
    plot_stacked_bars(
        dock_name_list, name_map_ext, ax_reverse, reverse_top_arr,
        score_names_ext, reverse_labels, title='',
        ylabel='Success rate', show_legend=False,
    )
    ax_reverse.set_title('(D) Reverse screening', loc='left', fontweight='bold', fontsize=9)

    # Shared legends for screening
    ax_forward.legend(
        title='Cutoff', loc='upper left', framealpha=0.9,
        edgecolor='none', bbox_to_anchor=(1.0, 1.05),
        title_fontproperties={'weight': 'bold', 'size': 7},
    )
    ax_reverse.legend(
        title='Cutoff', loc='upper left', framealpha=0.9,
        edgecolor='none', bbox_to_anchor=(1.0, 1.05),
        title_fontproperties={'weight': 'bold', 'size': 7},
    )

    # Row 3: Enrichment factors
    ef_titles = ['(E) EF @ 1%', '(F) EF @ 2%', '(G) EF @ 5%']
    ef_axes = [ax_ef1, ax_ef2, ax_ef5]

    for ax, (methods, groups), ef_dict, title in zip(
        ef_axes, ef_methods_groups, ef_dicts, ef_titles,
    ):
        plot_ci_horizontal(
            name_map_ext, ax, ef_dict, methods, groups,
            xlabel='Mean enrichment factor (90% CI)', stat_key='mean',
        )
        ax.set_title(title, loc='left', fontweight='bold', fontsize=9)
        ax.set_xlim(0, None)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    plt.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.05)

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")

    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PNG version saved to: {png_path}")

    plt.close(fig)
