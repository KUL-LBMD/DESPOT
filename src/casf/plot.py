"""
CASF Benchmark Combined Figure for JCIM Paper
==============================================

Creates a publication-quality 4-row combined figure:
- Row 1: Pearson correlation | Spearman correlation
- Row 2: Docking success rate | Binding funnel heatmap
- Row 3: Forward screening | Reverse screening
- Row 4: EF@1% | EF@2% | EF@5%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.casf.stats import bca_pearson_ci, bca_mean_ci, friedman_nemenyi

# ============================================================================
# Configuration
# ============================================================================


# Color scheme - professional palette
CI_COLOR = '#2166AC'  # Blue for confidence intervals
POINT_COLOR = '#1A1A1A'  # Dark gray for point estimates
STRIPE_COLOR = '#E5E5E5'  # Light gray for significance stripes
BAR_COLORS = ['#4393C3', '#92C5DE', '#D1E5F0']  # Blue gradient for bars
HEATMAP_CMAP = 'mako_r'

# Figure settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 10,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

# ============================================================================
# Helper Functions
# ============================================================================

def build_significance_groups(stats_dict, pvals, stat_key='estimate'):
    """Build groups of methods that are not significantly different."""
    if stat_key == 'estimate':
        methods_sorted = sorted(stats_dict.keys(),
                               key=lambda m: stats_dict[m]['estimate'],
                               reverse=True)
    else:
        methods_sorted = sorted(stats_dict.keys(),
                               key=lambda m: stats_dict[m]['mean'],
                               reverse=True)
    
    groups = []
    current_group = [methods_sorted[0]]
    
    for i in range(1, len(methods_sorted)):
        prev = methods_sorted[i-1]
        curr = methods_sorted[i]
        
        if pvals.loc[curr, prev] > 0.1:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]
    
    groups.append(current_group)
    return methods_sorted, groups

def plot_ci_horizontal(name_map, ax, stats_dict, methods_sorted, groups, xlabel, 
                       stat_key='estimate', clean_names=True):
    """Plot horizontal confidence interval plot with significance stripes."""
    y_positions = {m: i for i, m in enumerate(methods_sorted)}
    y = np.arange(len(methods_sorted))
    
    # Draw background stripes for significance groups
    for g_idx, group in enumerate(groups):
        if g_idx % 2 == 0:
            y_min = y_positions[group[-1]] - 0.5
            y_max = y_positions[group[0]] + 0.5
            ax.axhspan(y_min, y_max, color=STRIPE_COLOR, zorder=0, alpha=0.9)
    
    # Draw confidence intervals and point estimates
    for m in methods_sorted:
        i = y_positions[m]
        if stat_key == 'estimate':
            point = stats_dict[m]['estimate']
        else:
            point = stats_dict[m]['mean']
        lo = stats_dict[m]['ci_low']
        hi = stats_dict[m]['ci_high']
        
        # CI line
        ax.plot([lo, hi], [i, i], color=CI_COLOR, lw=2.0, zorder=2, solid_capstyle='round')
        # Point estimate
        ax.plot(point, i, 'o', color=POINT_COLOR, zorder=2.5, markersize=4)
    
    # Axes formatting
    ax.set_yticks(y)
    if clean_names:
        ax.set_yticklabels([name_map.get(m, m) for m in methods_sorted])
    else:
        ax.set_yticklabels(methods_sorted)

    # Make DESPOT labels bold
    for tick_label in ax.get_yticklabels():
        if 'DESPOT' in tick_label.get_text():
            tick_label.set_fontweight('bold')

    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_stacked_bars(name_list, name_map, ax, data_arr, labels, bar_labels, title, ylabel=None, 
                      show_legend=False, legend_title=None):
    """Plot stacked bar chart sorted by dark blue (bottom) values in descending order."""
    S = len(labels)
    
    # Sort indices based on dark blue values (column 0) in descending order
    sort_idx = np.argsort(data_arr[:, 0])[::-1]
    data_arr_sorted = data_arr[sort_idx]
    labels_sorted = [name_list[i] for i in sort_idx]
    
    x = np.arange(S)
    width = 0.65
    
    # Plot bars (stacked - largest at bottom)
    ax.bar(x, data_arr_sorted[:, 2], width, label=bar_labels[2], 
           color=BAR_COLORS[2], edgecolor='white', linewidth=0.5)
    ax.bar(x, data_arr_sorted[:, 1], width, label=bar_labels[1], 
           color=BAR_COLORS[1], edgecolor='white', linewidth=0.5)
    ax.bar(x, data_arr_sorted[:, 0], width, label=bar_labels[0], 
           color=BAR_COLORS[0], edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([name_map.get(f'{n}_score', n) for n in labels_sorted], 
                       rotation=45, ha='right')
    
    # Make DESPOT labels bold
    for tick_label in ax.get_xticklabels():
        if 'DESPOT' in tick_label.get_text():
            tick_label.set_fontweight('bold')

    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight='medium', pad=8)
    
    if ylabel:
        ax.set_ylabel(ylabel)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    
    if show_legend:
        ax.legend(title=legend_title, loc='upper left', framealpha=0.9, edgecolor='none', bbox_to_anchor=(1.0, 1.05), title_fontproperties={'weight': 'bold', 'size': 7})
    
    return ax.get_legend_handles_labels()

# ============================================================================
# Main Script
# ============================================================================

def generate_combined_figure(output_path, name_list, name_list_clean, score_df, spearman_arr, dock_top_arr, dock_spearman_thresholds, forward_top_arr, reverse_top_arr, ef_arr):
    """Generate the complete combined figure."""
    
    print("Loading data and computing statistics...")
    
    # -------------------------------------------------------------------------
    # Step 1: Load all data
    # -------------------------------------------------------------------------

    score_names = [f'{x}_score' for x in name_list]
    name_map = dict(zip(score_names, name_list_clean))    
    
    # Scoring power (Pearson)
    affinities = score_df['logKa'].values
    score_dict = {}
    for score in score_names:
        score_values = score_df[score]
        score_dict[score] = bca_pearson_ci(score_values, affinities)
    
    # Ranking power (Spearman)
    rank_dict = {}
    for i, score in enumerate(score_names):
        rank_values = spearman_arr[i, :]
        rank_dict[score] = bca_mean_ci(rank_values)
    
    # Compute EF statistics for 1%, 2%, 5%
    ef_dicts = []
    for ef_idx in [0, 1, 2]:  # 1%, 2%, 5%
        ef_slice = ef_arr[:, :, ef_idx]
        ef_dict = {}
        for i, score in enumerate(score_names):
            ef_values = ef_slice[i, :]
            ef_dict[score] = bca_mean_ci(ef_values)
        ef_dicts.append(ef_dict)
    
    print("Computing significance tests...")
    
    # Significance tests
    pvals_pearson = friedman_nemenyi(score_dict)
    pvals_spearman = friedman_nemenyi(rank_dict)
    pvals_ef = [friedman_nemenyi(ef_dict) for ef_dict in ef_dicts]
    
    # Build significance groups
    methods_pearson, groups_pearson = build_significance_groups(
        score_dict, pvals_pearson, stat_key='estimate')
    methods_spearman, groups_spearman = build_significance_groups(
        rank_dict, pvals_spearman, stat_key='mean')
    
    ef_methods_groups = []
    for ef_dict, pvals in zip(ef_dicts, pvals_ef):
        methods, groups = build_significance_groups(ef_dict, pvals, stat_key='mean')
        ef_methods_groups.append((methods, groups))
    
    # -------------------------------------------------------------------------
    # Step 2: Create figure with GridSpec
    # -------------------------------------------------------------------------
    
    print("Creating figure...")
    
    fig = plt.figure(figsize=(7.5, 11))
    
    # Create main GridSpec: 4 rows with different height ratios
    gs_main = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1, 1.1, 1, 1],
                                hspace=0.7)
    
    # Row 1: Pearson | Spearman (2 columns)
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], 
                                                wspace=0.4)
    ax_pearson = fig.add_subplot(gs_row1[0])
    ax_spearman = fig.add_subplot(gs_row1[1])
    
    # Row 2: Docking | Heatmap (2 columns, unequal widths)
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], 
                                                wspace=0.6, width_ratios=[1, 1])
    ax_docking = fig.add_subplot(gs_row2[0])
    ax_heatmap = fig.add_subplot(gs_row2[1])
    
    # Row 3: Forward | Reverse screening (2 columns)
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[2], 
                                                wspace=0.6)
    ax_forward = fig.add_subplot(gs_row3[0])
    ax_reverse = fig.add_subplot(gs_row3[1])
    
    # Row 4: EF@1% | EF@2% | EF@5% (3 columns)
    gs_row4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[3], 
                                                wspace=0.6)
    ax_ef1 = fig.add_subplot(gs_row4[0])
    ax_ef2 = fig.add_subplot(gs_row4[1])
    ax_ef5 = fig.add_subplot(gs_row4[2])
    
    # -------------------------------------------------------------------------
    # Step 3: Plot each panel
    # -------------------------------------------------------------------------
    
    # Row 1: Pearson correlation
    plot_ci_horizontal(name_map, ax_pearson, score_dict, methods_pearson, groups_pearson,
                       xlabel='Pearson correlation (90% CI)', stat_key='estimate')
    ax_pearson.set_title('(A) Scoring power', loc='left', fontweight='bold', fontsize=9)
    ax_pearson.set_xlim(0.0, 1.0)
    
    # Row 1: Spearman correlation
    plot_ci_horizontal(name_map, ax_spearman, rank_dict, methods_spearman, groups_spearman,
                       xlabel='Mean Spearman ρ (90% CI)', stat_key='mean')
    ax_spearman.set_title('(B) Ranking power', loc='left', fontweight='bold', fontsize=9)
    ax_spearman.set_xlim(0.0, 1.0)
    
    # Row 2: Docking success rate
    docking_labels = ['Top-1', 'Top-2', 'Top-3']
    plot_stacked_bars(name_list, name_map, ax_docking, dock_top_arr, score_names, docking_labels,
                      title='', ylabel='Success rate', show_legend=True,
                      legend_title='Recovery')
    ax_docking.set_title('(C) Docking power', loc='left', fontweight='bold', fontsize=9)
    ax_docking.set_ylim(0.4, 1.0)

    # Row 2: Binding funnel heatmap
    threshold_list = list(range(2, 11))
    
    # Sort rows by descending row mean
    row_means = dock_spearman_thresholds.mean(axis=1)
    sort_idx = np.argsort(row_means)[::-1]
    spearman_thresholds_sorted = dock_spearman_thresholds[sort_idx]
    score_names_sorted = [score_names[i] for i in sort_idx]
    
    hm = sns.heatmap(spearman_thresholds_sorted, 
                     xticklabels=threshold_list,
                     yticklabels=[name_map.get(s, s) for s in score_names_sorted],
                     cmap=HEATMAP_CMAP, ax=ax_heatmap, 
                     vmin=0.3, vmax=0.8,
                     cbar_kws={'label': 'Mean −ρ(RMSD, score)', 'shrink': 0.8})
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0)

    # Make DESPOT labels bold
    for tick_label in ax_heatmap.get_yticklabels():
        if 'DESPOT' in tick_label.get_text():
            tick_label.set_fontweight('bold')

    ax_heatmap.set_xlabel('RMSD threshold (Å)')
    ax_heatmap.set_title('(D) Binding funnel', loc='left', fontweight='bold', fontsize=9)
    
    # Row 3: Forward screening
    forward_labels = ['Top 1%', 'Top 2%', 'Top 5%']
    plot_stacked_bars(name_list, name_map, ax_forward, forward_top_arr, score_names, forward_labels,
                      title='', ylabel='Success rate', show_legend=False)
    ax_forward.set_title('(E) Forward screening', loc='left', fontweight='bold', fontsize=9)

    # Row 3: Reverse screening
    reverse_labels = ['Top 2%', 'Top 5%', 'Top 10%']
    handles, labels = plot_stacked_bars(name_list, name_map, ax_reverse, reverse_top_arr, score_names, 
                                         reverse_labels, title='', ylabel = 'Success rate', show_legend=False)
    ax_reverse.set_title('(F) Reverse screening', loc='left', fontweight='bold', fontsize=9)
    
    # Add shared legend for screening plots
    ax_forward.legend(title='Cutoff', loc='upper left', framealpha=0.9, edgecolor='none', bbox_to_anchor=(1.0, 1.05), title_fontproperties={'weight': 'bold', 'size': 7})
    ax_reverse.legend(title='Cutoff', loc='upper left', framealpha=0.9, edgecolor='none', bbox_to_anchor=(1.0, 1.05), title_fontproperties={'weight': 'bold', 'size': 7})
    
    # Row 4: Enrichment factors
    ef_titles = ['(G) EF @ 1%', '(H) EF @ 2%', '(I) EF @ 5%']
    ef_axes = [ax_ef1, ax_ef2, ax_ef5]
    
    for idx, (ax, (methods, groups), ef_dict, title) in enumerate(
            zip(ef_axes, ef_methods_groups, ef_dicts, ef_titles)):
        plot_ci_horizontal(name_map, ax, ef_dict, methods, groups,
                           xlabel='Mean enrichment factor (90% CI)', stat_key='mean')
        ax.set_title(title, loc='left', fontweight='bold', fontsize=9)
        ax.set_xlim(0, None)
    
    # -------------------------------------------------------------------------
    # Step 4: Final adjustments and save
    # -------------------------------------------------------------------------
    
    # Adjust layout
    plt.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.05)
    
    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    # Also save as PNG for quick viewing
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PNG version saved to: {png_path}")
    
    plt.close(fig)
