"""
Plotting Utilities
==================

Shared visualization functions for analysis scripts.

This module provides:
- Publication-quality plot styling
- Standard plot types (forest, scatter, model comparison)
- Color palettes and formatting

Usage:
    from analysis.visualization import (
        set_publication_style,
        create_model_comparison_plot,
        create_forest_plot,
        create_scatter_with_regression
    )
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Color palette for consistent styling
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'accent': '#F18F01',       # Orange
    'success': '#2ECC71',      # Green
    'warning': '#F39C12',      # Yellow
    'danger': '#E74C3C',       # Red
    'neutral': '#95A5A6',      # Gray

    # Gender colors
    'male': '#3498DB',
    'female': '#E74C3C',

    # Model comparison
    'model0': '#95A5A6',
    'model1': '#3498DB',
    'model2': '#E74C3C',
    'model3': '#2ECC71',
}

# Significance markers
SIG_MARKERS = {
    0.001: '***',
    0.01: '**',
    0.05: '*',
    1.0: 'ns'
}


def get_significance_marker(p: float) -> str:
    """Get significance marker for p-value."""
    for threshold, marker in SIG_MARKERS.items():
        if p < threshold:
            return marker
    return 'ns'


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (8, 6),
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


# =============================================================================
# MODEL COMPARISON PLOTS
# =============================================================================

def create_model_comparison_plot(
    results: pd.DataFrame,
    output_path: Optional[Path] = None,
    max_plots: int = 4
) -> plt.Figure:
    """
    Create model comparison visualization (R² bar charts).

    Args:
        results: DataFrame with model0_r2, model1_r2, model2_r2, model3_r2 columns
        output_path: Path to save figure (optional)
        max_plots: Maximum number of subplots

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    n_plots = min(len(results), max_plots)
    if n_plots == 0:
        return None

    nrows = (n_plots + 1) // 2
    ncols = 2 if n_plots > 1 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))

    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    model_labels = ['Covariates\nOnly', '+ DASS', '+ UCLA', '+ UCLA×Gender']
    colors = [COLORS['model0'], COLORS['model1'], COLORS['model2'], COLORS['model3']]

    for i, (_, row) in enumerate(results.head(max_plots).iterrows()):
        ax = axes[i]

        r2_values = [
            row.get('model0_r2', 0),
            row.get('model1_r2', 0),
            row.get('model2_r2', 0),
            row.get('model3_r2', 0)
        ]

        bars = ax.bar(range(4), r2_values, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        # Add ΔR² annotations
        delta_keys = [None, 'delta_r2_dass', 'delta_r2_ucla', 'delta_r2_interaction']
        p_keys = [None, 'p_dass', 'p_ucla', 'p_interaction']

        for j in range(1, 4):
            delta = row.get(delta_keys[j], 0)
            p = row.get(p_keys[j], 1.0)
            sig = get_significance_marker(p)

            ax.text(j, r2_values[j] + 0.01, f'ΔR²={delta:.3f}\n{sig}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(4))
        ax.set_xticklabels(model_labels, fontsize=10)
        ax.set_ylabel('R²', fontsize=11, fontweight='bold')

        outcome = row.get('outcome', row.get('analysis_name', f'Analysis {i+1}'))
        n = row.get('n', 0)
        ax.set_title(f"{outcome}\n(N={n:.0f})", fontsize=12, fontweight='bold')

        ax.set_ylim(0, max(r2_values) * 1.25)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


# =============================================================================
# FOREST PLOTS
# =============================================================================

def create_forest_plot(
    effects: pd.DataFrame,
    effect_col: str = 'beta',
    ci_low_col: str = 'ci_low',
    ci_high_col: str = 'ci_high',
    label_col: str = 'label',
    output_path: Optional[Path] = None,
    title: str = "Effect Sizes (95% CI)"
) -> plt.Figure:
    """
    Create forest plot of effect sizes with confidence intervals.

    Args:
        effects: DataFrame with effect estimates and CIs
        effect_col: Column name for point estimates
        ci_low_col: Column name for CI lower bounds
        ci_high_col: Column name for CI upper bounds
        label_col: Column name for labels
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, max(4, len(effects) * 0.5)))

    y_positions = range(len(effects))

    # Plot effect points and CIs
    for i, (_, row) in enumerate(effects.iterrows()):
        effect = row[effect_col]
        ci_low = row.get(ci_low_col, effect - 0.5)
        ci_high = row.get(ci_high_col, effect + 0.5)

        # Determine color based on significance
        p = row.get('p', 1.0)
        color = COLORS['success'] if p < 0.05 else COLORS['neutral']

        # CI line
        ax.hlines(i, ci_low, ci_high, color=color, linewidth=2)

        # Effect point
        ax.scatter(effect, i, color=color, s=100, zorder=5, edgecolor='black')

    # Reference line at 0
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(effects[label_col])
    ax.set_xlabel('Effect Size (β)', fontweight='bold')
    ax.set_title(title, fontweight='bold')

    # Legend
    sig_patch = mpatches.Patch(color=COLORS['success'], label='p < 0.05')
    ns_patch = mpatches.Patch(color=COLORS['neutral'], label='p ≥ 0.05')
    ax.legend(handles=[sig_patch, ns_patch], loc='lower right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


# =============================================================================
# SCATTER PLOTS
# =============================================================================

def create_scatter_with_regression(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    output_path: Optional[Path] = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None
) -> plt.Figure:
    """
    Create scatter plot with regression line(s).

    Args:
        data: DataFrame with x and y columns
        x: X-axis column name
        y: Y-axis column name
        hue: Grouping variable (e.g., 'gender_male')
        output_path: Path to save figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    if hue:
        palette = {0: COLORS['female'], 1: COLORS['male']}
        sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=palette,
                       alpha=0.6, ax=ax, legend=True)

        for hue_val, color in palette.items():
            subset = data[data[hue] == hue_val]
            if len(subset) > 2:
                sns.regplot(data=subset, x=x, y=y, scatter=False,
                           color=color, ax=ax, line_kws={'linewidth': 2})
    else:
        sns.scatterplot(data=data, x=x, y=y, color=COLORS['primary'],
                       alpha=0.6, ax=ax)
        sns.regplot(data=data, x=x, y=y, scatter=False,
                   color=COLORS['primary'], ax=ax, line_kws={'linewidth': 2})

    ax.set_xlabel(xlabel or x, fontweight='bold')
    ax.set_ylabel(ylabel or y, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold')

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = ['Female' if l == '0' else 'Male' if l == '1' else l for l in labels]
        ax.legend(handles, new_labels, title='Gender')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


# =============================================================================
# DISTRIBUTION PLOTS
# =============================================================================

def create_distribution_comparison(
    data: pd.DataFrame,
    x: str,
    group: str,
    output_path: Optional[Path] = None,
    title: str = None
) -> plt.Figure:
    """
    Create distribution comparison plot (violin + box).

    Args:
        data: DataFrame with variable and grouping columns
        x: Variable column name
        group: Grouping column name
        output_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    palette = {0: COLORS['female'], 1: COLORS['male']}
    sns.violinplot(data=data, x=group, y=x, palette=palette, ax=ax, alpha=0.7)
    sns.boxplot(data=data, x=group, y=x, palette=palette, ax=ax,
               width=0.2, boxprops={'zorder': 2})

    ax.set_xticklabels(['Female', 'Male'])
    ax.set_xlabel('Gender', fontweight='bold')
    ax.set_ylabel(x, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


# =============================================================================
# HEATMAPS
# =============================================================================

def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Correlation Matrix",
    annot: bool = True
) -> plt.Figure:
    """
    Create correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame
        output_path: Path to save figure
        title: Plot title
        annot: Show correlation values

    Returns:
        matplotlib Figure
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f',
               cmap='RdBu_r', center=0, vmin=-1, vmax=1,
               square=True, ax=ax,
               cbar_kws={'shrink': 0.8, 'label': 'Correlation'})

    ax.set_title(title, fontweight='bold', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


# =============================================================================
# SAVING UTILITIES
# =============================================================================

def save_figure(fig: plt.Figure, path: Path, formats: List[str] = None):
    """
    Save figure in multiple formats.

    Args:
        fig: matplotlib Figure
        path: Base path (without extension)
        formats: List of formats (default: ['png', 'pdf'])
    """
    if formats is None:
        formats = ['png']

    path = Path(path)

    for fmt in formats:
        output_path = path.with_suffix(f'.{fmt}')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', format=fmt)
