#!/mnt/r/VENV/dj/bin/python
"""
Action Space Distribution and Action-State Heatmap Analysis.

Generates two visualization images from robot manipulation demo data:
  1. action_distribution.png   -- 4x6 histogram grid of 23 action dims + KDE overlay
  2. action_state_heatmap.png  -- state statistics heatmap + action correlation matrix

Sampling strategy: loads file-000.parquet from chunk-000, chunk-010, chunk-020
(~600K-700K frames total) to keep memory manageable.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DATA = '/mnt/g/DATA/b1k_26/datasets/demos/data'
CHUNKS = ['chunk-000', 'chunk-010', 'chunk-020']
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 200

N_ACTION_DIMS = 23
N_STATE_DIMS = 61


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sampled_data():
    """Load file-000.parquet from three spaced-out chunks."""
    frames = []
    for c in CHUNKS:
        path = os.path.join(BASE_DATA, c, 'file-000.parquet')
        if not os.path.exists(path):
            print(f'WARNING: {path} not found, skipping', file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        frames.append(df)
        print(f'  Loaded {path}  ({len(df):,} rows)')

    if not frames:
        print('ERROR: no data loaded', file=sys.stderr)
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True)
    print(f'  Total rows: {len(data):,}')

    actions = np.stack(data['action'].values)        # (N, 23)
    states = np.stack(data['observation.state'].values)  # (N, 61)
    return data, actions, states


# ---------------------------------------------------------------------------
# Image 1: Action Distribution (4x6 grid)
# ---------------------------------------------------------------------------
def plot_action_distribution(actions):
    """4x6 histogram grid: 23 action dims + 1 KDE overlay summary."""
    fig, axes = plt.subplots(4, 6, figsize=(18, 20))
    fig.suptitle('Action Space Distribution (23 dims)',
                 fontsize=18, fontweight='bold', y=0.995)

    axes_flat = axes.flatten()

    # Per-dimension histograms
    for dim in range(N_ACTION_DIMS):
        ax = axes_flat[dim]
        vals = actions[:, dim]

        # Clip extreme outliers for better visualization (0.1-99.9 percentile)
        lo, hi = np.percentile(vals, [0.1, 99.9])
        clipped = vals[(vals >= lo) & (vals <= hi)]

        ax.hist(clipped, bins=100, color='#4C72B0', alpha=0.75,
                edgecolor='none', density=True)
        ax.set_title(f'Action Dim {dim}', fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)

        # Annotate with basic stats
        mu, sigma = np.mean(vals), np.std(vals)
        ax.text(0.97, 0.95, f'$\\mu$={mu:.3f}\n$\\sigma$={sigma:.3f}',
                transform=ax.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.8, edgecolor='#cccccc'))

    # 24th subplot: KDE overlay of all dims
    ax_kde = axes_flat[N_ACTION_DIMS]
    cmap = plt.cm.tab20
    for dim in range(N_ACTION_DIMS):
        vals = actions[:, dim]
        lo, hi = np.percentile(vals, [0.5, 99.5])
        clipped = vals[(vals >= lo) & (vals <= hi)]
        # Subsample for KDE speed
        if len(clipped) > 50000:
            clipped = np.random.default_rng(42).choice(clipped, 50000,
                                                       replace=False)
        try:
            kde = sp_stats.gaussian_kde(clipped, bw_method=0.15)
            xs = np.linspace(lo, hi, 200)
            ax_kde.plot(xs, kde(xs), color=cmap(dim / N_ACTION_DIMS),
                        alpha=0.65, linewidth=0.9)
        except Exception:
            pass  # skip if KDE fails (e.g., constant values)

    ax_kde.set_title('All Dims (KDE overlay)', fontsize=10, fontweight='bold')
    ax_kde.tick_params(labelsize=7)

    # Hide the remaining unused subplot (25th position, index 24)
    for idx in range(N_ACTION_DIMS + 1, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUTPUT_DIR, 'action_distribution.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved {out_path}')


# ---------------------------------------------------------------------------
# Image 2: Action-State Heatmap (two panels)
# ---------------------------------------------------------------------------
def plot_action_state_heatmap(actions, states):
    """Two-panel figure: state statistics heatmap + action correlation."""
    fig, (ax_state, ax_corr) = plt.subplots(1, 2, figsize=(18, 16),
                                            gridspec_kw={'width_ratios': [1, 1.3]})
    fig.suptitle('Action-State Analysis', fontsize=18, fontweight='bold',
                 y=0.995)

    # --- Panel 1: State dimensions summary heatmap ---
    stat_min = np.min(states, axis=0)    # (61,)
    stat_max = np.max(states, axis=0)
    stat_mean = np.mean(states, axis=0)
    stat_std = np.std(states, axis=0)

    stat_matrix = np.column_stack([stat_min, stat_max, stat_mean, stat_std])
    # Normalize each column for better color contrast
    stat_norm = np.zeros_like(stat_matrix)
    for col in range(4):
        col_data = stat_matrix[:, col]
        col_range = col_data.max() - col_data.min()
        if col_range > 0:
            stat_norm[:, col] = (col_data - col_data.min()) / col_range
        else:
            stat_norm[:, col] = 0.0

    im = ax_state.imshow(stat_norm, aspect='auto', cmap='YlOrRd',
                         interpolation='nearest')
    ax_state.set_xticks([0, 1, 2, 3])
    ax_state.set_xticklabels(['Min', 'Max', 'Mean', 'Std'], fontsize=10)
    ax_state.set_yticks(range(0, N_STATE_DIMS, 5))
    ax_state.set_yticklabels([f'Dim {i}' for i in range(0, N_STATE_DIMS, 5)],
                             fontsize=8)
    ax_state.set_ylabel('State Dimension', fontsize=12)
    ax_state.set_title('State Space Statistics (61 dims)',
                       fontsize=14, fontweight='bold', pad=12)

    # Annotate cells with actual (un-normalized) values
    for row in range(N_STATE_DIMS):
        for col in range(4):
            val = stat_matrix[row, col]
            # Choose text color based on background brightness
            bg = stat_norm[row, col]
            txt_color = 'white' if bg > 0.6 else 'black'
            ax_state.text(col, row, f'{val:.2f}', ha='center', va='center',
                          fontsize=4.5, color=txt_color)

    cbar1 = fig.colorbar(im, ax=ax_state, fraction=0.03, pad=0.04)
    cbar1.set_label('Normalized value', fontsize=9)

    # --- Panel 2: Action correlation matrix ---
    corr = np.corrcoef(actions.T)  # (23, 23)

    mask = np.zeros_like(corr, dtype=bool)
    # No mask -- show full matrix for completeness

    sns.heatmap(corr, ax=ax_corr, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True,
                linewidths=0.3, linecolor='#eeeeee',
                cbar_kws={'label': 'Pearson r', 'shrink': 0.7},
                xticklabels=[str(i) for i in range(N_ACTION_DIMS)],
                yticklabels=[str(i) for i in range(N_ACTION_DIMS)],
                annot=False, fmt='.1f')

    # Annotate diagonal explicitly with "1.0"
    for i in range(N_ACTION_DIMS):
        ax_corr.text(i + 0.5, i + 0.5, '1.0', ha='center', va='center',
                     fontsize=6, fontweight='bold', color='black')

    ax_corr.set_title('Action Inter-Dimension Correlation',
                      fontsize=14, fontweight='bold', pad=12)
    ax_corr.set_xlabel('Action Dimension', fontsize=11)
    ax_corr.set_ylabel('Action Dimension', fontsize=11)
    ax_corr.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUTPUT_DIR, 'action_state_heatmap.png')
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('Loading sampled data ...')
    _data, actions, states = load_sampled_data()
    print(f'  actions shape:  {actions.shape}')
    print(f'  states shape:   {states.shape}')

    print('Plotting action distributions ...')
    plot_action_distribution(actions)

    print('Plotting action-state heatmaps ...')
    plot_action_state_heatmap(actions, states)

    print('Done.')


if __name__ == '__main__':
    main()
