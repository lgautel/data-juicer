#!/mnt/r/VENV/dj/bin/python
"""
BEHAVIOR-1K Reward Analysis
============================
Three-panel analysis of reward distribution, temporal positioning,
and per-task success rates across all 100 chunks.

Generates: reward_analysis.png
"""
import os
import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['font.family'] = ['DejaVu Sans']

# ---------------------------------------------------------------------------
# Palette (from dataviz reference, light mode)
# ---------------------------------------------------------------------------
CAT_BLUE   = '#2a78d6'
CAT_AQUA   = '#1baf7a'
CAT_YELLOW = '#eda100'
CAT_RED    = '#e34948'

INK_PRIMARY   = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED     = '#898781'
GRIDLINE      = '#e1e0d9'
BASELINE      = '#c3c2b7'
SURFACE       = '#fcfcfb'
GOOD_TEXT     = '#006300'

# ---------------------------------------------------------------------------
# 1. Load episode metadata
# ---------------------------------------------------------------------------
print('Loading episode metadata ...')
ep_dir = '/mnt/g/DATA/b1k_26/datasets/demos/meta/episodes'
all_eps = []
for chunk in sorted(os.listdir(ep_dir)):
    for f in sorted(glob.glob(os.path.join(ep_dir, chunk, '*.parquet'))):
        all_eps.append(pd.read_parquet(f))
eps = pd.concat(all_eps, ignore_index=True)
eps['task_name'] = eps['tasks'].apply(lambda x: x[0] if isinstance(x, list) else x)
print(f'  Episodes loaded: {len(eps)}')

# Task name lookup
tasks_df = pd.read_parquet('/mnt/g/DATA/b1k_26/datasets/demos/meta/tasks.parquet')

# Map task_index -> task_name from tasks parquet
task_idx_to_name = dict(zip(tasks_df['task_index'], tasks_df.index))

# Episode length lookup: episode_index -> length
ep_length = eps.set_index('episode_index')['length'].to_dict()

# ---------------------------------------------------------------------------
# 2. Load reward data from all chunk file-000 parquets
# ---------------------------------------------------------------------------
print('Loading reward data from all chunks ...')
data_dir = '/mnt/g/DATA/b1k_26/datasets/demos/data'
cols = ['next.reward', 'episode_index', 'task_index', 'frame_index']
all_data = []
chunks = sorted(os.listdir(data_dir))
for i, chunk in enumerate(chunks):
    fp = os.path.join(data_dir, chunk, 'file-000.parquet')
    if os.path.exists(fp):
        df = pd.read_parquet(fp, columns=cols)
        all_data.append(df)
    if (i + 1) % 20 == 0:
        print(f'  ... loaded {i + 1}/{len(chunks)} chunks')
data = pd.concat(all_data, ignore_index=True)
print(f'  Total rows: {len(data):,}')

# ---------------------------------------------------------------------------
# 3. Compute statistics
# ---------------------------------------------------------------------------
reward = data['next.reward']

# Panel 1: Value distribution
val_counts = reward.value_counts().sort_index()
unique_vals = sorted(val_counts.index)
counts = [val_counts.get(v, 0) for v in unique_vals]
labels = [f'{v:+g}' if v != 0 else '0' for v in unique_vals]

# Panel 2: Relative position of non-zero rewards
nonzero = data[data['next.reward'] != 0].copy()
nonzero['ep_length'] = nonzero['episode_index'].map(ep_length)
nonzero = nonzero.dropna(subset=['ep_length'])
nonzero['rel_pos'] = nonzero['frame_index'] / nonzero['ep_length']
# Clip to [0, 1] for safety
nonzero['rel_pos'] = nonzero['rel_pos'].clip(0.0, 1.0)

# Panel 3: Per-task success/failure rate
task_rewards = data[data['next.reward'] != 0].copy()
# For each (task_index, episode_index), get the max reward (+1 = success, -1 = fail)
ep_outcome = (
    task_rewards
    .groupby(['task_index', 'episode_index'])['next.reward']
    .max()
    .reset_index()
)
ep_outcome['success'] = (ep_outcome['next.reward'] > 0).astype(int)
ep_outcome['failure'] = (ep_outcome['next.reward'] < 0).astype(int)

# Also count episodes with no reward signal at all per task
all_ep_per_task = (
    data
    .groupby('task_index')['episode_index']
    .nunique()
    .reset_index()
    .rename(columns={'episode_index': 'total_episodes'})
)

task_stats = (
    ep_outcome
    .groupby('task_index')
    .agg(successes=('success', 'sum'), failures=('failure', 'sum'))
    .reset_index()
)
task_stats = task_stats.merge(all_ep_per_task, on='task_index', how='right').fillna(0)
task_stats['successes'] = task_stats['successes'].astype(int)
task_stats['failures'] = task_stats['failures'].astype(int)
task_stats['success_rate'] = task_stats['successes'] / task_stats['total_episodes']
task_stats['failure_rate'] = task_stats['failures'] / task_stats['total_episodes']
task_stats['task_name'] = task_stats['task_index'].map(task_idx_to_name)
task_stats['task_name'] = task_stats['task_name'].fillna(
    task_stats['task_index'].astype(str)
)

# Sort by success rate descending, take top 30
top30 = task_stats.sort_values('success_rate', ascending=False).head(30).copy()
top30 = top30.sort_values('success_rate', ascending=True)  # bottom-to-top in plot

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(16, 14))
fig.patch.set_facecolor(SURFACE)

# ---- Panel 1: Reward Value Distribution (log scale) ----
ax1 = axes[0]
ax1.set_facecolor(SURFACE)
bar_colors = []
for v in unique_vals:
    if v > 0:
        bar_colors.append(CAT_AQUA)
    elif v < 0:
        bar_colors.append(CAT_RED)
    else:
        bar_colors.append(CAT_BLUE)

bars1 = ax1.bar(labels, counts, color=bar_colors, width=0.5,
                edgecolor=SURFACE, linewidth=2)
ax1.set_yscale('log')
ax1.set_ylabel('Count (log scale)', fontsize=12, color=INK_SECONDARY)
ax1.set_title('Reward Value Distribution', fontsize=14,
              fontweight='bold', color=INK_PRIMARY, pad=12)

# Direct labels on bars (selective: all 3 bars is fine since there are only 3)
for bar, cnt in zip(bars1, counts):
    if cnt > 0:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 cnt * 1.3,
                 f'{cnt:,}',
                 ha='center', va='bottom', fontsize=11,
                 fontweight='bold', color=INK_PRIMARY)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color(BASELINE)
ax1.spines['bottom'].set_color(BASELINE)
ax1.tick_params(colors=INK_MUTED)
ax1.yaxis.grid(True, color=GRIDLINE, linewidth=0.5)
ax1.set_axisbelow(True)

# Annotation: sparsity
total = sum(counts)
zero_ct = val_counts.get(0, 0)
pct_zero = zero_ct / total * 100 if total > 0 else 0
ax1.text(0.98, 0.92,
         f'{pct_zero:.2f}% of frames have reward = 0',
         transform=ax1.transAxes, ha='right', va='top',
         fontsize=10, color=INK_SECONDARY,
         bbox=dict(boxstyle='round,pad=0.4', facecolor=SURFACE,
                   edgecolor=GRIDLINE, linewidth=0.5))

# ---- Panel 2: Non-zero reward position within episodes ----
ax2 = axes[1]
ax2.set_facecolor(SURFACE)

pos_rewards = nonzero[nonzero['next.reward'] > 0]['rel_pos']
neg_rewards = nonzero[nonzero['next.reward'] < 0]['rel_pos']

bins = np.linspace(0, 1, 51)
if len(pos_rewards) > 0:
    ax2.hist(pos_rewards, bins=bins, color=CAT_AQUA, alpha=0.75,
             edgecolor=SURFACE, linewidth=1, label='+1 (success)')
if len(neg_rewards) > 0:
    ax2.hist(neg_rewards, bins=bins, color=CAT_RED, alpha=0.65,
             edgecolor=SURFACE, linewidth=1, label='-1 (failure)')

ax2.set_xlabel('Relative Position in Episode (frame_index / episode_length)',
               fontsize=12, color=INK_SECONDARY)
ax2.set_ylabel('Count', fontsize=12, color=INK_SECONDARY)
ax2.set_title('Temporal Position of Non-Zero Rewards Within Episodes',
              fontsize=14, fontweight='bold', color=INK_PRIMARY, pad=12)
ax2.set_xlim(0, 1)
ax2.legend(fontsize=10, frameon=True, edgecolor=GRIDLINE,
           facecolor=SURFACE, framealpha=1)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color(BASELINE)
ax2.spines['bottom'].set_color(BASELINE)
ax2.tick_params(colors=INK_MUTED)
ax2.yaxis.grid(True, color=GRIDLINE, linewidth=0.5)
ax2.set_axisbelow(True)

# ---- Panel 3: Per-task success/failure rate (top 30) ----
ax3 = axes[2]
ax3.set_facecolor(SURFACE)

y_pos = np.arange(len(top30))
bar_height = 0.6

# Stacked horizontal bars: success (left) then failure (right of success)
bars_s = ax3.barh(y_pos, top30['success_rate'].values, height=bar_height,
                  color=CAT_AQUA, edgecolor=SURFACE, linewidth=1,
                  label='Success Rate')
bars_f = ax3.barh(y_pos, top30['failure_rate'].values, height=bar_height,
                  left=top30['success_rate'].values,
                  color=CAT_RED, edgecolor=SURFACE, linewidth=1,
                  label='Failure Rate')

# Truncate long task names for readability
display_names = []
for name in top30['task_name'].values:
    short = str(name).replace('_', ' ')
    if len(short) > 35:
        short = short[:32] + '...'
    display_names.append(short)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(display_names, fontsize=9, color=INK_SECONDARY)
ax3.set_xlabel('Rate', fontsize=12, color=INK_SECONDARY)
ax3.set_title('Per-Task Success / Failure Rate (Top 30 by Success Rate)',
              fontsize=14, fontweight='bold', color=INK_PRIMARY, pad=12)
ax3.set_xlim(0, 1.05)
ax3.legend(fontsize=10, loc='lower right', frameon=True,
           edgecolor=GRIDLINE, facecolor=SURFACE, framealpha=1)

# Selective direct labels: success rate value at bar end (only if > 0)
for i, (sr, fr) in enumerate(
        zip(top30['success_rate'].values, top30['failure_rate'].values)):
    total_rate = sr + fr
    if sr > 0:
        ax3.text(total_rate + 0.01, i,
                 f'{sr:.0%}',
                 ha='left', va='center', fontsize=8, color=INK_SECONDARY)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color(BASELINE)
ax3.spines['bottom'].set_color(BASELINE)
ax3.tick_params(colors=INK_MUTED)
ax3.xaxis.grid(True, color=GRIDLINE, linewidth=0.5)
ax3.set_axisbelow(True)

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
plt.tight_layout(h_pad=3.0)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'reward_analysis.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=SURFACE)
plt.close()
print(f'Saved: {out_path}')
