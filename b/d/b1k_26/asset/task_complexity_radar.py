#!/mnt/r/VENV/dj/bin/python
"""
BEHAVIOR-1K Task Complexity Radar
==================================
Spider/radar chart comparing 10 representative tasks (3 shortest,
4 middle, 3 longest by avg episode length) across 5 complexity axes.

Generates: task_complexity_radar.png
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
# 8 categorical slots + composite encoding (dashed lines) for slots 9-10
# ---------------------------------------------------------------------------
CAT_COLORS = [
    '#2a78d6',  # 1 blue
    '#1baf7a',  # 2 aqua
    '#eda100',  # 3 yellow
    '#008300',  # 4 green
    '#4a3aa7',  # 5 violet
    '#e34948',  # 6 red
    '#e87ba4',  # 7 magenta
    '#eb6834',  # 8 orange
]
# Slots 9-10: reuse slots 1-2 with dashed lines (composite encoding)
CAT_COLORS_EXT = CAT_COLORS + [CAT_COLORS[0], CAT_COLORS[1]]
LINE_STYLES = ['solid'] * 8 + ['dashed', 'dashed']

INK_PRIMARY   = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED     = '#898781'
GRIDLINE      = '#e1e0d9'
BASELINE      = '#c3c2b7'
SURFACE       = '#fcfcfb'

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
eps['task_name'] = eps['tasks'].apply(lambda x: str(x[0]) if isinstance(x, (list, np.ndarray)) else str(x))
print(f'  Episodes: {len(eps)}')

# Task name lookup
tasks_df = pd.read_parquet('/mnt/g/DATA/b1k_26/datasets/demos/meta/tasks.parquet')
task_idx_to_name = dict(zip(tasks_df['task_index'], tasks_df.index))

# ---------------------------------------------------------------------------
# 2. Per-task episode statistics
# ---------------------------------------------------------------------------
task_ep_stats = (
    eps.groupby('task_name')
    .agg(
        avg_length=('length', 'mean'),
        std_length=('length', 'std'),
        ep_count=('episode_index', 'count'),
    )
    .reset_index()
)
task_ep_stats['std_length'] = task_ep_stats['std_length'].fillna(0)

# ---------------------------------------------------------------------------
# 3. Select 10 tasks: 3 shortest, 3 longest, 4 middle
# ---------------------------------------------------------------------------
sorted_tasks = task_ep_stats.sort_values('avg_length').reset_index(drop=True)
n = len(sorted_tasks)
shortest_3 = sorted_tasks.head(3)
longest_3 = sorted_tasks.tail(3)
# 4 middle: evenly spaced from the middle region
mid_start = n // 4
mid_end = 3 * n // 4
mid_indices = np.linspace(mid_start, mid_end, 4, dtype=int)
middle_4 = sorted_tasks.iloc[mid_indices]

selected = pd.concat([shortest_3, middle_4, longest_3], ignore_index=True)
selected_names = set(selected['task_name'].values)
print(f'  Selected tasks: {list(selected_names)}')

# ---------------------------------------------------------------------------
# 4. Load reward data for success rate computation
# ---------------------------------------------------------------------------
print('Loading reward data for success rates ...')
data_dir = '/mnt/g/DATA/b1k_26/datasets/demos/data'
reward_cols = ['next.reward', 'episode_index', 'task_index']
reward_parts = []
for chunk in sorted(os.listdir(data_dir)):
    fp = os.path.join(data_dir, chunk, 'file-000.parquet')
    if os.path.exists(fp):
        reward_parts.append(pd.read_parquet(fp, columns=reward_cols))
reward_data = pd.concat(reward_parts, ignore_index=True)

# Map task_index -> task_name
reward_data['task_name'] = reward_data['task_index'].map(task_idx_to_name)

# Per-task success rate: fraction of episodes with at least one +1 reward
nonzero_rewards = reward_data[reward_data['next.reward'] != 0]
ep_outcomes = (
    nonzero_rewards
    .groupby(['task_name', 'episode_index'])['next.reward']
    .max()
    .reset_index()
)
ep_outcomes['success'] = (ep_outcomes['next.reward'] > 0).astype(int)
task_success = (
    ep_outcomes
    .groupby('task_name')['success']
    .mean()
    .reset_index()
    .rename(columns={'success': 'success_rate'})
)

# ---------------------------------------------------------------------------
# 5. Load a sample of chunks for action diversity
# ---------------------------------------------------------------------------
print('Loading action data (sampled chunks) ...')
sample_chunks = [f'chunk-{i:03d}' for i in range(0, 100, 10)]  # 10 chunks
action_parts = []
for chunk in sample_chunks:
    fp = os.path.join(data_dir, chunk, 'file-000.parquet')
    if os.path.exists(fp):
        df = pd.read_parquet(fp, columns=['action', 'task_index'])
        df['task_name'] = df['task_index'].map(task_idx_to_name)
        # Only keep selected tasks to save memory
        df = df[df['task_name'].isin(selected_names)]
        action_parts.append(df)
action_data = pd.concat(action_parts, ignore_index=True)

# Compute action diversity: mean std across action dimensions per task
def compute_action_diversity(group):
    actions = np.array(group['action'].tolist())
    if len(actions) < 2:
        return 0.0
    per_dim_std = np.std(actions, axis=0)
    return float(np.mean(per_dim_std))

task_action_div = (
    action_data
    .groupby('task_name')
    .apply(compute_action_diversity, include_groups=False)
    .reset_index()
    .rename(columns={0: 'action_diversity'})
)

# ---------------------------------------------------------------------------
# 6. Merge all metrics for selected tasks
# ---------------------------------------------------------------------------
radar_data = selected.merge(task_success, on='task_name', how='left')
radar_data = radar_data.merge(task_action_div, on='task_name', how='left')
radar_data['success_rate'] = radar_data['success_rate'].fillna(0)
radar_data['action_diversity'] = radar_data['action_diversity'].fillna(0)

# Normalize each axis to [0, 1] across ALL tasks (not just selected)
# to give meaningful relative scale
def normalize(series):
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return series * 0 + 0.5
    return (series - lo) / (hi - lo)

radar_data['norm_duration']       = normalize(radar_data['avg_length'])
radar_data['norm_duration_var']   = normalize(radar_data['std_length'])
radar_data['norm_action_div']     = normalize(radar_data['action_diversity'])
radar_data['norm_success']        = normalize(radar_data['success_rate'])
radar_data['norm_ep_count']       = normalize(radar_data['ep_count'])

axes_labels = [
    'Avg Duration',
    'Duration Variance',
    'Action Diversity',
    'Success Rate',
    'Episode Count',
]
metric_cols = [
    'norm_duration',
    'norm_duration_var',
    'norm_action_div',
    'norm_success',
    'norm_ep_count',
]

# ---------------------------------------------------------------------------
# 7. Radar plot
# ---------------------------------------------------------------------------
print('Plotting radar chart ...')
num_axes = len(axes_labels)
angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# Sort selected tasks by avg_length for legend clarity
radar_data = radar_data.sort_values('avg_length').reset_index(drop=True)

# Group labels for the legend
group_labels = (
    ['short'] * 3 + ['mid'] * 4 + ['long'] * 3
)

for i, row in radar_data.iterrows():
    values = [row[c] for c in metric_cols]
    values += values[:1]  # close polygon

    task_label = str(row['task_name']).replace('_', ' ')
    if len(task_label) > 30:
        task_label = task_label[:27] + '...'

    color = CAT_COLORS_EXT[i]
    ls = LINE_STYLES[i]
    suffix = ' *' if ls == 'dashed' else ''

    ax.plot(angles, values, linewidth=2, linestyle=ls,
            color=color, label=f'{task_label}{suffix}')
    ax.fill(angles, values, alpha=0.08, color=color)

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(axes_labels, fontsize=12, color=INK_PRIMARY, fontweight='bold')

# Radial grid
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                    fontsize=9, color=INK_MUTED)
ax.set_ylim(0, 1.05)

# Style the grid and spines
ax.spines['polar'].set_color(BASELINE)
ax.spines['polar'].set_linewidth(0.5)
ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.5)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.5)

ax.set_title(
    'Task Complexity Profiles\n'
    '(3 shortest, 4 middle, 3 longest tasks by avg episode length)',
    fontsize=15, fontweight='bold', color=INK_PRIMARY,
    pad=30,
)

# Legend: outside bottom, two columns
# Note about dashed entries
legend = ax.legend(
    loc='upper left',
    bbox_to_anchor=(-0.15, -0.05),
    ncol=2,
    fontsize=9,
    frameon=True,
    edgecolor=GRIDLINE,
    facecolor=SURFACE,
    framealpha=1,
    title='Tasks (* = composite encoding, dashed line)',
    title_fontsize=9,
)
legend.get_title().set_color(INK_SECONDARY)

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'task_complexity_radar.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=SURFACE)
plt.close()
print(f'Saved: {out_path}')
