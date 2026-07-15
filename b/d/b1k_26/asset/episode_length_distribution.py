#!/mnt/r/VENV/dj/bin/python
"""
BEHAVIOR-1K Episode Length Distribution Analysis

Loads episode metadata from parquet chunks, produces a 3-panel figure:
  1. Global episode length histogram with mean/median markers
  2. Per-task boxplot for 20 representative tasks
  3. Top/bottom 5 tasks bar chart by mean episode length
"""
import os
import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

matplotlib.rcParams['font.family'] = ['DejaVu Sans']
plt.style.use('bmh')

FPS = 30
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'episode_length_distribution.png')

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
ep_dir = '/mnt/g/DATA/b1k_26/datasets/demos/meta/episodes'
all_eps = []
for chunk in sorted(os.listdir(ep_dir)):
    chunk_path = os.path.join(ep_dir, chunk)
    if not os.path.isdir(chunk_path):
        continue
    for f in sorted(glob.glob(os.path.join(chunk_path, '*.parquet'))):
        all_eps.append(pd.read_parquet(f, columns=['episode_index', 'tasks',
                                                    'length']))
eps = pd.concat(all_eps, ignore_index=True)
eps['task_name'] = eps['tasks'].apply(
    lambda x: str(x[0]).replace('_', ' ') if hasattr(x, '__len__') else
    str(x).replace('_', ' ')
)

print(f'Loaded {len(eps)} episodes across {eps["task_name"].nunique()} tasks')

# Per-task statistics
task_stats = (eps.groupby('task_name')['length']
              .agg(['mean', 'median', 'std', 'count'])
              .sort_values('mean'))

# ---------------------------------------------------------------------------
# 2. Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                         gridspec_kw={'height_ratios': [1, 1.3, 0.8]})

# ---- Panel 1: Global histogram ----
ax1 = axes[0]
lengths = eps['length'].values
mean_len = np.mean(lengths)
median_len = np.median(lengths)

ax1.hist(lengths, bins=80, color='#457B9D', edgecolor='white', linewidth=0.5,
         alpha=0.85)
ax1.axvline(mean_len, color='#E63946', linestyle='-', linewidth=2,
            label=f'Mean = {mean_len:.0f} frames ({mean_len / FPS:.1f}s)')
ax1.axvline(median_len, color='#264653', linestyle='--', linewidth=2,
            label=f'Median = {median_len:.0f} frames ({median_len / FPS:.1f}s)')
ax1.legend(fontsize=11, loc='upper right')
ax1.set_xlabel('Episode Length (frames)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Global Episode Length Distribution (20,000 episodes)',
              fontsize=14, fontweight='bold')

# Secondary x-axis: seconds
ax1_sec = ax1.secondary_xaxis('top', functions=(lambda x: x / FPS,
                                                 lambda x: x * FPS))
ax1_sec.set_xlabel('Duration (seconds)', fontsize=11)

# ---- Panel 2: Per-task boxplot (20 representative tasks) ----
ax2 = axes[1]

sorted_tasks = task_stats.index.tolist()  # already sorted by mean
n_tasks = len(sorted_tasks)

# Select 20 representative tasks: shortest 5, longest 5, 10 evenly from middle
shortest_5 = sorted_tasks[:5]
longest_5 = sorted_tasks[-5:]
middle_range = sorted_tasks[5:-5]
if len(middle_range) >= 10:
    indices = np.linspace(0, len(middle_range) - 1, 10, dtype=int)
    middle_10 = [middle_range[i] for i in indices]
else:
    middle_10 = middle_range
selected_tasks = shortest_5 + middle_10 + longest_5

# Re-sort selected tasks by median for the boxplot
selected_medians = task_stats.loc[selected_tasks, 'median'].sort_values()
selected_tasks_sorted = selected_medians.index.tolist()

# Gather data for boxplot
box_data = [eps.loc[eps['task_name'] == t, 'length'].values
            for t in selected_tasks_sorted]

# Color by category
colors_box = []
for t in selected_tasks_sorted:
    if t in shortest_5:
        colors_box.append('#2A9D8F')
    elif t in longest_5:
        colors_box.append('#E76F51')
    else:
        colors_box.append('#457B9D')

bp = ax2.boxplot(box_data, vert=False, patch_artist=True,
                 widths=0.6,
                 boxprops=dict(linewidth=0.8),
                 medianprops=dict(color='#264653', linewidth=1.5),
                 whiskerprops=dict(linewidth=0.8),
                 capprops=dict(linewidth=0.8),
                 flierprops=dict(marker='.', markersize=3, alpha=0.4))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Capitalize first letter of each task name for labels
display_labels = [t.title() for t in selected_tasks_sorted]
ax2.set_yticklabels(display_labels, fontsize=9)
ax2.set_xlabel('Episode Length (frames)', fontsize=12)
ax2.set_title('Per-Task Episode Length Distribution (20 representative tasks)',
              fontsize=14, fontweight='bold')

# Legend for color categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2A9D8F', alpha=0.75, label='Shortest 5 tasks'),
    Patch(facecolor='#457B9D', alpha=0.75, label='Middle (evenly sampled)'),
    Patch(facecolor='#E76F51', alpha=0.75, label='Longest 5 tasks'),
]
ax2.legend(handles=legend_elements, fontsize=9, loc='lower right')

# Secondary x-axis: seconds
ax2_sec = ax2.secondary_xaxis('top', functions=(lambda x: x / FPS,
                                                 lambda x: x * FPS))
ax2_sec.set_xlabel('Duration (seconds)', fontsize=11)

# ---- Panel 3: Top/bottom 5 tasks bar chart ----
ax3 = axes[2]

bottom_5 = task_stats.head(5)
top_5 = task_stats.tail(5).iloc[::-1]  # reverse so longest first

all_10 = pd.concat([top_5, bottom_5])
labels = [t.title() for t in all_10.index]
means = all_10['mean'].values
stds = all_10['std'].values
colors_bar = ['#E76F51'] * 5 + ['#2A9D8F'] * 5

y_pos = np.arange(len(labels))
bars = ax3.barh(y_pos, means, xerr=stds, height=0.6,
                color=colors_bar, edgecolor='white', linewidth=0.8,
                capsize=3, error_kw=dict(linewidth=1, alpha=0.6))

# Annotate bars with mean value
for i, (m, s) in enumerate(zip(means, stds)):
    ax3.text(m + s + 200, i, f'{m:.0f} ({m / FPS:.1f}s)',
             va='center', fontsize=9, color='#333333')

ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels, fontsize=9)
ax3.set_xlabel('Mean Episode Length (frames)', fontsize=12)
ax3.set_title('Longest vs Shortest Tasks by Mean Episode Length',
              fontsize=14, fontweight='bold')
ax3.invert_yaxis()

# Legend
legend_bars = [
    Patch(facecolor='#E76F51', label='5 Longest tasks'),
    Patch(facecolor='#2A9D8F', label='5 Shortest tasks'),
]
ax3.legend(handles=legend_bars, fontsize=9, loc='lower right')

# Secondary x-axis: seconds
ax3_sec = ax3.secondary_xaxis('top', functions=(lambda x: x / FPS,
                                                 lambda x: x * FPS))
ax3_sec.set_xlabel('Duration (seconds)', fontsize=11)

# ---------------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------------
plt.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved to {OUT_PATH}')
plt.close(fig)
