#!/mnt/r/VENV/dj/bin/python
# -*- coding: utf-8 -*-
"""Data quality analysis for BEHAVIOR-1K 2026 Challenge Demos dataset.

Implements simplified versions of data-juicer _au operator analyses:
- Sudden change detection (jerk analysis)
- Extreme value detection
- Static segment detection (state variance)

Produces: data_quality_analysis.png
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter

DEMOS_DIR = '/mnt/g/DATA/b1k_26/datasets/demos'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Action space definition (from openpi R1Pro config) ----
ACTION_GROUPS = {
    'base': (0, 3),
    'torso': (3, 7),
    'left_arm': (7, 14),
    'left_gripper': (14, 15),
    'right_arm': (15, 22),
    'right_gripper': (22, 23),
}

# ---- Load episode metadata ----
def load_episode_meta():
    ep_dir = os.path.join(DEMOS_DIR, 'meta/episodes')
    all_eps = []
    for chunk in sorted(os.listdir(ep_dir)):
        for f in sorted(glob.glob(os.path.join(ep_dir, chunk, '*.parquet'))):
            all_eps.append(pd.read_parquet(f))
    eps = pd.concat(all_eps, ignore_index=True)
    eps['task_name'] = eps['tasks'].apply(
        lambda x: str(x[0]) if isinstance(x, list) else str(x))
    return eps


# ---- Sample episodes: 2 per task, first 10 tasks ----
def sample_episodes(eps_meta, n_tasks=10, n_per_task=2):
    tasks = sorted(eps_meta['task_name'].unique())[:n_tasks]
    sampled = []
    for t in tasks:
        task_eps = eps_meta[eps_meta['task_name'] == t].head(n_per_task)
        sampled.append(task_eps)
    return pd.concat(sampled, ignore_index=True)


def load_episode_data(episode_row, data_dir):
    """Load a single episode's data using metadata-guided direct lookup."""
    chunk_idx = episode_row['data/chunk_index']
    file_idx = episode_row['data/file_index']
    pq_path = os.path.join(data_dir, f'chunk-{chunk_idx:03d}', f'file-{file_idx:03d}.parquet')
    if not os.path.exists(pq_path):
        return None
    df = pd.read_parquet(pq_path, columns=['episode_index', 'action', 'observation.state'])
    ep_data = df[df['episode_index'] == episode_row['episode_index']]
    return ep_data if len(ep_data) > 0 else None


# ---- Analysis 1: Sudden Change / Jerk Detection ----
def analyze_sudden_changes(actions, median_window=5, savgol_window=11, mad_scale=6.0):
    """Detect sudden changes using median + Savitzky-Golay smoothing + jerk."""
    T, D = actions.shape
    if T < savgol_window + 2:
        return {'flagged_ratio': 0.0, 'max_run': 0, 'n_flagged': 0}

    smoothed = np.copy(actions)
    for d in range(D):
        smoothed[:, d] = medfilt(smoothed[:, d], kernel_size=min(median_window, T) | 1)
    if T >= savgol_window:
        for d in range(D):
            smoothed[:, d] = savgol_filter(smoothed[:, d], savgol_window, 3)

    residual = np.abs(actions - smoothed)
    acc = np.abs(np.diff(actions, n=2, axis=0))
    jerk = np.abs(np.diff(actions, n=3, axis=0))

    # MAD thresholds per dimension
    def mad_threshold(x, scale):
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0) * 1.4826
        return med + scale * mad

    res_thresh = mad_threshold(residual, mad_scale)
    acc_thresh = mad_threshold(acc, mad_scale)
    jerk_thresh = mad_threshold(jerk, mad_scale)

    min_len = min(T, len(acc), len(jerk))
    flagged = np.zeros(min_len, dtype=bool)
    for i in range(min_len):
        res_flag = np.any(residual[i, :] > res_thresh)
        acc_flag = np.any(acc[min(i, len(acc)-1), :] > acc_thresh)
        jerk_flag = np.any(jerk[min(i, len(jerk)-1), :] > jerk_thresh)
        flagged[i] = res_flag and (acc_flag or jerk_flag)

    # Max consecutive run
    max_run = 0
    cur_run = 0
    for f in flagged:
        if f:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    return {
        'flagged_ratio': float(flagged.sum()) / max(1, len(flagged)),
        'max_run': int(max_run),
        'n_flagged': int(flagged.sum()),
        'n_frames': int(min_len),
    }


# ---- Analysis 2: Extreme Value Detection ----
def analyze_extreme_values(actions, states, percentile=99.9):
    """Check for extreme values using IQR method."""
    results = {}
    for name, arr in [('action', actions), ('state', states)]:
        q1 = np.percentile(arr, 1, axis=0)
        q99 = np.percentile(arr, 99, axis=0)
        iqr = q99 - q1
        lower = q1 - 3.0 * iqr
        upper = q99 + 3.0 * iqr
        extreme_mask = np.any((arr < lower) | (arr > upper), axis=1)
        results[name] = {
            'n_extreme_frames': int(extreme_mask.sum()),
            'extreme_ratio': float(extreme_mask.sum()) / max(1, len(arr)),
            'n_dims_with_extremes': int(np.sum(
                np.any((arr < lower) | (arr > upper), axis=0)
            )),
        }
    return results


# ---- Analysis 3: Static Segment Detection ----
def analyze_static_segments(states, window=30, threshold=0.001, min_run=30):
    """Detect static segments based on state variance in sliding windows."""
    T, D = states.shape
    if T < window:
        return {'static_ratio': 0.0, 'n_segments': 0, 'avg_segment_len': 0}

    # Rolling std for each dimension
    is_static = np.zeros(T, dtype=bool)
    for i in range(T - window + 1):
        win_std = np.std(states[i:i+window, :], axis=0)
        if np.mean(win_std) < threshold:
            is_static[i:i+window] = True

    # Find contiguous static runs
    segments = []
    start = None
    for i in range(T):
        if is_static[i] and start is None:
            start = i
        elif not is_static[i] and start is not None:
            if i - start >= min_run:
                segments.append((start, i))
            start = None
    if start is not None and T - start >= min_run:
        segments.append((start, T))

    total_static = sum(e - s for s, e in segments)
    return {
        'static_ratio': float(total_static) / max(1, T),
        'n_segments': len(segments),
        'avg_segment_len': float(total_static) / max(1, len(segments)),
        'total_static_frames': total_static,
    }


def main():
    print("Loading episode metadata...")
    eps_meta = load_episode_meta()
    sampled = sample_episodes(eps_meta, n_tasks=20, n_per_task=1)
    data_dir = os.path.join(DEMOS_DIR, 'data')

    sudden_results = []
    extreme_results = []
    static_results = []
    task_names = []

    print(f"Analyzing {len(sampled)} episodes...")
    for idx, row in sampled.iterrows():
        ep_idx = row['episode_index']
        task = row['task_name']
        print(f"  Episode {ep_idx} ({task})...", end=' ')

        ep_data = load_episode_data(row, data_dir)
        if ep_data is None or len(ep_data) < 20:
            print("SKIP (not found or too short)")
            continue

        actions = np.stack(ep_data['action'].values)
        states = np.stack(ep_data['observation.state'].values)

        sc = analyze_sudden_changes(actions)
        ev = analyze_extreme_values(actions, states)
        ss = analyze_static_segments(states)

        sudden_results.append(sc)
        extreme_results.append(ev)
        static_results.append(ss)
        clean_name = task.strip("[]'\"").replace('_', ' ')
        task_names.append(clean_name)
        print(f"OK (frames={sc['n_frames']}, flagged={sc['flagged_ratio']:.3f}, "
              f"static={ss['static_ratio']:.3f})")

    if not task_names:
        print("No data analyzed!")
        return

    # ---- Plot results ----
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle('Data Quality Analysis (data-juicer Operator Metrics)', fontsize=16, fontweight='bold')

    # Panel 1: Sudden change flagged ratio by task
    ax1 = axes[0]
    x = range(len(task_names))
    ratios = [r['flagged_ratio'] * 100 for r in sudden_results]
    colors1 = ['#e74c3c' if r > 5 else '#f39c12' if r > 1 else '#2ecc71' for r in ratios]
    ax1.bar(x, ratios, color=colors1, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Flagged Frame Ratio (%)')
    ax1.set_title('Sudden Change Detection (Median+SavGol Smoothing + Jerk Analysis)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_names, rotation=45, ha='right', fontsize=7)
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='High threshold (5%)')
    ax1.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Medium threshold (1%)')
    ax1.legend(fontsize=8)

    # Panel 2: Extreme value counts
    ax2 = axes[1]
    action_extremes = [r['action']['extreme_ratio'] * 100 for r in extreme_results]
    state_extremes = [r['state']['extreme_ratio'] * 100 for r in extreme_results]
    width = 0.35
    ax2.bar([i - width/2 for i in x], action_extremes, width, label='Action', color='#3498db')
    ax2.bar([i + width/2 for i in x], state_extremes, width, label='State', color='#9b59b6')
    ax2.set_ylabel('Extreme Value Frame Ratio (%)')
    ax2.set_title('Extreme Value Detection (IQR Method, 3x IQR beyond Q1/Q99)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_names, rotation=45, ha='right', fontsize=7)
    ax2.legend(fontsize=8)

    # Panel 3: Static segment ratio
    ax3 = axes[2]
    static_ratios = [r['static_ratio'] * 100 for r in static_results]
    colors3 = ['#e74c3c' if r > 30 else '#f39c12' if r > 10 else '#2ecc71' for r in static_ratios]
    ax3.bar(x, static_ratios, color=colors3, edgecolor='white', linewidth=0.5)
    ax3.set_ylabel('Static Segment Ratio (%)')
    ax3.set_title('Static Segment Detection (Rolling State Std < 0.001, Window=30 frames)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(task_names, rotation=45, ha='right', fontsize=7)
    ax3.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='High threshold (30%)')
    ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Medium threshold (10%)')
    ax3.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'data_quality_analysis.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Save JSON results for embedding in document
    results = {
        'tasks': task_names,
        'sudden_change': sudden_results,
        'extreme_values': extreme_results,
        'static_segments': static_results,
    }
    json_path = os.path.join(OUT_DIR, 'data_quality_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    avg_flagged = np.mean([r['flagged_ratio'] for r in sudden_results]) * 100
    avg_action_extreme = np.mean([r['action']['extreme_ratio'] for r in extreme_results]) * 100
    avg_state_extreme = np.mean([r['state']['extreme_ratio'] for r in extreme_results]) * 100
    avg_static = np.mean([r['static_ratio'] for r in static_results]) * 100
    print(f"Avg sudden change flagged ratio: {avg_flagged:.2f}%")
    print(f"Avg action extreme value ratio:  {avg_action_extreme:.2f}%")
    print(f"Avg state extreme value ratio:   {avg_state_extreme:.2f}%")
    print(f"Avg static segment ratio:        {avg_static:.2f}%")


if __name__ == '__main__':
    main()
