"""Generate Check 3: Visual examples of frame quality detection types."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def draw_detection_examples():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Check 3: Video Quality Detection Examples',
                 fontsize=16, fontweight='bold', y=0.98, fontfamily='serif')

    # --- Panel A: Black / Corrupt / Blur Detection ---
    ax = axes[0, 0]
    ax.set_title('(A) Frame Quality Scoring', fontsize=12, fontweight='bold',
                 pad=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis('off')

    np.random.seed(42)
    n_frames = 20
    frame_w, frame_h = 3.5, 2.0
    cols = 5
    rows = 4

    frame_types = (
        ['black'] * 2
        + ['normal'] * 5
        + ['blur'] * 2
        + ['normal'] * 7
        + ['blur'] * 1
        + ['normal'] * 2
        + ['black'] * 1
    )

    type_colors = {
        'normal': '#C8E6C9',
        'black': '#212121',
        'blur': '#FFCC80',
        'corrupt': '#EF9A9A',
    }
    type_labels = {
        'normal': 'OK',
        'black': 'BLACK',
        'blur': 'BLUR',
        'corrupt': 'CORRUPT',
    }

    for i in range(n_frames):
        col = i % cols
        row = i // cols
        x = col * (frame_w + 0.8) + 2
        y = 8.5 - row * (frame_h + 0.5)
        ft = frame_types[i]
        color = type_colors[ft]
        text_color = 'white' if ft == 'black' else '#333'

        rect = FancyBboxPatch((x, y), frame_w, frame_h,
                              boxstyle="round,pad=0.1",
                              facecolor=color,
                              edgecolor='#C62828' if ft != 'normal' else '#4CAF50',
                              linewidth=2 if ft != 'normal' else 1)
        ax.add_patch(rect)
        ax.text(x + frame_w / 2, y + frame_h * 0.65,
                f'Frame {i}', fontsize=7, ha='center', color=text_color)
        if ft != 'normal':
            ax.text(x + frame_w / 2, y + frame_h * 0.3,
                    type_labels[ft], fontsize=8, ha='center',
                    fontweight='bold', color='#C62828')
        else:
            ax.text(x + frame_w / 2, y + frame_h * 0.3,
                    'OK', fontsize=7, ha='center', color='#2E7D32')

    # --- Panel B: Static Segment Detection ---
    ax = axes[0, 1]
    ax.set_title('(B) Static Segment Detection (Joint Evidence)',
                 fontsize=12, fontweight='bold', pad=10)

    T = 200
    t = np.arange(T)

    ssim_values = np.ones(T) * 0.99
    ssim_values[30:170] = 0.85 + 0.1 * np.random.rand(140)
    ssim_values[30:170] -= 0.05 * np.sin(np.linspace(0, 4 * np.pi, 140))
    ssim_values = np.clip(ssim_values, 0.7, 1.0)

    state_std = np.zeros(T)
    state_std[30:170] = 0.01 + 0.02 * np.abs(np.sin(np.linspace(0, 6 * np.pi, 140)))
    state_std[:30] = 0.0002 + 0.0001 * np.random.rand(30)
    state_std[170:] = 0.0003 + 0.0001 * np.random.rand(30)

    ax2 = ax.twinx()

    ax.plot(t, ssim_values, color='#1565C0', linewidth=1.5, alpha=0.8,
            label='SSIM (adjacent frames)')
    ax.axhline(y=0.98, color='#1565C0', linestyle='--', linewidth=1,
               alpha=0.5, label='SSIM threshold = 0.98')
    ax.set_ylabel('SSIM', color='#1565C0', fontsize=10)
    ax.set_ylim(0.65, 1.02)
    ax.tick_params(axis='y', labelcolor='#1565C0')

    ax2.plot(t, state_std, color='#E65100', linewidth=1.5, alpha=0.8,
             label='State std (sliding window)')
    ax2.axhline(y=0.001, color='#E65100', linestyle='--', linewidth=1,
                alpha=0.5, label='State threshold = 0.001')
    ax2.set_ylabel('State Std Dev', color='#E65100', fontsize=10)
    ax2.set_ylim(-0.005, 0.04)
    ax2.tick_params(axis='y', labelcolor='#E65100')

    ax.axvspan(0, 30, alpha=0.15, color='red', label='Static segment (start)')
    ax.axvspan(170, 200, alpha=0.15, color='red', label='Static segment (end)')
    ax.axvspan(30, 170, alpha=0.05, color='green')

    ax.set_xlabel('Frame Index', fontsize=10)
    ax.set_xlim(0, T)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower center',
              fontsize=7, ncol=2, framealpha=0.9)

    ax.text(15, 0.68, 'REMOVE\n(start)', fontsize=9, ha='center',
            fontweight='bold', color='#C62828')
    ax.text(185, 0.68, 'REMOVE\n(end)', fontsize=9, ha='center',
            fontweight='bold', color='#C62828')
    ax.text(100, 0.68, 'KEEP (active manipulation)', fontsize=9,
            ha='center', color='#2E7D32')

    # --- Panel C: Keyframe Protection ---
    ax = axes[1, 0]
    ax.set_title('(C) Keyframe Protection (Gripper Closure)',
                 fontsize=12, fontweight='bold', pad=10)

    T = 200
    t = np.arange(T)

    gripper = np.ones(T) * 80
    gripper[40:60] = np.linspace(80, 10, 20)
    gripper[60:140] = 10
    gripper[140:160] = np.linspace(10, 80, 20)
    gripper += np.random.randn(T) * 1.5

    gripper_delta = np.abs(np.diff(gripper))
    gripper_delta = np.append(gripper_delta, 0)

    ax.plot(t, gripper, color='#1565C0', linewidth=1.5, label='Gripper state')
    ax.set_ylabel('Gripper Value', color='#1565C0', fontsize=10)
    ax.tick_params(axis='y', labelcolor='#1565C0')
    ax.set_ylim(0, 100)

    ax3 = ax.twinx()
    ax3.bar(t, gripper_delta, width=1.0, alpha=0.3, color='#E65100',
            label='|Gripper delta|')
    ax3.axhline(y=5.0, color='#E65100', linestyle='--', linewidth=1,
                alpha=0.5, label='Delta threshold = 5.0')
    ax3.set_ylabel('|Delta|', color='#E65100', fontsize=10)
    ax3.tick_params(axis='y', labelcolor='#E65100')

    close_start, close_end = 40, 60
    open_start, open_end = 140, 160
    protect_w = 3

    for (cs, ce, label) in [(close_start, close_end, 'CLOSE'),
                             (open_start, open_end, 'OPEN')]:
        ax.axvspan(cs - protect_w, ce + protect_w, alpha=0.15, color='#4CAF50')
        ax.text((cs + ce) / 2, 95, f'Protected\n({label})',
                fontsize=8, ha='center', va='top', fontweight='bold',
                color='#2E7D32')

    ax.set_xlabel('Frame Index', fontsize=10)
    ax.set_xlim(0, T)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
              fontsize=7, framealpha=0.9)

    # --- Panel D: Removal Safety Check ---
    ax = axes[1, 1]
    ax.set_title('(D) Removal Safety: Action Discontinuity Check',
                 fontsize=12, fontweight='bold', pad=10)

    T_orig = 100
    t_orig = np.arange(T_orig)
    actions_orig = np.cumsum(np.random.randn(T_orig, 2) * 0.3, axis=0)

    keep_mask = np.ones(T_orig, dtype=bool)
    keep_mask[5:15] = False
    keep_mask[80:95] = False

    kept_idx = np.where(keep_mask)[0]
    actions_kept = actions_orig[kept_idx]

    ax.plot(t_orig, actions_orig[:, 0], 'b-', alpha=0.3, linewidth=1,
            label='Original action dim 0')
    ax.plot(kept_idx, actions_kept[:, 0], 'b-', linewidth=2,
            label='After removal dim 0')

    boundaries = np.where(np.diff(kept_idx) > 1)[0]
    for b in boundaries:
        i_before = kept_idx[b]
        i_after = kept_idx[b + 1]
        jump = np.linalg.norm(actions_orig[i_after] - actions_orig[i_before])

        ax.axvspan(i_before, i_after, alpha=0.15, color='red')
        ax.annotate(
            f'Jump = {jump:.2f}',
            xy=((i_before + i_after) / 2, actions_orig[i_after, 0]),
            xytext=((i_before + i_after) / 2, actions_orig[i_after, 0] + 2),
            fontsize=8, ha='center', color='#C62828', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.2))

    ax.set_xlabel('Frame Index', fontsize=10)
    ax.set_ylabel('Action Value', fontsize=10)
    ax.legend(loc='lower left', fontsize=7, framealpha=0.9)

    for rm_start, rm_end in [(5, 15), (80, 95)]:
        ax.text((rm_start + rm_end) / 2,
                ax.get_ylim()[0] + 0.5,
                'REMOVED', fontsize=7, ha='center',
                color='#C62828', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFCDD2',
                          edgecolor='#C62828', linewidth=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = 'b/d/QwenRobotmanip/asset/video_quality_detection_examples.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    draw_detection_examples()
