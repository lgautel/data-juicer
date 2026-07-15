"""Generate Check 3: Video Quality Filtering pipeline architecture diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    title_y = 9.5
    ax.text(8, title_y, 'Check 3: Video Quality Filtering Pipeline',
            fontsize=16, fontweight='bold', ha='center', va='center',
            fontfamily='serif')
    ax.text(8, title_y - 0.4, '(Qwen-RobotManip Data Curation)',
            fontsize=11, ha='center', va='center', color='#666666',
            fontfamily='serif')

    colors = {
        'input': '#E8F5E9',
        'mapper': '#E3F2FD',
        'filter': '#FFF3E0',
        'output': '#F3E5F5',
        'arrow': '#455A64',
        'meta': '#ECEFF1',
    }

    def draw_box(x, y, w, h, title, subtitle, color, border_color='#333'):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor=border_color,
                             linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.65, title,
                fontsize=10, fontweight='bold', ha='center', va='center',
                fontfamily='sans-serif')
        ax.text(x + w / 2, y + h * 0.3, subtitle,
                fontsize=7.5, ha='center', va='center', color='#555',
                fontfamily='sans-serif', style='italic')

    def draw_arrow(x1, y1, x2, y2, label=None, color='#455A64'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.8, connectionstyle='arc3,rad=0'))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, fontsize=7, color='#777',
                    ha='left', va='center', fontfamily='monospace')

    def draw_meta_label(x, y, text):
        ax.text(x, y, text, fontsize=6.5, ha='center', va='center',
                color='#1565C0', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD',
                          edgecolor='#90CAF9', linewidth=0.8))

    # Input
    draw_box(0.5, 7.2, 3.5, 1.2,
             'Input Episode',
             'video frames + states + actions',
             colors['input'], '#4CAF50')

    # Step 1
    draw_box(5.5, 7.2, 4.5, 1.2,
             'Step 1: Frame Quality Scorer',
             'black / blur / corrupt detection (Mapper)',
             colors['mapper'], '#1976D2')
    draw_arrow(4.0, 7.8, 5.5, 7.8)
    draw_meta_label(7.75, 6.8, 'meta["frame_quality_report"]')

    # Step 2
    draw_box(11.0, 7.2, 4.5, 1.2,
             'Step 2: Key Frame Detector',
             'gripper closure + velocity peaks (Mapper)',
             colors['mapper'], '#1976D2')
    draw_arrow(10.0, 7.8, 11.0, 7.8)
    draw_meta_label(13.25, 6.8, 'meta["key_frame_report"]')

    # Step 3
    draw_box(3.5, 4.8, 5.5, 1.2,
             'Step 3: Static Segment Detector',
             'SSIM + state stdev, edge-biased, mask merge (Mapper)',
             colors['mapper'], '#1976D2')
    draw_arrow(7.75, 6.8, 6.25, 6.0, color='#1565C0')
    draw_arrow(13.25, 6.8, 6.75, 6.0, color='#1565C0')
    draw_meta_label(6.25, 4.35, 'meta["video_quality_removal_mask"]')

    # Step 4
    draw_box(3.5, 2.8, 5.5, 1.2,
             'Step 4: Frame Removal Safety',
             'action discontinuity check (Filter)',
             colors['filter'], '#E65100')
    draw_arrow(6.25, 4.8, 6.25, 4.0)
    draw_meta_label(10.5, 3.4, 'stats["frame_removal_safe"]')
    draw_arrow(9.0, 3.4, 10.0, 3.4, color='#E65100')

    # Decision diamond
    diamond_x, diamond_y = 6.25, 1.8
    diamond = plt.Polygon(
        [(diamond_x, diamond_y + 0.45),
         (diamond_x + 0.8, diamond_y),
         (diamond_x, diamond_y - 0.45),
         (diamond_x - 0.8, diamond_y)],
        closed=True, facecolor='#FFF9C4', edgecolor='#F9A825',
        linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, 'Safe?', fontsize=9, fontweight='bold',
            ha='center', va='center')
    draw_arrow(6.25, 2.8, 6.25, 2.25, color='#E65100')

    # Step 5 (safe=True)
    draw_box(9.0, 1.0, 5.5, 1.2,
             'Step 5: Frame Removal',
             'trim frames/states/actions/timestamps (Mapper)',
             colors['mapper'], '#1976D2')
    ax.annotate('', xy=(9.0, 1.6), xytext=(7.05, 1.8),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.8))
    ax.text(8.0, 2.0, 'Yes', fontsize=9, color='#2E7D32',
            fontweight='bold', ha='center')

    # Unsafe path
    draw_box(0.5, 1.0, 3.5, 1.2,
             'Keep Original',
             'mark warning, no modification',
             '#FFCDD2', '#C62828')
    ax.annotate('', xy=(3.0, 1.6), xytext=(5.45, 1.8),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.8))
    ax.text(4.0, 2.0, 'No', fontsize=9, color='#C62828',
            fontweight='bold', ha='center')

    # Output
    draw_box(10.5, -0.2, 3.5, 0.8,
             'Output: Cleaned Episode',
             '',
             colors['output'], '#7B1FA2')
    draw_arrow(11.75, 1.0, 12.25, 0.6)

    # Legend
    legend_y = 0.3
    legend_items = [
        (colors['mapper'], 'Mapper (annotate / transform)'),
        (colors['filter'], 'Filter (keep / reject)'),
        ('#E3F2FD', 'Fields.meta communication'),
    ]
    for i, (c, label) in enumerate(legend_items):
        rx = 0.5 + i * 3.5
        rect = FancyBboxPatch((rx, legend_y - 0.15), 0.3, 0.3,
                              boxstyle="round,pad=0.05",
                              facecolor=c, edgecolor='#999', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(rx + 0.45, legend_y, label, fontsize=7.5, va='center',
                fontfamily='sans-serif')

    plt.tight_layout()
    out_path = 'b/d/QwenRobotmanip/asset/video_quality_pipeline.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    draw_pipeline()
