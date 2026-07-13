#!/usr/bin/env python3
"""生成 Instruction Consistency 三阶段流水线架构图。

输出: instruction_consistency_pipeline.png
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── 颜色方案 ──
C_STAGE1 = '#4A90D9'
C_STAGE2 = '#E8833A'
C_STAGE3 = '#7B68EE'
C_INPUT  = '#5CB85C'
C_OUTPUT = '#D9534F'
C_DATA   = '#F5F5DC'
C_ARROW  = '#333333'

def draw_box(x, y, w, h, text, color, fontsize=10, bold=False, alpha=0.85):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='#333',
                         linewidth=1.5, alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color='white' if color not in [C_DATA, '#F5F5DC'] else '#333',
            wrap=True, zorder=3)

def draw_arrow(x1, y1, x2, y2, label='', color=C_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2, connectionstyle='arc3,rad=0'),
                zorder=4)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.15, label, ha='center', va='bottom',
                fontsize=7.5, color='#555', style='italic', zorder=5)

# ── 标题 ──
ax.text(8, 8.5, 'Instruction Consistency Check — Three-Stage VLM Pipeline',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#222')
ax.text(8, 8.1, 'Qwen-RobotManip Data Curation (Check 1)',
        ha='center', va='center', fontsize=10, color='#666')

# ── 输入 ──
draw_box(0.3, 5.5, 2.2, 1.2, 'Input\n\nEpisode Video\n+ Instruction', C_INPUT, 9, True)

# ── Stage 1 ──
draw_box(3.5, 5.5, 3.0, 1.2,
         'Stage 1: Temporal\nNormalization\n\nSubtask Segmentation\n(speed minima)', C_STAGE1, 9, True)

# ── 中间数据 ──
draw_box(3.8, 3.8, 2.4, 0.9,
         'Segments\n[clip₁, clip₂, ..., clipₙ]', C_DATA, 8)

# ── Stage 2 ──
draw_box(7.0, 5.5, 3.0, 1.2,
         'Stage 2: Structured\nReasoning VLM\n\n4-dim analysis -> verdict\n(CoT prompting)', C_STAGE2, 9, True)

# Stage 2 子步骤
draw_box(7.2, 3.8, 2.6, 0.9,
         'Objects / Actions\nTemporal / Interaction\n-> verdict + confidence', C_DATA, 7.5)

# ── 分支判定 ──
# 菱形
diamond_x, diamond_y = 10.8, 6.1
diamond = plt.Polygon(
    [(diamond_x, diamond_y - 0.5),
     (diamond_x + 0.6, diamond_y),
     (diamond_x, diamond_y + 0.5),
     (diamond_x - 0.6, diamond_y)],
    facecolor='#FFF3CD', edgecolor='#333', linewidth=1.5, zorder=2)
ax.add_patch(diamond)
ax.text(diamond_x, diamond_y, 'conf\n≥ θ?',
        ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

# ── Stage 3 ──
draw_box(11.8, 5.5, 3.0, 1.2,
         'Stage 3: Multi-Expert\nAdjudication\n\nN VLMs evaluate\n-> cross-model voting', C_STAGE3, 9, True)

# Stage 3 子模型
y_models = 3.8
for i, name in enumerate(['VLM₁\n(Gemini)', 'VLM₂\n(Qwen)', 'VLM₃\n(GPT-4o)']):
    draw_box(11.5 + i * 1.1, y_models, 1.0, 0.8, name, C_STAGE3, 7, alpha=0.6)

# 投票框
draw_box(12.0, 2.6, 2.3, 0.7,
         'Majority Vote\nscore > 0.5 -> keep', C_DATA, 8)

# ── 输出 ──
draw_box(6.0, 0.8, 2.0, 0.9,
         'Consistent\n-> Keep', C_INPUT, 9, True)
draw_box(9.0, 0.8, 2.0, 0.9,
         'Inconsistent\n-> Discard', C_OUTPUT, 9, True)

# ── 箭头 ──
# Input -> Stage 1
draw_arrow(2.5, 6.1, 3.5, 6.1)

# Stage 1 -> Segments
draw_arrow(5.0, 5.5, 5.0, 4.7, 'split')

# Segments -> Stage 2
draw_arrow(6.2, 4.25, 7.0, 5.5)

# Stage 2 -> 4-dim analysis
draw_arrow(8.5, 5.5, 8.5, 4.7, 'evaluate')

# Stage 2 -> diamond
draw_arrow(10.0, 6.1, 10.2, 6.1)

# diamond -> Keep (high confidence consistent)
draw_arrow(diamond_x, diamond_y - 0.5, 7.0, 1.7, 'Yes (consistent)')

# diamond -> Stage 3 (ambiguous/inconsistent)
draw_arrow(diamond_x + 0.6, diamond_y, 11.8, 6.1, 'No')

# Stage 3 -> sub-models
draw_arrow(13.3, 5.5, 13.3, 4.6)

# sub-models -> vote
draw_arrow(13.1, 3.8, 13.1, 3.3, 'vote')

# vote -> keep/discard
draw_arrow(12.0, 2.9, 8.0, 1.7)
draw_arrow(14.3, 2.9, 11.0, 1.7)

# ── 图例 ──
legend_elements = [
    mpatches.Patch(facecolor=C_STAGE1, edgecolor='#333', label='Stage 1: Temporal Normalization'),
    mpatches.Patch(facecolor=C_STAGE2, edgecolor='#333', label='Stage 2: Structured Reasoning VLM'),
    mpatches.Patch(facecolor=C_STAGE3, edgecolor='#333', label='Stage 3: Multi-Expert Adjudication'),
    mpatches.Patch(facecolor=C_DATA, edgecolor='#333', label='Intermediate Data'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8,
          framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'instruction_consistency_pipeline.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')
plt.close()
