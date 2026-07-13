#!/usr/bin/env python3
"""生成多专家投票策略对比图。

展示三种投票策略（majority / weighted / unanimous_override）
在不同专家意见分布下的决策差异。

输出: instruction_consistency_voting.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor('white')
fig.suptitle('Multi-Expert Cross-Model Adjudication — Voting Strategy Comparison',
             fontsize=14, fontweight='bold', y=0.98)

# ── 场景数据 ──
scenarios = [
    {
        'name': 'Scenario A:\nClear Majority',
        'experts': [
            ('Gemini-2.0',  'consistent',   0.85),
            ('Qwen-VL',     'consistent',   0.72),
            ('GPT-4o',      'inconsistent', 0.60),
        ],
    },
    {
        'name': 'Scenario B:\nSplit with Confidence Gap',
        'experts': [
            ('Gemini-2.0',  'inconsistent', 0.92),
            ('Qwen-VL',     'consistent',   0.55),
            ('GPT-4o',      'consistent',   0.51),
        ],
    },
    {
        'name': 'Scenario C:\nUnanimous Agreement',
        'experts': [
            ('Gemini-2.0',  'consistent',   0.90),
            ('Qwen-VL',     'consistent',   0.88),
            ('GPT-4o',      'consistent',   0.85),
        ],
    },
]

C_CONSISTENT   = '#5CB85C'
C_INCONSISTENT = '#D9534F'
C_AMBIGUOUS    = '#F0AD4E'

strategies = ['Majority Vote', 'Weighted Vote', 'Unanimous Override']

def compute_verdicts(experts):
    results = {}

    # Majority
    c_count = sum(1 for _, v, _ in experts if v == 'consistent')
    results['Majority Vote'] = 'consistent' if c_count > len(experts) / 2 else 'inconsistent'

    # Weighted
    w_c = sum(conf for _, v, conf in experts if v == 'consistent')
    w_total = sum(conf for _, _, conf in experts)
    results['Weighted Vote'] = 'consistent' if w_c / w_total > 0.5 else 'inconsistent'

    # Unanimous override
    verdicts = set(v for _, v, _ in experts)
    if len(verdicts) == 1:
        results['Unanimous Override'] = verdicts.pop()
    else:
        results['Unanimous Override'] = 'ambiguous'

    return results

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    # 场景标题
    ax.text(1.5, 5.2, scenario['name'],
            ha='center', va='center', fontsize=11, fontweight='bold', color='#222')

    # 专家投票展示
    for i, (name, verdict, conf) in enumerate(scenario['experts']):
        color = C_CONSISTENT if verdict == 'consistent' else C_INCONSISTENT
        box = FancyBboxPatch((0.1, 3.8 - i * 0.75), 2.8, 0.6,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='#555',
                             linewidth=1, alpha=0.75)
        ax.add_patch(box)

        label = f'{name}: {verdict} ({conf:.2f})'
        ax.text(1.5, 4.1 - i * 0.75, label,
                ha='center', va='center', fontsize=8.5,
                fontweight='bold', color='white')

    # 投票结果
    verdicts = compute_verdicts(scenario['experts'])
    y_base = 1.2
    for j, strategy in enumerate(strategies):
        v = verdicts[strategy]
        if v == 'consistent':
            color, symbol = C_CONSISTENT, '[OK]'
        elif v == 'inconsistent':
            color, symbol = C_INCONSISTENT, '[X]'
        else:
            color, symbol = C_AMBIGUOUS, '[?]'

        box = FancyBboxPatch((0.1, y_base - j * 0.55), 2.8, 0.45,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor='#555',
                             linewidth=1, alpha=0.6)
        ax.add_patch(box)

        ax.text(1.5, y_base + 0.22 - j * 0.55,
                f'{symbol} {strategy}: {v}',
                ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

    # 分隔线
    ax.plot([0.3, 2.7], [1.55, 1.55], color='#999', linewidth=1, linestyle='--')
    ax.text(1.5, 1.65, '↓ Strategy Results ↓',
            ha='center', va='bottom', fontsize=7, color='#888', style='italic')

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(os.path.dirname(__file__), 'instruction_consistency_voting.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')
plt.close()
