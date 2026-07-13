"""
BEHAVIOR-1K vs 主要具身AI基准的多维度对比雷达图
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['DejaVu Sans']

categories = [
    'Activities',
    'Scene Types',
    'Object\nCategories',
    'Simulation\nFeatures',
    'Visual\nQuality',
    'Objects per\nActivity',
]

def normalize(vals, maxvals):
    return [v / m for v, m in zip(vals, maxvals)]

maxvals = [1000, 8, 1949, 7, 5.0, 47]

benchmarks = {
    'BEHAVIOR-1K':   [1000, 8, 1949, 7, 3.20, 47],
    'BEHAVIOR-100':  [100,  1, 391,  2, 1.69, 34],
    'VirtualHome':   [549,  1, 308,  0, 0,    24],
    'Habitat 2.0':   [3,    1, 41,   2, 1.74, 5],
    'AI2-THOR':      [1,    1, 118,  1, 1.73, 5],
    'RFUniverse':    [5,    1, 0,    4, 0,    6],
}

colors = {
    'BEHAVIOR-1K':  '#E63946',
    'BEHAVIOR-100': '#457B9D',
    'VirtualHome':  '#2A9D8F',
    'Habitat 2.0':  '#E9C46A',
    'AI2-THOR':     '#F4A261',
    'RFUniverse':   '#264653',
}

N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for name, vals in benchmarks.items():
    normed = normalize(vals, maxvals)
    normed += normed[:1]
    ax.plot(angles, normed, 'o-', linewidth=2.5 if name == 'BEHAVIOR-1K' else 1.5,
            label=name, color=colors[name],
            markersize=8 if name == 'BEHAVIOR-1K' else 5,
            zorder=10 if name == 'BEHAVIOR-1K' else 1)
    ax.fill(angles, normed, alpha=0.15 if name == 'BEHAVIOR-1K' else 0.05,
            color=colors[name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9, color='gray')
ax.spines['polar'].set_visible(False)

ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11, framealpha=0.9)
ax.set_title('BEHAVIOR-1K vs Major Embodied AI Benchmarks\n(Normalized to BEHAVIOR-1K Maximum)',
             fontsize=14, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig('b/d/b1k_26/asset/benchmark_comparison_radar.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: benchmark_comparison_radar.png")
