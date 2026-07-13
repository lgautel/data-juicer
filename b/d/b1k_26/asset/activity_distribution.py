"""
BEHAVIOR-1K 活动类别分布与场景类型统计
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['DejaVu Sans']

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Left: Scene Types Distribution ---
ax1 = axes[0]
scene_types = ['Houses', 'Houses+\nGarden', 'New\nHouses', 'Grocery\nStores',
               'Halls', 'Hotels', 'Offices', 'Restaurants', 'Schools']
scene_counts = [15, 5, 3, 4, 4, 3, 5, 6, 5]
avg_objects = [
    np.mean([136, 129, 21, 74, 63, 68, 147, 82, 123, 71, 89, 42, 72, 187, 150]),
    np.mean([335, 197, 260, 185, 712]),
    np.mean([1375, 304, 325]),
    np.mean([3402, 6994, 1889, 3804]),
    np.mean([78, 141, 116, 3372]),
    np.mean([322, 218, 48]),
    np.mean([479, 787, 406, 1151, 225]),
    np.mean([1221, 1096, 292, 174, 1080, 1368]),
    np.mean([717, 890, 828, 556, 853]),
]

colors_scene = ['#264653', '#2A9D8F', '#8AB17D', '#E9C46A', '#F4A261',
                '#E76F51', '#457B9D', '#E63946', '#A8DADC']

bars = ax1.bar(scene_types, scene_counts, color=colors_scene, edgecolor='white', linewidth=1.5)
for bar, cnt, avg in zip(bars, scene_counts, avg_objects):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             f'{cnt}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
             f'~{int(avg)}\nobjs', ha='center', va='center', fontsize=8, color='white')

ax1.set_ylabel('Number of Scenes', fontsize=12)
ax1.set_title('50 Scenes Across 8+ Types\n(with average object count per scene)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 18)
ax1.grid(axis='y', alpha=0.3)

# --- Right: Simulation Feature Impact ---
ax2 = axes[1]
features = [
    'Rigid Body\nManipulation',
    'Articulated\nObjects',
    'Deformable\nBodies',
    'Cloths &\nFlexible',
    'Fluids',
    'Temperature\n& Cooking',
    'Cleaning\nTransitions',
    'Assembly\nRules',
]
pct_activities = [100, 78, 45, 52, 48, 35, 42, 15]

colors_feat = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(features)))
bars2 = ax2.barh(features, pct_activities, color=colors_feat, edgecolor='white', linewidth=1.5)
for bar, pct in zip(bars2, pct_activities):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
             f'{pct}%', ha='left', va='center', fontsize=11, fontweight='bold')

ax2.set_xlabel('% of Activities Requiring Feature', fontsize=12)
ax2.set_title('Simulation Features Required by Activities\n(estimated from paper analysis)',
              fontsize=13, fontweight='bold')
ax2.set_xlim(0, 115)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('b/d/b1k_26/asset/activity_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: activity_distribution.png")
