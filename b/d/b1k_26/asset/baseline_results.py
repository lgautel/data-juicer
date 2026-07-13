"""
BEHAVIOR-1K 基线实验结果: 任务成功率对比 + 消融分析
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['DejaVu Sans']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Task Success Rates ---
ax1 = axes[0]
tasks = ['StoreDecoration', 'CollectTrash', 'CleanTable']
methods = ['RL-VMC (SAC)', 'RL-Prim. (PPO)', 'RL-Prim.-Hist.']
data = np.array([
    [0.0, 0.0, 0.0],
    [0.48, 0.42, 0.77],
    [0.55, 0.63, 0.88],
])
errs = np.array([
    [0.0, 0.0, 0.0],
    [0.06, 0.02, 0.08],
    [0.05, 0.03, 0.02],
])

x = np.arange(len(tasks))
width = 0.25
colors_bar = ['#264653', '#2A9D8F', '#E63946']

for i, (method, color) in enumerate(zip(methods, colors_bar)):
    bars = ax1.bar(x + i * width, data[i] * 100, width, yerr=errs[i] * 100,
                   label=method, color=color, capsize=4, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, data[i]):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     f'{val*100:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Activity', fontsize=12)
ax1.set_ylabel('Success Rate (%)', fontsize=12)
ax1.set_title('Task Success Rates by Method', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels(tasks, fontsize=11)
ax1.set_ylim(0, 105)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# --- Right: Ablation Study (RL-Prim.) ---
ax2 = axes[1]
conditions = ['Assistive Grasp\n+ Teleport', 'Assistive Grasp\n+ Full Motion', 'Physics Grasp\n+ Full Motion']
abl_data = np.array([
    [0.48, 0.42, 0.77],
    [0.46, 0.36, 0.73],
    [0.0, 0.0, 0.0],
])
abl_errs = np.array([
    [0.06, 0.02, 0.08],
    [0.04, 0.08, 0.03],
    [0.0, 0.0, 0.0],
])

task_colors = ['#457B9D', '#E9C46A', '#E76F51']
x2 = np.arange(len(conditions))

for i, (task, color) in enumerate(zip(tasks, task_colors)):
    bars = ax2.bar(x2 + i * width, abl_data[:, i] * 100, width, yerr=abl_errs[:, i] * 100,
                   label=task, color=color, capsize=4, edgecolor='white', linewidth=0.5)

ax2.set_xlabel('Physics Realism Configuration', fontsize=12)
ax2.set_ylabel('Success Rate (%)', fontsize=12)
ax2.set_title('Ablation: Impact of Physics Realism (RL-Prim.)', fontsize=13, fontweight='bold')
ax2.set_xticks(x2 + width)
ax2.set_xticklabels(conditions, fontsize=10)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(axis='y', alpha=0.3)

ax2.annotate('Physics grasping\ndrops to 0%',
             xy=(2 + width, 2), fontsize=10, color='#E63946', fontweight='bold',
             ha='center')

plt.tight_layout()
plt.savefig('b/d/b1k_26/asset/baseline_results.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: baseline_results.png")
