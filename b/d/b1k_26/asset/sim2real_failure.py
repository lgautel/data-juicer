"""
BEHAVIOR-1K Sim-to-Real 迁移失败原因分析
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['DejaVu Sans']

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- Left: Success Rates Comparison ---
ax1 = axes[0]
conditions = ['Simulation\n(RL-Prim.)', 'Real World\n(Optimal Policy)', 'Real World\n(Trained Policy)']
success_rates = [40, 22, 0]
colors_bar = ['#2A9D8F', '#E9C46A', '#E63946']

bars = ax1.bar(conditions, success_rates, color=colors_bar, edgecolor='white',
               linewidth=2, width=0.6)
for bar, val in zip(bars, success_rates):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.set_ylabel('Success Rate (%)', fontsize=13)
ax1.set_title('CollectTrash: Sim vs Real Success Rates', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 55)
ax1.grid(axis='y', alpha=0.3)

arrow_props = dict(arrowstyle='->', color='#E63946', lw=2)
ax1.annotate('', xy=(1, 22), xytext=(0, 40),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
ax1.annotate('', xy=(2, 0), xytext=(1, 22),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
ax1.text(0.5, 35, 'Actuation\ngap', ha='center', fontsize=9, color='gray', style='italic')
ax1.text(1.5, 15, 'Perception\ngap', ha='center', fontsize=9, color='gray', style='italic')

# --- Right: Failure Cause Breakdown ---
ax2 = axes[1]
scenarios = ['Simulation\n(S)', 'Real-Optimal\n(R-OP)', 'Real-Trained\n(R-TP)']

failure_types = ['Policy Errors', 'Grasping', 'Motion Planning', 'Object Detection', 'Navigation Compounding']
colors_fail = ['#457B9D', '#E63946', '#E9C46A', '#2A9D8F', '#F4A261']

sim_failures = [70, 0, 20, 0, 10]
real_opt_failures = [0, 40, 15, 30, 15]
real_trained_failures = [44, 38, 5, 8, 5]

data_stack = np.array([sim_failures, real_opt_failures, real_trained_failures])

bottom = np.zeros(3)
x = np.arange(3)
for i, (ftype, color) in enumerate(zip(failure_types, colors_fail)):
    bars = ax2.bar(x, data_stack[:, i], 0.55, bottom=bottom, label=ftype,
                   color=color, edgecolor='white', linewidth=0.5)
    for j, (b, val) in enumerate(zip(bars, data_stack[:, i])):
        if val > 8:
            ax2.text(b.get_x() + b.get_width() / 2, bottom[j] + val / 2,
                     f'{val}%', ha='center', va='center', fontsize=9,
                     color='white', fontweight='bold')
    bottom += data_stack[:, i]

ax2.set_xticks(x)
ax2.set_xticklabels(scenarios, fontsize=11)
ax2.set_ylabel('Failure Cause Distribution (%)', fontsize=13)
ax2.set_title('Failure Analysis: Sim vs Real', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.0, 1.0))
ax2.set_ylim(0, 110)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('b/d/b1k_26/asset/sim2real_failure.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: sim2real_failure.png")
