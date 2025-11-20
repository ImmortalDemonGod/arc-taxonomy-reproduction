#!/usr/bin/env python3
"""
Figure 7.5: S3-A Performance Profiles Scatter Plot
Shows 4 distinct failure modes within S3-A category via cell vs grid accuracy.
Section 7.4: Demonstrates diagnostic granularity of the framework.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)

# Paths
BASE_DIR = Path(__file__).parent.parent
CHAMPION_FILE = BASE_DIR / "paper/champion_FINAL_CLEAR.txt"
OUTPUT_PATH = BASE_DIR / "paper/figures/s3_performance_profiles.png"

# Data from champion_FINAL_CLEAR.txt, Section 2.5 (lines 301-330)
# And Table 7.2 in paper (lines 613-621)

S3A_TASKS = {
    # Perfect Solvers (1 task)
    '445eab21': {'cell': 100.0, 'grid': 100.0, 'group': 'Perfect Solvers'},
    
    # Partial Success (2 tasks)
    '95990924': {'cell': 83.7, 'grid': 10.0, 'group': 'Partial Success'},
    'e50d258f': {'cell': 51.5, 'grid': 2.2, 'group': 'Partial Success'},
    
    # Compositional Failure (11 tasks) - High cell (>80%), zero grid
    '31aa019c': {'cell': 97.6, 'grid': 0.0, 'group': 'Compositional Failure'},
    'a2fd1cf0': {'cell': 97.0, 'grid': 0.0, 'group': 'Compositional Failure'},
    '0a938d79': {'cell': 94.1, 'grid': 0.0, 'group': 'Compositional Failure'},
    'b548a754': {'cell': 92.7, 'grid': 0.0, 'group': 'Compositional Failure'},
    'b6afb2da': {'cell': 88.2, 'grid': 0.0, 'group': 'Compositional Failure'},
    '928ad970': {'cell': 87.9, 'grid': 0.0, 'group': 'Compositional Failure'},
    '67a423a3': {'cell': 87.9, 'grid': 0.0, 'group': 'Compositional Failure'},
    'ec883f72': {'cell': 86.8, 'grid': 0.0, 'group': 'Compositional Failure'},
    '50cb2852': {'cell': 86.5, 'grid': 0.0, 'group': 'Compositional Failure'},
    'f35d900a': {'cell': 85.4, 'grid': 0.0, 'group': 'Compositional Failure'},
    '25d487eb': {'cell': 82.3, 'grid': 0.0, 'group': 'Compositional Failure'},
    
    # Low Affinity & Unsolved (4 tasks) - Low cell (<80%), zero grid
    '9af7a82c': {'cell': 77.7, 'grid': 0.0, 'group': 'Low Affinity'},
    'c909285e': {'cell': 74.2, 'grid': 0.0, 'group': 'Low Affinity'},
    '913fb3ed': {'cell': 69.4, 'grid': 0.0, 'group': 'Low Affinity'},
    '00d62c1b': {'cell': 61.4, 'grid': 0.0, 'group': 'Low Affinity'}
}

GROUP_COLORS = {
    'Perfect Solvers': '#2ecc71',  # Green
    'Partial Success': '#3498db',   # Blue
    'Compositional Failure': '#e74c3c',  # Red
    'Low Affinity': '#95a5a6'  # Gray
}

GROUP_MARKERS = {
    'Perfect Solvers': '*',
    'Partial Success': 'o',
    'Compositional Failure': '^',
    'Low Affinity': 's'
}

GROUP_SIZES = {
    'Perfect Solvers': 400,
    'Partial Success': 200,
    'Compositional Failure': 150,
    'Low Affinity': 150
}

def main():
    # Prepare data
    groups = {}
    for task_id, data in S3A_TASKS.items():
        group = data['group']
        if group not in groups:
            groups[group] = {'cells': [], 'grids': [], 'task_ids': []}
        groups[group]['cells'].append(data['cell'])
        groups[group]['grids'].append(data['grid'])
        groups[group]['task_ids'].append(task_id)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot threshold regions
    # Compositional Gap region (cell > 80%, grid < 10%)
    ax.axhspan(0, 10, xmin=0.8, xmax=1.0, alpha=0.15, color='red', zorder=0)
    ax.text(90, 5, 'Compositional Gap\nRegion', ha='center', va='center',
           fontsize=11, fontweight='bold', color='red', style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor='red', linewidth=2, alpha=0.8))
    
    # Threshold lines
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7, 
              label='Grid accuracy threshold (10%)')
    ax.axvline(x=80, color='orange', linestyle='--', linewidth=2, alpha=0.7,
              label='Cell accuracy threshold (80%)')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1.5, alpha=0.4, 
           label='y=x (perfect composition)')
    
    # Plot each group
    for group_name in ['Perfect Solvers', 'Partial Success', 'Compositional Failure', 'Low Affinity']:
        if group_name in groups:
            data = groups[group_name]
            ax.scatter(data['cells'], data['grids'],
                      c=GROUP_COLORS[group_name],
                      marker=GROUP_MARKERS[group_name],
                      s=GROUP_SIZES[group_name],
                      label=f'{group_name} (n={len(data["cells"])})',
                      edgecolors='black', linewidth=1.5,
                      alpha=0.8, zorder=3)
            
            # Annotate example tasks
            if group_name == 'Perfect Solvers':
                ax.annotate('445eab21', xy=(data['cells'][0], data['grids'][0]),
                          xytext=(data['cells'][0]-10, data['grids'][0]-8),
                          fontsize=8, ha='center',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='green', linewidth=1.5),
                          arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
            elif group_name == 'Compositional Failure':
                # Annotate the most extreme example
                max_cell_idx = data['cells'].index(max(data['cells']))
                ax.annotate(data['task_ids'][max_cell_idx], 
                          xy=(data['cells'][max_cell_idx], data['grids'][max_cell_idx]),
                          xytext=(data['cells'][max_cell_idx]-5, 15),
                          fontsize=8, ha='center',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='red', linewidth=1.5),
                          arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
    
    # Styling
    ax.set_xlabel('Cell-Level Accuracy (%)\n(Local Pattern Understanding)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Grid-Level Accuracy (%)\n(Global Solution Synthesis)', 
                 fontsize=13, fontweight='bold')
    ax.set_title('S3-A Performance Profiles: Four Distinct Failure Modes\n' +
                'Diagnostic Granularity Within a Single Category (18 validation tasks)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(50, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, 
             edgecolor='black', ncol=1)
    
    # Add interpretation boxes
    interpretations = [
        {'pos': (55, 95), 'text': '✓ Architecture can solve\n(compositionally shallow)', 
         'color': 'green'},
        {'pos': (68, 75), 'text': '~ Partial capacity\n(edge of ceiling)', 
         'color': 'blue'},
        {'pos': (92, 85), 'text': '✗ Learns parts, not whole\n(compositional bottleneck)', 
         'color': 'red'},
        {'pos': (68, 30), 'text': '✗ Low local affinity\n(need different priors)', 
         'color': 'gray'}
    ]
    
    for interp in interpretations:
        ax.text(interp['pos'][0], interp['pos'][1], interp['text'],
               fontsize=9, ha='center', va='center', style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor=interp['color'], linewidth=2, alpha=0.9))
    
    # Add statistics box
    stats_text = ('Mean Grid Accuracy by Group:\n'
                 '• Perfect: 100.0%\n'
                 '• Partial: 6.1%\n'
                 '• Compositional: 0.0%\n'
                 '• Low Affinity: 0.0%\n\n'
                 'Mean Cell Accuracy:\n'
                 '• Compositional: 88.5%\n'
                 '→ Architectural ceiling!')
    ax.text(0.98, 0.65, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow',
                    edgecolor='black', linewidth=2, alpha=0.95))
    
    plt.tight_layout()
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_PATH}")
    print(f"  Total S3-A validation tasks: {len(S3A_TASKS)}")
    print(f"  Compositional Failure: {len(groups['Compositional Failure']['cells'])} tasks")
    print(f"  Mean cell accuracy (comp. failure): {np.mean(groups['Compositional Failure']['cells']):.1f}%")

if __name__ == "__main__":
    main()
