#!/usr/bin/env python
"""
Creates a bar chart showing maximum accuracy by dimension for each matrix operation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Output directory
OUTPUT_DIR = Path(__file__).parent / "graphs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
DATA_PATH = Path(__file__).parent / "combined_results_dim123.csv"
# DATA_PATH = Path(__file__).parent / "shallow_nn/shallow_nn_dim123_combined.csv"
# DATA_PATH = "/home/rahul3/projects/def-sbrugiap/rahul3/icprai_2026/LAWT/src/neuralnet/results/icprai_2026/combined_results_dim123.csv"
df = pd.read_csv(DATA_PATH)

# Map operation names to more readable labels
operation_labels = {
    'exp': 'Matrix Exponential',
    'log': 'Matrix Logarithm',
    'sign': 'Matrix Sign',
    'sin': 'Matrix Sine',
    'cos': 'Matrix Cosine', 
}

# Color palette for dimensions
dim_colors = {1: '#2ecc71', 2: '#e74c3c', 3: '#3498db'}

print("Data loaded successfully!")


def plot_max_accuracy_by_dimension():
    """Create a bar chart showing maximum accuracy by dimension for each operation."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_tol = df[df['tolerance'] == 0.05]
    
    # Calculate max accuracy achieved for each operation and dimension
    max_acc = df_tol.groupby(['operation', 'dim'])['accuracy'].max().reset_index()
    max_acc['accuracy_pct'] = max_acc['accuracy'] * 100
    max_acc['operation_label'] = max_acc['operation'].map(operation_labels)
    
    # Create grouped data
    x = np.arange(5)
    width = 0.28
    operations_order = ['exp', 'log', 'sign', 'sin', 'cos']
    
    for i, dim in enumerate([1, 2, 3]):
        heights = [max_acc[(max_acc['operation'] == op) & (max_acc['dim'] == dim)]['accuracy_pct'].values[0] 
                   for op in operations_order]
        bars = ax.bar(x + (i - 1) * width, heights, width, 
                     label=f'{dim}×{dim} Matrix', color=dim_colors[dim],
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, h in zip(bars, heights):
            if h > 2:
                ax.annotate(f'{h:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, h),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            elif h < 1:
                ax.annotate('~0%',
                           xy=(bar.get_x() + bar.get_width() / 2, h + 2),
                           ha='center', va='bottom', fontsize=8, color='#7f8c8d')
    
    ax.set_ylabel('Maximum Tolerance-Based Accuracy Achieved (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Matrix Function', fontsize=13, fontweight='bold')
    ax.set_title('Maximum Accuracy By Dimension (Tolerance = 0.05)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([operation_labels[op] for op in operations_order], fontsize=11)
    ax.legend(title='Matrix Size', loc='upper right', fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dnn_max_accuracy_by_dimension.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: dnn_max_accuracy_by_dimension.png")


if __name__ == "__main__":
    plot_max_accuracy_by_dimension()
    print(f"Output saved to: {OUTPUT_DIR}")
