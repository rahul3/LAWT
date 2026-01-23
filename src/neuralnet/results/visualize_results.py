#!/usr/bin/env python3
"""
Visualize shallow neural network experiment results.

Creates comprehensive visualizations for each operation (exp, log, sign, sin, cos),
showing performance across dimensions, training samples, and tolerance levels.

Requirements:
    - pandas
    - matplotlib
    - seaborn
    - numpy

Usage:
    python visualize_results.py --csv_path /path/to/results.csv --output_dir /path/to/output/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse


# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def load_and_prepare_data(csv_path):
    """Load and prepare the data for visualization."""
    df = pd.read_csv(csv_path)
    
    # Ensure correct data types
    df['dim'] = df['dim'].astype(int)
    df['train_sample'] = df['train_sample'].astype(int)
    df['tolerance'] = df['tolerance'].astype(float)
    df['correct_predictions'] = df['correct_predictions'].astype(float)
    
    return df


def create_operation_plots(df, operation, output_dir):
    """
    Create comprehensive plots for a single operation.
    
    Args:
        df: DataFrame with all data
        operation: Operation name (e.g., 'sin', 'cos', 'log', 'exp', 'sign')
        output_dir: Directory to save plots
    """
    # Filter data for this operation
    op_df = df[df['operation'] == operation].copy()
    
    if len(op_df) == 0:
        print(f"Warning: No data found for operation '{operation}'")
        return
    
    # Create output directory for this operation
    op_dir = Path(output_dir) / operation
    op_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique dimensions
    dims = sorted(op_df['dim'].unique())
    
    print(f"\nCreating plots for {operation}...")
    print(f"  Dimensions: {dims}")
    print(f"  Training samples: {sorted(op_df['train_sample'].unique())}")
    print(f"  Tolerances: {sorted(op_df['tolerance'].unique())}")
    
    # Plot 1: Line plot - Accuracy vs Training Samples (separate plot for each dimension)
    create_accuracy_vs_samples_plot(op_df, operation, op_dir, dims)
    
    # Plot 2: Heatmap - Performance across dimensions and training samples for each tolerance
    create_heatmap_plots(op_df, operation, op_dir)
    
    # Plot 3: Faceted plot - All dimensions together
    create_faceted_tolerance_plot(op_df, operation, op_dir)
    
    # Plot 4: Learning curves - Compare all dimensions at highest tolerance
    create_learning_curves_comparison(op_df, operation, op_dir, dims)
    
    print(f"  ✓ Plots saved to {op_dir}")


def create_accuracy_vs_samples_plot(op_df, operation, op_dir, dims):
    """Create line plots showing accuracy vs training samples for each dimension."""
    n_dims = len(dims)
    n_cols = min(3, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_dims == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Color palette for tolerances
    tolerances = sorted(op_df['tolerance'].unique())
    colors = sns.color_palette("viridis", len(tolerances))
    
    for idx, dim in enumerate(dims):
        ax = axes[idx]
        dim_df = op_df[op_df['dim'] == dim].copy()
        
        for tol, color in zip(tolerances, colors):
            tol_df = dim_df[dim_df['tolerance'] == tol].sort_values('train_sample')
            ax.plot(tol_df['train_sample'], tol_df['correct_predictions'], 
                   marker='o', linewidth=2, markersize=6, label=f'tol={tol}', color=color)
        
        ax.set_xlabel('Training Samples', fontweight='bold')
        ax.set_ylabel('Accuracy (Correct Predictions)', fontweight='bold')
        ax.set_title(f'{operation.upper()} - Dimension {dim}', fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.set_ylim([0, 1.05])
        
        # Format x-axis labels
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Hide extra subplots if any
    for idx in range(n_dims, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{operation.upper()} - Accuracy vs Training Samples by Dimension', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(op_dir / f'{operation}_accuracy_vs_samples.png', bbox_inches='tight')
    plt.close()


def create_heatmap_plots(op_df, operation, op_dir):
    """Create heatmap showing performance across dimensions and training samples."""
    tolerances = sorted(op_df['tolerance'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, tol in enumerate(tolerances):
        ax = axes[idx]
        tol_df = op_df[op_df['tolerance'] == tol].copy()
        
        # Pivot table for heatmap
        pivot_df = tol_df.pivot_table(
            values='correct_predictions',
            index='dim',
            columns='train_sample',
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Accuracy'},
                   linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Training Samples', fontweight='bold')
        ax.set_ylabel('Dimension', fontweight='bold')
        ax.set_title(f'{operation.upper()} - Tolerance: {tol}', fontsize=13, fontweight='bold')
        
        # Format x-axis labels
        ax.set_xticklabels([f'{int(float(x.get_text())):,}' for x in ax.get_xticklabels()], 
                          rotation=45, ha='right')
    
    plt.suptitle(f'{operation.upper()} - Performance Heatmap Across Dimensions and Training Samples', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(op_dir / f'{operation}_heatmap.png', bbox_inches='tight')
    plt.close()


def create_faceted_tolerance_plot(op_df, operation, op_dir):
    """Create faceted plot showing all tolerances together."""
    tolerances = sorted(op_df['tolerance'].unique())
    dims = sorted(op_df['dim'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Color palette for dimensions
    colors = sns.color_palette("tab10", len(dims))
    
    for idx, tol in enumerate(tolerances):
        ax = axes[idx]
        tol_df = op_df[op_df['tolerance'] == tol].copy()
        
        for dim, color in zip(dims, colors):
            dim_df = tol_df[tol_df['dim'] == dim].sort_values('train_sample')
            ax.plot(dim_df['train_sample'], dim_df['correct_predictions'], 
                   marker='o', linewidth=2, markersize=6, label=f'dim={dim}', color=color)
        
        ax.set_xlabel('Training Samples', fontweight='bold')
        ax.set_ylabel('Accuracy (Correct Predictions)', fontweight='bold')
        ax.set_title(f'Tolerance: {tol}', fontsize=13, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True, ncol=2)
        ax.set_ylim([0, 1.05])
        
        # Format x-axis labels
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.suptitle(f'{operation.upper()} - Accuracy vs Training Samples (All Dimensions)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(op_dir / f'{operation}_faceted_tolerances.png', bbox_inches='tight')
    plt.close()


def create_learning_curves_comparison(op_df, operation, op_dir, dims):
    """Create comparison of learning curves across dimensions at highest tolerance."""
    # Use the highest (most lenient) tolerance
    max_tol = op_df['tolerance'].max()
    tol_df = op_df[op_df['tolerance'] == max_tol].copy()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Color palette for dimensions
    colors = sns.color_palette("husl", len(dims))
    
    for dim, color in zip(dims, colors):
        dim_df = tol_df[tol_df['dim'] == dim].sort_values('train_sample')
        ax.plot(dim_df['train_sample'], dim_df['correct_predictions'], 
               marker='o', linewidth=2.5, markersize=8, label=f'Dimension {dim}', 
               color=color, alpha=0.8)
    
    ax.set_xlabel('Training Samples', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (Correct Predictions)', fontsize=13, fontweight='bold')
    ax.set_title(f'{operation.upper()} - Learning Curves Comparison (Tolerance: {max_tol})', 
                fontsize=15, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='best', fontsize=11)
    ax.set_ylim([0, 1.05])
    
    # Format x-axis labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add horizontal line at 0.95 for reference
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='95% accuracy')
    
    plt.tight_layout()
    plt.savefig(op_dir / f'{operation}_learning_curves_comparison.png', bbox_inches='tight')
    plt.close()


def create_summary_comparison(df, output_dir):
    """Create a summary comparison across all operations."""
    print("\nCreating summary comparison plot...")
    
    operations = sorted(df['operation'].unique())
    max_tol = df['tolerance'].max()
    
    # Filter for highest tolerance and highest training samples for each operation
    summary_df = df[df['tolerance'] == max_tol].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Final accuracy comparison across operations and dimensions
    ax = axes[0]
    
    # Get the maximum training sample for each operation-dimension pair
    max_samples = summary_df.groupby(['operation', 'dim'])['train_sample'].max().reset_index()
    max_samples.columns = ['operation', 'dim', 'max_train_sample']
    
    # Merge to get accuracy at max training samples
    comparison_df = summary_df.merge(
        max_samples, 
        left_on=['operation', 'dim', 'train_sample'],
        right_on=['operation', 'dim', 'max_train_sample']
    )
    
    # Pivot for plotting
    pivot_comp = comparison_df.pivot_table(
        values='correct_predictions',
        index='dim',
        columns='operation',
        aggfunc='mean'
    )
    
    pivot_comp.plot(kind='bar', ax=ax, width=0.8, colormap='Set2')
    ax.set_xlabel('Dimension', fontweight='bold', fontsize=12)
    ax.set_ylabel('Final Accuracy', fontweight='bold', fontsize=12)
    ax.set_title(f'Final Accuracy Comparison (Max Training Samples, tol={max_tol})', 
                fontsize=13, fontweight='bold')
    ax.legend(title='Operation', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Plot 2: Sample efficiency - training samples needed to reach 90% accuracy
    ax = axes[1]
    
    threshold = 0.90
    efficiency_data = []
    
    for operation in operations:
        for dim in sorted(df['dim'].unique()):
            dim_op_df = summary_df[
                (summary_df['operation'] == operation) & 
                (summary_df['dim'] == dim)
            ].sort_values('train_sample')
            
            # Find first sample size that reaches threshold
            above_threshold = dim_op_df[dim_op_df['correct_predictions'] >= threshold]
            if len(above_threshold) > 0:
                samples_needed = above_threshold['train_sample'].min()
                efficiency_data.append({
                    'operation': operation,
                    'dim': dim,
                    'samples_needed': samples_needed
                })
    
    if efficiency_data:
        eff_df = pd.DataFrame(efficiency_data)
        pivot_eff = eff_df.pivot_table(
            values='samples_needed',
            index='dim',
            columns='operation',
            aggfunc='mean'
        )
        
        pivot_eff.plot(kind='bar', ax=ax, width=0.8, colormap='Set2')
        ax.set_xlabel('Dimension', fontweight='bold', fontsize=12)
        ax.set_ylabel('Training Samples Needed', fontweight='bold', fontsize=12)
        ax.set_title(f'Sample Efficiency (Samples to reach {threshold*100:.0f}% accuracy)', 
                    fontsize=13, fontweight='bold')
        ax.legend(title='Operation', frameon=True, fancybox=True, shadow=True)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.suptitle('Cross-Operation Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'summary_comparison.png', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Summary comparison saved to {Path(output_dir) / 'summary_comparison.png'}")


def main():
    """Main function to generate all visualizations."""
    parser = argparse.ArgumentParser(
        description="Visualize shallow neural network experiment results"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/rahul3/projects/def-sbrugiap/rahul3/icprai_2026/results/shallow_nn_results.csv",
        help="Path to consolidated results CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same directory as this script)"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent / "plots"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {args.csv_path}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = load_and_prepare_data(args.csv_path)
    
    print(f"\nLoaded {len(df)} rows")
    print(f"Operations: {sorted(df['operation'].unique())}")
    print(f"Dimensions: {sorted(df['dim'].unique())}")
    
    # Create plots for each operation
    operations = sorted(df['operation'].unique())
    
    for operation in operations:
        create_operation_plots(df, operation, output_dir)
    
    # Create summary comparison
    create_summary_comparison(df, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ All visualizations complete!")
    print(f"✓ Plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
