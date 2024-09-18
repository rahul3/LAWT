import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the combined CSV file
df = pd.read_csv('/mnt/wd_2tb/thesis_transformers/experiments/shallow_net/csv_results/shallownet_combined_results.csv')

# Set up the plot style
sns.set_style("whitegrid")
sns.set_palette("deep")

# Create a new figure for each dimension
for dim in df['dim'].unique():
    plt.figure(figsize=(12, 8))
    
    # Filter data for the current dimension
    dim_data = df[df['dim'] == dim]
    
    # Iterate through each operation
    for operation in dim_data['operation'].unique():
        op_data = dim_data[dim_data['operation'] == operation]
        
        # Sort data by train_sample to ensure correct line plot
        op_data = op_data.sort_values('train_sample')
        
        # Plot the main line
        plt.plot(op_data['train_sample'], op_data['y_main'], label=operation.capitalize(), marker='o')
        
        # Add shaded area
        plt.fill_between(op_data['train_sample'], op_data['l_shaded'], op_data['u_shaded'], alpha=0.2)

    # Set the scale of x-axis to logarithmic
    plt.xscale('log', base=2)
    plt.yscale('log')

    # Set custom x-ticks
    x_ticks = [2**k for k in range(5, 19)]
    plt.xticks(x_ticks, fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xticks(x_ticks, [f'2^{k}' for k in range(5, 19)], rotation=45)

    # Set labels and title
    plt.xlabel('Training Sample Size')
    plt.ylabel(r"Relative Error - $L_{Frobenius}(\mu)$")
    plt.title(f'Shallow Neural Network - Relative Error vs Training Sample Size for Dimension {dim}')

    # Add legend
    plt.legend(title='Operation', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Save the plot
    plt.savefig(f'/mnt/wd_2tb/thesis_transformers/experiments/shallow_net/plots/snn_dimension_{dim}_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Plots have been generated and saved.")