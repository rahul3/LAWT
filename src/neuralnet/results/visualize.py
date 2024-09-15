import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('/mnt/wd_2tb/thesis_transformers/experiments/deepnetwork/csv_results/results_sin.csv')

# Set up the plot style
sns.set_palette("deep")

# Create a new figure
plt.figure(figsize=(12, 8))

# Iterate through each dimension
for dim in df['dim'].unique():
    dim_data = df[df['dim'] == dim]
    
    # Plot the main line
    plt.plot(dim_data['train_sample'], dim_data['y_main'], label=f'Dim {dim}', marker='o')
    
    # Add shaded area
    plt.fill_between(dim_data['train_sample'], dim_data['l_shaded'], dim_data['u_shaded'], alpha=0.2)

# Set the scale of x-axis to logarithmic
plt.xscale('log', base=2)
plt.yscale('log')

# Set custom x-ticks
x_ticks = [2**k for k in range(5, 19)]
plt.xticks(x_ticks)

# Set labels and title
plt.xlabel('Training Sample Size')
plt.ylabel(r"Relative Error - $L_{Frobenius}\left(\mu\right)$")
plt.title('Test Loss vs Training Sample Size for Sin Operation')

# Add legend
plt.legend(title='Matrix Dimension', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display the plot
plt.tight_layout()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()