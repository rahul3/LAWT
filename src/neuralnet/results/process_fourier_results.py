import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def combine_csv_files(folder_path):
    all_files = []
    
    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    # Combine all CSV files into a single DataFrame
    combined_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    
    return combined_df

folder_path = "/mnt/wd_2tb/encoder_fourier_models/sign"
folder_path_2 = "/mnt/wd_2tb/evaluation/encoder_fourier"
combined_df_1 = combine_csv_files(folder_path)
combined_df_2 = combine_csv_files(folder_path_2)
combined_df = pd.concat([combined_df_1, combined_df_2], ignore_index=True)
# combined_df.to_csv("combined_results.csv", index=False)

df = combined_df.copy()

df.columns = [col.lower() for col in df.columns]
df.rename(columns={'dimension': 'dim'}, inplace=True)
df.rename(columns={'train_samples': 'train_sample'}, inplace=True)



# Set up the plot style
sns.set_style("whitegrid")
sns.set_palette("deep")

layers = df['num_layers'].unique()
test_samples = df['test_samples'].unique()

for layer in layers:
    df = combined_df.copy()

    df.columns = [col.lower() for col in df.columns]
    df.rename(columns={'dimension': 'dim'}, inplace=True)
    df.rename(columns={'train_samples': 'train_sample'}, inplace=True)
    df = df[df['num_layers'] == layer]

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
        plt.title(f'{layer} Layer Transformer Encoder - Relative Error vs Training Sample Size (Dimension {dim})')
        # Add caption with the number of test samples
        caption = f"Number of Test Samples: {test_samples[0]}"
        plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)

        # Add legend
        plt.legend(title='Operation', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout
        plt.tight_layout()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # Save the plot
        graphs_path = "/mnt/wd_2tb/evaluation/encoder_fourier/graphs_2"
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)
        plt.savefig(f'{graphs_path}/fourier_layers_{layer}_dim_{dim}_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

print("Plots have been generated and saved.")