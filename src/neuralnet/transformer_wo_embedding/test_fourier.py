import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import pandas as pd
import numpy as np
from scipy.linalg import expm, signm, cosm, sinm, logm
from train_enc_fourier import NNMatrixData
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_logger

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_fourier_features, sigma=1.0):
        super(FourierFeatures, self).__init__()
        
        # Initialize B with random values from a normal distribution
        self.B = nn.Parameter(torch.randn(input_dim, num_fourier_features) * sigma, requires_grad=False)
        self.num_fourier_features = num_fourier_features

    def forward(self, x):
        # x should be of shape [batch_size, input_dim]
        # Project x onto B
        x_proj = 2 * math.pi * x @ self.B
        
        # Compute sine and cosine features
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
class FourierFeaturesEncoder(nn.Module):
    def __init__(self, input_dim, num_fourier_features, sigma=1.0):
        super(FourierFeaturesEncoder, self).__init__()
        self.fourier_features = FourierFeatures(input_dim, num_fourier_features, sigma)

    def forward(self, x):
        return self.fourier_features(x)


class MatrixApproximator(nn.Module):
    def __init__(self, input_dim, num_fourier_features, d_model, nhead, num_layers):
        super(MatrixApproximator, self).__init__()
        self.encoder = FourierFeaturesEncoder(input_dim, num_fourier_features)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.final_linear = nn.Linear(d_model, input_dim)
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: [batch_size, 5, 5]
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [batch_size, 25]
        ff_x = self.encoder(x_flat)  # [batch_size, 200]
        ff_x = ff_x.unsqueeze(1)  # [batch_size, 1, 200]
        tf_x = self.transformer_encoder(ff_x)  # [batch_size, 1, 200]
        tf_x = tf_x.squeeze(1)  # [batch_size, 200]
        output = self.final_linear(tf_x)  # [batch_size, 25]
        return output.view(batch_size, int(self.input_dim**0.5), int(self.input_dim**0.5))  # [batch_size, 5, 5]
    

class FrobeniusNormLoss(nn.Module):
    def __init__(self):
        super(FrobeniusNormLoss, self).__init__()

    def forward(self, output, target):
        frob_norm = torch.norm(output - target, p='fro', dim=(1, 2))
        return frob_norm.mean()

def evaluate_model(model, test_loader, criterion, logger):
    model.eval()
    total_relative_error = 0
    total_samples = 0
    
    relative_error_lst = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device).to(torch.float64), batch_y.to(device).to(torch.float64)
            predicted = model(batch_x)
            
            
            relative_error = torch.norm(predicted - batch_y, p='fro', dim=(1, 2)) / (torch.norm(batch_y, p='fro', dim=(1, 2)) + 1e-10)
            total_relative_error += relative_error.sum().item()
            total_samples += batch_x.size(0)
            relative_error_lst.append(relative_error)
            
            # mu = np.mean(np.log10(np_predicted))

    avg_relative_error = total_relative_error / total_samples
    logger.info(f'Average Relative Error: {avg_relative_error:.4f}')

    return relative_error_lst, avg_relative_error

def main():
    base_dir = "/mnt/wd_2tb/experiments/encoder_fourier"
    logger = get_logger(__name__, log_file=os.path.join(base_dir, "evaluation_results.log"))

    operation_lst = ["sign"]
    # operation_lst = ["exponential", "sign", "cos", "sin", "log"]
    dim_lst = [3, 5, 8]
    num_layers_lst = [2, 4, 8, 16]
    sample_size_lst = [2**k for k in range(5, 19)]

    results = []

    for operation in operation_lst:
        for dim in dim_lst:
            for num_layers in num_layers_lst:
                for sample_size in sample_size_lst:
                    try:
                        model_dir = os.path.join(base_dir, operation, f"dim_{dim}", f"layers_{num_layers}")
                        model_path = os.path.join(model_dir, f'{operation}_dim_{dim}_layers_{num_layers}_model.pth')
                        test_set_path = os.path.join(model_dir, "test", f'test_dataset_{sample_size}.pt')

                        if not os.path.exists(model_path) or not os.path.exists(test_set_path):
                            logger.info(f"Skipping {operation}, dim={dim}, layers={num_layers}, samples={sample_size} due to missing files.")
                            continue

                        logger.info(f"Evaluating model: {operation}, dim={dim}, layers={num_layers}, samples={sample_size}")

                        # Load the model
                        model = MatrixApproximator(input_dim=dim**2, num_fourier_features=100, d_model=200, nhead=8, num_layers=num_layers).to(device)
                        model.load_state_dict(torch.load(model_path))

                        # Load the test dataset
                        test_dataset = torch.load(test_set_path)
                        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=5)

                        # Evaluate the model
                        criterion = FrobeniusNormLoss()
                        relative_error, avg_relative_error = evaluate_model(model, test_loader, criterion, logger)
                        
                        
                        relative_error = torch.cat(relative_error)
                        mu = torch.mean(torch.log10(relative_error))
                        sigma = (relative_error.shape[0] -1)**-1 * torch.sum((torch.log10(relative_error) - mu)**2)
                        y_main = 10**mu 
                
                        u_shaded = 10**(mu - sigma)
                        l_shaded = 10**(mu + sigma)

                        results.append({
                            'Operation': operation,
                            'Dimension': dim,
                            'Num_Layers': num_layers,
                            'Sample_Size': sample_size,
                            'Average_Relative_Error': avg_relative_error,
                            'Mu': mu.item(),
                            'Sigma': sigma.item(),
                            'Y_Main': y_main.item(),
                            'U_Shaded': u_shaded.item(),
                            'L_Shaded': l_shaded.item()
                        })
                        
                        logger.info(results)

                    except Exception as e:
                        logger.error(f"An error occurred during evaluation: {str(e)}")
                        logger.error(f"Skipping this configuration and moving to the next one.")
                        continue

    # Save results to CSV and Excel
    df = pd.DataFrame(results)
    csv_path = os.path.join(base_dir, 'evaluation_results.csv')
    excel_path = os.path.join(base_dir, 'evaluation_results.xlsx')

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False, sheet_name='Results')

    logger.info(f"Evaluation results saved to {csv_path} and {excel_path}")

if __name__ == "__main__":
    main()