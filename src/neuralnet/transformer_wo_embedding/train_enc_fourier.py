import torch
import torch.nn as nn
from scipy.linalg import expm, signm, cosm, sinm, logm
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import random_split
import datetime
import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore', message='logm result may be inaccurate')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import get_logger
from loss import RelativeErrorL1

torch.set_default_dtype(torch.float64)

# Setup seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def uniform_rand(size, low, high):
    """Generate a random tensor with uniform distribution with coefficients between low and high"""
    return torch.rand(size) * (high - low) + low


class NNMatrixData(Dataset):
    "Dataset for generating various types of matrices"

    def __init__(self, n_examples, operation="exponential", distribution="gaussian", dim=1, coeff_lower=-5, coeff_upper=5, **kwargs):
        super().__init__()
        self.n_examples = n_examples
        self.distribution = distribution

        if self.distribution == "gaussian":
            self.data = torch.randn(n_examples, dim, dim)
            if coeff_upper is not None:
                self.data = coeff_upper / math.sqrt(3.0) * self.data
        elif self.distribution == "uniform":
            self.data = uniform_rand((n_examples, dim, dim), coeff_lower, coeff_upper)
            
            
        if operation=="exponential" or operation=="exp":
            if dim == 1:
                self.target = torch.exp(self.data)
            else:
                self.target = torch.tensor(np.array([expm(m.numpy()) for m in self.data]))
        elif operation=="square":
            if dim == 1:
                self.target = self.data ** 2
            else:
                self.target = torch.tensor(np.array([m.numpy() @ m.numpy() for m in self.data]))
        elif operation=="sign":
            if dim == 1:
                self.target = torch.sign(self.data)
            else:
                self.target = torch.tensor(np.array([signm(m.numpy()) for m in self.data]))
        elif operation=="cos":
            if dim == 1:
                self.target = torch.cos(self.data)
            else:
                self.target = torch.tensor(np.array([cosm(m.numpy()) for m in self.data]))
        elif operation=="sin":
            if dim == 1:
                self.target = torch.sin(self.data)
            else:
                self.target = torch.tensor(np.array([sinm(m.numpy()) for m in self.data]))
        elif operation == "log":
            if dim == 1:
                self.target = torch.log(self.data)
            else:
                self.target = torch.tensor(np.array([logm(m.numpy()) for m in self.data]))
        else:
            self.target = torch.tensor(np.array([expm(m.numpy()) for m in self.data]))
            
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

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


# Usage
# x = torch.randn(10, 5, 5)
# y = torch.stack([torch.from_numpy(expm(x[i].numpy())) for i in range(x.shape[0])])
# model = MatrixApproximator(input_dim=25, num_fourier_features=100, d_model=200, nhead=8, num_layers=16)
# output = model(y)
# print(output.shape)  # Should be torch.Size([10, 5, 5])

if __name__ == '__main__':
    # Experiment parameters
    num_layers_lst = [2, 4, 8, 16]
    # operation_lst = ["exponential", "square", "sign", "cos", "sin", "log"]
    # operation_lst = ["exponential", "log", "sin", "cos"]
    operation_lst = ["log", "sin", "cos"]
    dim_lst = [3,5,8]
    sample_size_lst = [2**k for k in range(5, 19)]
    test_size = 2**14

    for operation in operation_lst: 
        for dim in dim_lst:
            for num_layers in num_layers_lst:
                for sample_size in sample_size_lst:
                    ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                    save_dir = f"/home/rahul3/projects/def-sbrugiap/rahul3/icprai_2026/fourier_encoder/{ID}/{operation}/dim_{dim}/layers_{num_layers}"
                    # save_dir = f"/home/rahulpadmanabhan/projects/ws1/experimentsencoder_fourier_models/{operation}/dim_{dim}/layers_{num_layers}"
                    # save_dir = f"/mnt/wd_2tb/thesis_transformers/experiments/encoder_fourier/{ID}"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    logger = get_logger(__name__, log_file=os.path.join(save_dir, f"{operation}_{ID}.log"))
                    logger.info(f"Running experiment for operation: {operation}, dim: {dim}, num_layers: {num_layers}, sample_size: {sample_size}, test_size: {test_size}")

                    # Set up training parameters
                    num_epochs = 600
                    batch_size = 64
                    learning_rate = 0.001

                    try:
                        # Create dataset and dataloader
                        dataset = NNMatrixData(n_examples=sample_size+test_size, distribution="gaussian", dim=dim, operation=operation, coeff_lower=-5, coeff_upper=5)
                        train_dataset, test_dataset = random_split(dataset, [sample_size, test_size])   
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

                        logger.info(f"Number of training examples: {len(train_dataset)}")
                        logger.info(f"Number of test examples: {len(test_dataset)}")
                        
                        # Save train and test datasets
                        train_dir = os.path.join(save_dir, "train")
                        test_dir = os.path.join(save_dir, "test")
                        logger.info(f"Saving train and test datasets to {train_dir} and {test_dir}")
                        if not os.path.exists(train_dir):
                            os.makedirs(train_dir)
                        if not os.path.exists(test_dir):
                            os.makedirs(test_dir)
                        
                        torch.save(train_dataset, os.path.join(train_dir, f'train_dataset_{sample_size}.pt'))
                        torch.save(test_dataset, os.path.join(test_dir, f'test_dataset_{sample_size}.pt'))
                        
                        logger.info(f"Saved train dataset with {len(train_dataset)} samples")
                        logger.info(f"Saved test dataset with {len(test_dataset)} samples")

                        # Initialize model, loss function, and optimizer
                        model = MatrixApproximator(input_dim=dim**2, num_fourier_features=3*dim**2, d_model=6*dim**2, nhead=dim**2, num_layers=num_layers).to(device)
                        
                        # Use RelativeErrorL1 for exponential, FrobeniusNormLoss for others
                        if operation == "exponential":
                            criterion = RelativeErrorL1()
                            logger.info("Using RelativeErrorL1 loss for exponential operation")
                        else:
                            criterion = FrobeniusNormLoss()
                            logger.info("Using FrobeniusNormLoss")
                        
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                        # Training loop
                        for epoch in range(num_epochs):
                            model.train()
                            total_loss = 0
                            for batch_x, batch_y in train_loader:
                                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                                optimizer.zero_grad()
                                output = model(batch_x)
                                loss = criterion(output, batch_y)
                                loss.backward()
                                optimizer.step()
                                total_loss += loss.item()
                
                            avg_loss = total_loss / len(train_loader)
                            if (epoch + 1) % 10 == 0:
                                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

                        logger.info("Training completed.")

                        # Initialize variables to accumulate errors
                        total_relative_error = 0
                        total_samples = 0
                        relative_error_lst = []

                        # Test the model
                        model.eval()
                        with torch.no_grad():
                            for batch_x, batch_y in test_loader:
                                batch_x, batch_y = batch_x.to(device).to(torch.float64), batch_y.to(device).to(torch.float64)
                                predicted = model(batch_x)
                                
                                # Calculate and accumulate relative error for this batch
                                relative_error = torch.norm(predicted - batch_y, p='fro', dim=(1, 2)) / (torch.norm(batch_y, p='fro', dim=(1, 2)) + 1e-10)
                                total_relative_error += relative_error.sum().item()
                                total_samples += batch_x.size(0)
                                relative_error_lst.append(relative_error)

                        # Calculate average relative error over entire test set
                        avg_relative_error = total_relative_error / total_samples   
                        logger.info(f'Average Relative Error: {avg_relative_error:.4f}')
                        
                        relative_error_tensor = torch.cat(relative_error_lst)
                        mu = torch.mean(torch.log10(relative_error_tensor))
                        sigma = (relative_error_tensor.shape[0] -1)**-1 * torch.sum((torch.log10(relative_error_tensor) - mu)**2)
                        y_main = 10**mu 
                
                        u_shaded = 10**(mu - sigma)
                        l_shaded = 10**(mu + sigma)
                
                        # Print out some sample predictions vs actuals
                        logger.info("\nSample Predictions vs Actuals:")
                        model.eval()
                        with torch.no_grad():
                            # Get a small batch of test data
                            sample_x, sample_y = next(iter(test_loader))
                            sample_x, sample_y = sample_x.to(device), sample_y.to(device)
                            
                            # Generate predictions
                            sample_pred = model(sample_x)
                            
                            # Print a few examples
                            for i in range(min(5, len(sample_x))):  # Print up to 5 examples
                                logger.info(f"\nExample {i+1}:")
                                logger.info("Input:")
                                logger.info(sample_x[i].cpu().numpy())
                                logger.info("\nPrediction:")
                                logger.info(sample_pred[i].cpu().numpy())
                                logger.info("\nActual:")
                                logger.info(sample_y[i].cpu().numpy())
                                logger.info("\nRelative Error:")
                                rel_error = torch.norm(sample_pred[i] - sample_y[i], p='fro') / torch.norm(sample_y[i], p='fro')
                                logger.info(f"{rel_error.item():.4f}")
                                logger.info("-" * 50)

                        # =====================================================
                        # Tolerance-based evaluation with 10,000 samples
                        # Using d1 metric (L1 relative error) to match deepnetwork_hd_train.py
                        # =====================================================
                        logger.info("\n" + "=" * 50)
                        logger.info("Tolerance-based Evaluation (10,000 samples)")
                        logger.info("=" * 50)
                        
                        evaluation_samples = 10000
                        evaluation_dataset = NNMatrixData(
                            n_examples=evaluation_samples,
                            distribution="gaussian",
                            dim=dim,
                            operation=operation,
                            coeff_lower=-5,
                            coeff_upper=5
                        )
                        evaluation_loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
                        
                        tols = [0.05, 0.02, 0.01, 0.005]
                        evaluations_correct = {tol: 0 for tol in tols}
                        
                        model.eval()
                        with torch.no_grad():
                            for eval_idx, (batch_x, batch_y) in enumerate(evaluation_loader):
                                batch_x = batch_x.to(device).to(torch.float64)
                                batch_y = batch_y.to(device).to(torch.float64)
                                output = model(batch_x)
                                
                                # Compute error matrix
                                error = torch.abs(output - batch_y)
                                
                                # Flatten each sample to compute d1 error per sample
                                error_flat = error.view(batch_x.size(0), -1)  # (batch, dim*dim)
                                y_flat = torch.abs(batch_y).view(batch_x.size(0), -1)  # (batch, dim*dim)
                                
                                # Compute d1 error per sample: sum(|error|) / sum(|y|)
                                # This matches LAWT's d1 metric: sum(|hyp - tgt|) / sum(|tgt|)
                                d1_error = error_flat.sum(dim=1) / (y_flat.sum(dim=1) + 1e-12)  # (batch,)
                                
                                for tol in tols:
                                    # A sample is correct if its d1 error is below tolerance
                                    sample_correct = d1_error < tol  # (batch,)
                                    correct_count = sample_correct.sum().item()
                                    evaluations_correct[tol] += correct_count
                        
                        # Log tolerance-based results
                        tolerance_accuracies = {}
                        for tol in tols:
                            accuracy = evaluations_correct[tol] / evaluation_samples
                            tolerance_accuracies[tol] = accuracy
                            logger.info(f"Tolerance {tol}: {evaluations_correct[tol]}/{evaluation_samples} correct, Accuracy: {accuracy:.4f}")

                        # Save the model
                        if not os.path.exists(os.path.join(save_dir, 'models')):
                            os.makedirs(os.path.join(save_dir, 'models'))
                            
                        model_save_path = os.path.join(save_dir, 'models', f'{operation}_dim_{dim}_layers_{num_layers}_{sample_size}_model.pth')
                        torch.save(model.state_dict(), model_save_path)
                        logger.info(f"Model saved to {model_save_path}")
                        
                        # Create a pandas DataFrame to store experiment results
                        experiment_data = {
                            'Operation': operation,
                            'Dimension': dim,
                            'Num_Layers': num_layers,
                            'Learning_Rate': learning_rate,
                            'Batch_Size': batch_size,
                            'Num_Epochs': num_epochs,
                            'Average_Relative_Error': avg_relative_error,
                            'Model_Path': model_save_path,
                            'Training_Set_Path': os.path.join(train_dir, f'train_dataset.pt'),
                            'Test_Set_Path': os.path.join(test_dir, f'test_dataset.pt'),
                            'Train_Samples': len(train_dataset),
                            'Test_Samples': len(test_dataset),
                            'Mu': mu.item(),
                            'Sigma': sigma.item(),
                            'Y_Main': y_main.item(),
                            'U_Shaded': u_shaded.item(),
                            'L_Shaded': l_shaded.item(),
                            # Tolerance-based accuracy results
                            'Accuracy_tol_0.05': tolerance_accuracies[0.05],
                            'Accuracy_tol_0.02': tolerance_accuracies[0.02],
                            'Accuracy_tol_0.01': tolerance_accuracies[0.01],
                            'Accuracy_tol_0.005': tolerance_accuracies[0.005],
                            'Evaluation_Samples': evaluation_samples
                        }
                        
                        df = pd.DataFrame([experiment_data])
                        
                        # Save to CSV
                        csv_path = os.path.join(save_dir, f'experiment_results_sample_size_{sample_size}.csv')
                        df.to_csv(csv_path, index=False, mode='a', header=not os.path.exists(csv_path))
                        logger.info(f"Experiment results also saved to {csv_path}")

                    except Exception as e:
                        logger.error(f"An error occurred during the experiment: {str(e)}")
                        logger.error(f"Skipping this configuration and moving to the next one.")
                        continue
