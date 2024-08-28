import os
import argparse
import csv
import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

from datagenerator import SingleDimData
from models import MatrixNet
from loss import FrobeniusNormLoss, LogFrobeniusNormLoss, MAPE

from graphs import training_val_loss

from common import get_logger, log_loss

# Setup seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

save_dir = "/home/rahulpadmanabhan/Development/ws1/masters_thesis_2/experiments/scalar/exp"
logger = get_logger(__name__, log_file=os.path.join(save_dir, "exp_3.log"))

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def calculate_relative_error(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))

if __name__ == "__main__":
    distribution = "uniform"
    coeff_lower = -1
    coeff_upper = 1
    batch_size = 128
    num_epochs = 100
    validation_interval = 10
    lr = 0.001

    test_samples = 2**15  # 16384
    k_values = range(5, 19)
    train_samples = [(2**k) + test_samples for k in k_values]
    logger.info(f"{train_samples=}, {test_samples=}")

    relative_errors = []

    for k, train_sample in zip(k_values, train_samples):
        actual_train_sample = int(train_sample) - test_samples
        logger.info(f"Training with {actual_train_sample} samples")
        
        # Load the saved datasets
        train_dataset = torch.load(os.path.join(save_dir, f'train_dataset_{str(actual_train_sample)}.pt'))
        test_dataset = torch.load(os.path.join(save_dir, f'test_dataset_{str(actual_train_sample)}.pt'))

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

        # Load the saved model
        model = MatrixNet(1).to(device)
        model.load_state_dict(torch.load(os.path.join(save_dir, f'exp_model_{str(actual_train_sample)}.pth')))

        model.eval()
        test_relative_error = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.view(-1, 1).to(device)
                y = y.view(-1, 1).to(device)

                test_output = model(x)
                test_relative_error += calculate_relative_error(y, test_output).item()

        test_relative_error /= len(test_loader)
        logger.info(f'Test Relative Error: {test_relative_error:.4f}')
        
        relative_errors.append((k, test_relative_error))

    # Plot the results
    k_values, errors = zip(*relative_errors)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, errors, marker='o')
    plt.xlabel('k (where number of samples = 2^k)')
    plt.ylabel('Relative Test Error')
    plt.title('Relative Test Error vs Number of Training Samples (2^k)')
    plt.grid(True)
    
    # Set x-ticks to show 2^k
    plt.xticks(k_values, [f'2^{k}' for k in k_values], rotation=45)
    
    # Use log scale for y-axis
    plt.yscale('log')
    
    # Add horizontal lines for powers of 10
    for y in [1e-1, 1e-2, 1e-3, 1e-4]:
        plt.axhline(y=y, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'relative_test_error_plot_2.png'))
    plt.close()

    logger.info("Plot saved as 'relative_test_error_plot.png'")