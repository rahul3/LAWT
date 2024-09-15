# Encode with Fourier Feature embeddings

import os
import datetime
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagenerator import NNData, NNMatrixData
from models import MatrixFunctionTransformer
from loss import FrobeniusNormLoss, LogFrobeniusNormLoss, MAPE

from graphs import training_val_loss

from common import get_logger, log_loss

torch.set_default_dtype(torch.float64)

# Setup seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate ID based on current datetime
ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
# ###########################################################################################
operation = "sign"
# ###########################################################################################
# save_dir = f"/mnt/wd_2tb/thesis_transformers/experiments/transformer_wo_embedding/{operation}/encoder_{ID}"
save_dir = f"/home/rahulpadmanabhan/projects/ws1/experiments/transformer_encoder/{operation}/encoder_{ID}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger(__name__, log_file=os.path.join(save_dir, f"{operation}_{ID}.log"))

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"Experiment ID: {ID}")
logger.info(f"Operation: {operation}")


    


for dim in range(1, 9):
    distribution = "uniform"
    if operation == "log":
        coeff_lower = 0 # for log, we need to ensure that the input is positive
    else:
        coeff_lower = -1
    coeff_upper = 1
    batch_size = 128
    num_epochs = 100
    validation_interval = 10
    lr = 0.001
    tolerance = 0.05
    
    logger.info(f"Training with {dim}x{dim} matrices")
    logger.info(f"Operation: {operation}")
    logger.info(f"Coefficient lower: {coeff_lower}")
    logger.info(f"Coefficient upper: {coeff_upper}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Validation interval: {validation_interval}")
    logger.info(f"Learning rate: {lr}")
    
    test_samples = 2**15 # 32768
    train_samples = [(2**k) + test_samples for k in range(5, 19)]
    logger.info(f"{train_samples=}, {test_samples=}")
    
    for train_sample in train_samples:
        train_sample = int(train_sample) - test_samples
        logger.info(f"Training with {train_sample} samples")
        full_dataset = NNMatrixData(n_examples=train_sample + test_samples,
                                    distribution=distribution,
                                    dim=dim,
                                    operation=operation,
                                    coeff_lower=coeff_lower,    
                                    coeff_upper=coeff_upper,
                                    only_real=True) # adding so only the real part of the matrix is used
        
        train_dataset, test_dataset = random_split(full_dataset, [train_sample, test_samples])
        
        train_dim_dir = os.path.join(save_dir, f"dim_{dim}", "train")
        test_dim_dir = os.path.join(save_dir, f"dim_{dim}", "test")
        if not os.path.exists(train_dim_dir):
            os.makedirs(train_dim_dir)
        if not os.path.exists(test_dim_dir):
            os.makedirs(test_dim_dir)
        torch.save(train_dataset, os.path.join(train_dim_dir, f'train_dataset_{str(train_sample)}.pt'))
        torch.save(test_dataset, os.path.join(test_dim_dir, f'test_dataset_{str(train_sample)}.pt'))
        
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        logger.info("=" * 50)
        
        d_model = dim*dim
        # Ensure d_model is divisible by nhead
        nhead = dim
        d_model = (dim // nhead) * nhead
        
        # model = MatrixNet(dim*dim).to(device)
        model = MatrixFunctionTransformer(d_model=d_model, nhead=dim, num_layers=2).to(device)
        criterion = FrobeniusNormLoss() if dim > 1 else nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        logger.info(f"d_model: {d_model}")
        logger.info(f"nhead: {nhead}")
        logger.info(f"dim: {dim}")
        logger.info(f"Model architecture:\n{model}")
        
        train_losses = []
        for epoch in range(num_epochs):
            # logger.info(f"Epoch {epoch+1}/{num_epochs}")
            model.train()
            for batch_idx, (x, y) in enumerate(train_loader):
                # Convert to PyTorch tensors and move to GPU
                # if dim == 1:
                x = x.view(x.size(0), dim, dim).to(device).to(torch.float64)
                y = y.view(y.size(0), dim, dim).to(device).to(torch.float64)
                
                # Encode the input using Fourier Features
                x_encoded = fourier_features(x)
                y_encoded = fourier_features(y)
                
                # Forward pass
                output = model(x)
                
                # Compute loss
                loss = criterion(output, y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
            if (epoch + 1) % validation_interval == 0:
                
                # Calculate the Frobenius norm of the difference for each matrix in the batch
                frob_norm_diff = torch.norm(output - y, p='fro', dim=(1,2))
                frob_norm_y = torch.norm(y, p='fro', dim=(1,2))
                
                # Check if the prediction is correct within the 0.05 tolerance
                correct_predictions = torch.sum((frob_norm_diff / frob_norm_y + 1e-8) <= tolerance)
                incorrect_predictions = frob_norm_diff.size(0) - correct_predictions
                
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Current Batch Loss: {loss.item():.4f}, Correct predictions: {correct_predictions}, Incorrect predictions: {incorrect_predictions}, tolerance: {tolerance}")
                
                
                
        model_save_dir = os.path.join(save_dir, f"dim_{dim}")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'{operation}_model_{str(train_sample)}.pth'))

        # Training loop complete
        logger.info("-" * 50)
        logger.info("Training loop complete with the following parameters:")
        logger.info(f"{train_sample=}, {test_samples=}, {num_epochs=}, {lr=}")
        logger.info("-" * 50)
        
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for test_idx, (x, y) in enumerate(test_loader):
                x = x.view(x.size(0), dim, dim).to(device).to(torch.float64)
                y = y.view(y.size(0), dim, dim).to(device).to(torch.float64)
                
                test_output = model(x)
                test_loss += criterion(test_output, y).item()
                
                # logger.info(f"Test: Batch {test_idx+1}/{len(test_loader)}, Current Test Batch Loss: {criterion(test_output, y).item():.4f}, Cumulative Test Loss: {test_loss:.4f}")
                
                # Calculate the Frobenius norm of the difference for each matrix in the batch
                frob_norm_diff = torch.norm(test_output - y, p='fro', dim=(1,2))
                frob_norm_y = torch.norm(y, p='fro', dim=(1,2))
                
                # Check if the prediction is correct within the 0.05 tolerance
                correct_predictions = torch.sum((frob_norm_diff / frob_norm_y + 1e-8) <= tolerance)
                incorrect_predictions = frob_norm_diff.size(0) - correct_predictions
                
            # Compare a few results
            for i in range(3):
                logger.info(f"\nExample {i+1}:")
                logger.info(f"Input A: {x[i].view(-1).cpu().numpy()}")
                logger.info(f"Actual Output: {y[i].view(-1).cpu().numpy()}")
                logger.info(f"Predicted Output: {test_output[i].view(-1).cpu().numpy()}")

        test_loss /= len(test_loader)
        logger.info(f'Test Loss: {test_loss:.4f}')
        logger.info(f"Test: Correct predictions: {correct_predictions}, Incorrect predictions: {incorrect_predictions}, tolerance: {tolerance}")
        
logger.info("-" * 50)
logger.info("Experiment complete")
logger.info(f"Experiment ID: {ID}")
logger.info(f"Operation: {operation}")
logger.info(f"Save directory: {save_dir}")
logger.info("-" * 50)
        
        
    