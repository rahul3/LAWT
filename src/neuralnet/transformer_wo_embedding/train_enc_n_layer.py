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
import sys

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
operation = "sin"
dim=5
# ###########################################################################################
save_dir = f"/mnt/wd_2tb/thesis_transformers/experiments/transformer_wo_embedding/{operation}/f'dim_{dim}'/encoder_{ID}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger(__name__, log_file=os.path.join(save_dir, f"{operation}_{ID}.log"))

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"Experiment ID: {ID}")
logger.info(f"Operation: {operation}")



distribution = "uniform"
if operation == "log":
    coeff_lower = 0 # for log, we need to ensure that the input is positive
else:
    coeff_lower = -1
coeff_upper = 1
batch_size = 128
num_epochs = 1000
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
train_samples = [(2**k) + test_samples for k in range(15, 16)]
logger.info(f"{train_samples=}, {test_samples=}")

# layers = [8, 16, 32, 64]
layers = 16
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
    
    train_dim_dir = os.path.join(save_dir, f"dim_{dim}", f"layers_{layers}", "train")
    test_dim_dir = os.path.join(save_dir, f"dim_{dim}", f"layers_{layers}", "test")
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
    model = MatrixFunctionTransformer(d_model=d_model, nhead=dim, num_layers=layers).to(device)
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
            
            
            
    model_save_dir = os.path.join(save_dir, f"dim_{dim}", f"layers_{layers}")
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


logger.info(f"Starting to predict values in the test_dataset")

model.eval()        
mu_lst = []
sigma_lst = []
y_main_lst = []
u_shaded_lst = []
l_shaded_lst = []
        
actuals = []
predicted = []

logger.info(f"Starting to predict values in the test_dataset")
with torch.no_grad():
    logger.info(f"{dim=}")
    for x, y in test_dataset:
        x = x.view(-1, dim, dim).to(device).to(torch.float64)
        y = y.view(-1, dim, dim).to(device).to(torch.float64)
        
        predicted.append(model(x))
        actuals.append(y)
        
    predicted = torch.cat(predicted, 0)
    actuals = torch.cat(actuals, 0)
    
    relative_error = torch.norm(predicted - actuals, p='fro', dim=(1,2)) / torch.norm(actuals, p='fro', dim=(1,2))
    logger.info(f"{relative_error.shape=}, {relative_error.mean()=}, {relative_error.std()=}")
    
    # Number of accurate samples
    accurate_samples = (relative_error <= tolerance).sum().item()
    total_samples = relative_error.size(0)
    logger.info(f"Number of accurate samples: {accurate_samples}/{total_samples}")
    
    # Calculate the mean and standard deviation of the relative error
    mean_relative_error = relative_error.mean().item()
    std_relative_error = relative_error.std().item()
    logger.info(f"Mean relative error: {mean_relative_error:.4f}")
    logger.info(f"Standard deviation of relative error: {std_relative_error:.4f}")
    
    logger.info(f"\n{predicted.shape=}\n{actuals.shape=}")
    
    np_predicted = predicted.view(-1).cpu().numpy()
    np_actuals = actuals.view(-1).cpu().numpy()

    logger.info(f"{np_predicted.dtype=},{np_actuals.dtype=}")
    
    # mu = np.mean(np.log10(np_predicted))
    y_is = np.abs(np_predicted - np_actuals)/(np.abs(np_actuals + 1e-6) ) # did this in a hurry, double check denominator
    mu = np.mean(np.log10(y_is))
    sigma = (np_predicted.shape[0] -1)**-1 * np.sum((np.log10(y_is) - mu)**2)
    
    y_main = 10**mu 
    
    u_shaded = 10**(mu - sigma)
    l_shaded = 10**(mu + sigma)
    
    logger.info(f"{mu=},{sigma=}")
    logger.info(f"{y_main=},{u_shaded=},{l_shaded=}")
    
    # x_raw = np.repeat(train_vals[idx], np_predicted.shape[0])
    # y_raw = np_actuals
    
    # logger.info(f"{x_raw.shape=}, {y_raw.shape=}")

    mu_lst.append(mu)
    sigma_lst.append(sigma)
    y_main_lst.append(y_main)
    u_shaded_lst.append(u_shaded)
    l_shaded_lst.append(l_shaded)
    
logger.info(f"{mu_lst=}, {sigma_lst=}, {y_main_lst=}, {u_shaded_lst=}, {l_shaded_lst=}")