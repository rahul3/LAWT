import os
import argparse
import csv
import datetime
import logging

import warnings
warnings.filterwarnings('ignore', message='logm result may be inaccurate')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagenerator import NNData, NNMatrixData
from models import MatrixNet, DeepMatrixNet
from loss import FrobeniusNormLoss, LogFrobeniusNormLoss, MAPE

from graphs import training_val_loss

from common import get_logger, log_loss

torch.set_default_dtype(torch.float64)

# Setup seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# operations_to_run = ["sin", "sign", "cos", "log", "exp"]
operations_to_run = ["log"]

for operation in operations_to_run:
    # Generate ID based on current datetime
    ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # ###########################################################################################
    # operation = "sin"
    # ###########################################################################################
    save_dir = f"/home/rahulpadmanabhan/Development/ws1/experiments/deepnetwork/{operation}/deepmatrixnet_{ID}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_logger(__name__, log_file=os.path.join(save_dir, f"{operation}_{ID}.log"))

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment ID: {ID}")



    logger.info(f"Experiment ID: {ID}")
    logger.info(f"Operation: {operation}")

    for dim in range(1, 8):
        distribution = "gaussian" if operation != "log" else "gaussian_positive"
        coeff_lower = -1
        coeff_upper = 1
        batch_size = 128
        num_epochs = 100
        validation_interval = 10
        lr = 0.001
        
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
                                        coeff_upper=coeff_upper)
            
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
            
            
            model = MatrixNet(dim*dim).to(device)
            # model = DeepMatrixNet(dim*dim).to(device)
            criterion = FrobeniusNormLoss().to(torch.float64) if dim > 1 else nn.MSELoss().to(torch.float64)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            logger.info(f"Model architecture:\n{model}")
            
            train_losses = []
            for epoch in range(num_epochs):
                # logger.info(f"Epoch {epoch+1}/{num_epochs}")
                model.train()
                for batch_idx, (x, y) in enumerate(train_loader):
                    # Convert to PyTorch tensors and move to GPU
                    if dim == 1:
                        x = x.view(-1, 1).to(device).to(torch.float64)
                        y = y.view(-1, 1).to(device).to(torch.float64)
                    else:
                        x = x.view(x.size(0), -1, dim*dim).to(device).to(torch.float64)  # Reshape to (batch, time, channels)
                        y = y.view(y.size(0), -1, dim*dim).to(device).to(torch.float64)  # Reshape to (batch, time, channels)
                    
                        
                    output = model(x)
                    # Compute loss
                    loss = criterion(output, y)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    
                if (epoch + 1) % validation_interval == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Current Batch Loss: {loss.item():.4f}")

                    
                    
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
            test_loss_real = 0
            test_loss_imag = 0
            with torch.no_grad():
                for test_idx, (x, y) in enumerate(test_loader):
                    if dim == 1:
                        x = x.view(-1, 1).to(device).to(torch.float64)
                        y = y.view(-1, 1).to(device).to(torch.float64)
                    else:
                        x = x.view(x.size(0), -1, dim*dim).to(device).to(torch.float64)  # Reshape to (batch, time, channels)
                        y = y.view(y.size(0), -1, dim*dim).to(device).to(torch.float64)  # Reshape to (batch, time, channels)
                    
                    output = model(x)
                    test_loss += criterion(output, y).item()
                    # logger.info(f"Test: Batch {test_idx+1}/{len(test_loader)}, Current Test Batch Loss: {criterion(test_output, y).item():.4f}, Cumulative Test Loss: {test_loss:.4f}")

                # Compare a few results
                for i in range(3):
                    logger.info(f"\nExample {i+1}:")
                    logger.info(f"Input A: {x[i].view(-1).cpu().numpy()}")
                    logger.info(f"Actual Output: {y[i].view(-1).cpu().numpy()}")
                    logger.info(f"Predicted Output: {output[i].view(-1).cpu().numpy()}")

            test_loss /= len(test_loader)
            logger.info(f'Test Loss: {test_loss:.4f}')

            with torch.no_grad():
                # create 10000 evaluation samples
                evaluation_samples = 10000
                evaluation_dataset = NNMatrixData(n_examples=evaluation_samples,
                                                  distribution=distribution,
                                                  dim=dim,
                                                  operation=operation,
                                                  coeff_lower=coeff_lower,
                                                  coeff_upper=coeff_upper)
                evaluation_loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
                evaluations_correct = dict()
                correct_predictions = dict()
                tols = [0.05, 0.02, 0.01, 0.005]
                # initialize the dictionaries
                for tol in tols:
                    evaluations_correct[tol] = 0
                    correct_predictions[tol] = 0
                for evaluation_idx, (x, y) in enumerate(evaluation_loader):
                    if dim == 1:
                        x = x.view(-1, 1).to(device).to(torch.float64)
                        y = y.view(-1, 1).to(device).to(torch.float64)
                    else:
                        x = x.view(x.size(0), -1, dim*dim).to(device).to(torch.float64)
                        y = y.view(y.size(0), -1, dim*dim).to(device).to(torch.float64)
                    output = model(x)
                    error = torch.abs(output - y)
                    for tol in tols:
                        correct_predictions[tol] = (error / (torch.abs(y) + 1e-12)) <= tol
                        evaluations_correct[tol] += correct_predictions[tol].sum().item()
                        logger.info(f"Evaluation {evaluation_idx+1}/{evaluation_samples}, Correct predictions: {correct_predictions[tol].sum().item()}/{x.size(0)}, Tolerance: {tol}")
                for tol in tols:
                    logger.info(f"Total correct predictions: {evaluations_correct[tol]}/{evaluation_samples}, Tolerance: {tol}")
                    logger.info(f"Accuracy: {evaluations_correct[tol]/evaluation_samples:.4f}, Tolerance: {tol}")

    logger.info("-" * 50)
    logger.info("Experiment complete")
    logger.info(f"Experiment ID: {ID}")
    logger.info(f"Operation: {operation}")
    logger.info(f"Save directory: {save_dir}")
    logger.info("-" * 50)
            
        
