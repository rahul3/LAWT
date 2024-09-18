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
import pandas as pd
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

operations_dct = {
    "log": "20240915123554",
}

for operation in operations_to_run:
    # Generate ID based on current datetime
    # ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # ###########################################################################################
    # operation = "sin"
    # ###########################################################################################
    save_dir = f"/mnt/wd_2tb/thesis_transformers/experiments/deepnetwork/{operation}/deepmatrixnet_{operations_dct[operation]}"
    log_file_path = os.path.join(save_dir, f"{operation}_{operations_dct[operation]}_evaluation_logdim1.log")
    
    logger = get_logger(__name__, log_file=log_file_path)
    

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info(f"Experiment ID: {operations_dct[operation]}")
    logger.info(f"Operation: {operation}")


    for dim in range(1, 2):
        test_samples = 2**15 # 32768
        train_samples = [(2**k) for k in range(5, 19)]
        logger.info(f"{train_samples=}, {test_samples=}")
        
        mu_lst = []
        sigma_lst = []
        y_main_lst = []
        u_shaded_lst = []
        l_shaded_lst = []
        
        for train_sample in train_samples:
            # train_dim_dir = os.path.join(save_dir, f"dim_{dim}", "train")
            test_dim_dir = os.path.join(save_dir, f"dim_{dim}", "test")
            
            # train_dataset = torch.load(os.path.join(train_dim_dir, f'train_dataset_{str(train_sample)}.pt'))
            test_dataset = torch.load(os.path.join(test_dim_dir, f'test_dataset_{str(train_sample)}.pt'))
            
            # train_dataset = train_dataset
            test_dataset = test_dataset
            
            model = DeepMatrixNet(dim*dim).to(device)
            model.load_state_dict(torch.load(os.path.join(save_dir, f"dim_{dim}", f'{operation}_model_{str(train_sample)}.pth')))
            model.eval()  # Set the model to evaluation mode
            # breakpoint()  # Commented out for now, but you can uncomment if needed for debugging
            
            
            criterion = FrobeniusNormLoss().to(torch.float64) if dim > 1 else nn.MSELoss().to(torch.float64)
            
            logger.info(f"Model architecture:")
            logger.info(model)
            
            logger.info(f"{len(test_dataset)=}")

            actuals = []
            predicted = []

            logger.info(f"Starting to predict values in the test_dataset")
            with torch.no_grad():
                logger.info(f"{dim=}")
                for x, y in test_dataset:
                    if dim == 1:
                        predicted.append(model(x.view(-1, 1).to(device).to(torch.float64)))
                        actuals.append(y.view(-1,1).to(device).to(torch.float64))
                    else:
                        predicted.append(model(x.view(1, dim*dim).to(device).to(torch.float64)))
                        actuals.append(y.view(1, dim*dim).to(device).to(torch.float64))
                        
                predicted = torch.cat(predicted, 0)
                actuals = torch.cat(actuals, 0)
                
                logger.info(f"\n{predicted.shape=}\n{actuals.shape=}")
                
                np_predicted = predicted.view(-1).cpu().numpy()
                np_actuals = actuals.view(-1).cpu().numpy()

                logger.info(f"{np_predicted.dtype=},{np_actuals.dtype=}")
                
                # mu = np.mean(np.log10(np_predicted))
                y_is = np.abs(np_predicted - np_actuals)/(np.abs(np_actuals) + 1e-10)
                mu = np.mean(np.log10(y_is))
                sigma = (np_predicted.shape[0] -1)**-1 * np.sum((np.log10(y_is) - mu)**2)
                
                y_main = 10**mu 
                
                u_shaded = 10**(mu - sigma)
                l_shaded = 10**(mu + sigma)
                
                logger.info(f"{mu=},{sigma=}")
                logger.info(f"{y_main=},{u_shaded=},{l_shaded=}")
                

                mu_lst.append(mu)
                sigma_lst.append(sigma)
                y_main_lst.append(y_main)
                u_shaded_lst.append(u_shaded)
                l_shaded_lst.append(l_shaded)
                
                
                # Create a dictionary to store the data
                data = {
                    'operation': operation,
                    'dim': dim,
                    'train_sample': train_sample,
                    'test_sample': test_samples,
                    'mu': mu_lst[-1],
                    'sigma': sigma_lst[-1],
                    'y_main': y_main_lst[-1],
                    'u_shaded': u_shaded_lst[-1],
                    'l_shaded': l_shaded_lst[-1]
                }

                # Create a DataFrame from the dictionary
                results_path = os.path.join("/mnt/wd_2tb/thesis_transformers/experiments/deepnetwork/csv_results", f'results_{operation}_dim1.csv')
                df = pd.DataFrame([data])

                # If the DataFrame doesn't exist, create it. Otherwise, append to it.
                try:
                    existing_df = pd.read_csv(results_path)
                    updated_df = pd.concat([existing_df, df], ignore_index=True)
                except FileNotFoundError:
                    updated_df = df

                # Save the updated DataFrame
                updated_df.to_csv(results_path, index=False)

                logger.info(f"Recorded data for dim={dim}, train_sample={train_sample}")


                