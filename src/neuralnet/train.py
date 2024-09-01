import os
import argparse
import csv
import datetime
import logging

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np

from datagenerator import ExperimentData
from models import MatrixNet
from loss import FrobeniusNormLoss, LogFrobeniusNormLoss, MAPE

from graphs import training_val_loss

from common import get_logger, log_loss

# Setup seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEBUG_MODE = False 




if __name__ == "__main__":
    
    class SetDefaultAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)
            if self.dest == "dim":
                setattr(namespace, "bandwidth", min(1, values-1))
    

    parser = argparse.ArgumentParser(description="Neural Network Training")
    
    # Add arguments
    parser.add_argument("--distribution", type=str, default="gaussian", help="Distribution type (gaussian or uniform)")
    parser.add_argument("--matrix_type", type=str, default="wigner", help="Matrix type")
    parser.add_argument("--coeff_lower", type=int, default=-1, help="Lower bound for coefficients")
    parser.add_argument("--coeff_upper", type=int, default=2, help="Upper bound for coefficients")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--total_samples", type=int, default=100000, help="Total number of samples")
    parser.add_argument("--validation_interval", type=int, default=100, help="Validation interval")
    parser.add_argument("--dim", type=int, default=1, help="Dimension", action=SetDefaultAction)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--loss", type=str, default="frobenius", help="Loss function")
    parser.add_argument("--bandwidth", type=int, default=1, help="Bandwidth")
    parser.add_argument("--operation", type=str, default="square", help="Operation")
    
    args = parser.parse_args()
    
    # Access the arguments
    distribution = args.distribution
    matrix_type = args.matrix_type
    coeff_lower = args.coeff_lower
    coeff_upper = args.coeff_upper
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    total_samples = args.total_samples
    validation_interval = args.validation_interval
    dim = args.dim
    lr = args.lr
    operation = args.operation
    optional_args = {"bandwidth": 1 or args.bandwidth}
    
    if args.loss == "frobenius":
        criterion = FrobeniusNormLoss()
    elif args.loss == "log_frobenius":
        criterion = LogFrobeniusNormLoss()
    elif args.loss == "mape":
        criterion = MAPE()
    else:
        raise ValueError("Invalid loss function")
    
    
    op_to_fn = {"square": "A^2", "exponential": "e^A", "sign": "sign(A)"}
    
    
    ID = datetime.datetime.strftime(datetime.datetime.now(),'%Y%M%d%s')
    save_dir = os.path.join(os.getcwd(), "experiments", f"{operation}", f"{matrix_type}", f"dim_{dim}", ID)
    
    logger_dir = os.path.join(save_dir, "logs")
    os.makedirs(logger_dir, exist_ok=True) if not DEBUG_MODE else None
    logger_path = os.path.join(logger_dir, f"matrix_{ID}.log")
    
    logger = get_logger(__name__, level=logging.DEBUG, log_file=logger_path)
    logger.info(f"Log file saved as: {logger_path}")
    logger.info(f"Results are going to be saved at: {save_dir}")
    
    logger.info(f"Arguments: {args}")
    
    # Logging and save path setup
    logger.info(f"{ID=}")
    # Create a directory for saving results
    os.makedirs(save_dir, exist_ok=True) if not DEBUG_MODE else None

    # Create a directory for saving graphs
    graphs_dir = os.path.join(save_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True) if not DEBUG_MODE else None
    logger.info(f"Graphs are going to be saved at: {graphs_dir}")
    
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model and optimizer setup
    input_size = dim ** 2
    # Create the model and move it to GPU
    model = MatrixNet(input_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []
    
    # Create the full dataset
    full_dataset = ExperimentData(n_examples=total_samples,
                                dim=dim, 
                                distribution=distribution,
                                matrix_type=matrix_type,
                                coeff_upper=None,
                                f_type='matrix',
                                operation=operation,
                                **optional_args)

    # Define the sizes of your train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))  # 70% for training
    val_size = int(0.15 * len(full_dataset))   # 15% for validation
    test_size = len(full_dataset) - train_size - val_size  # Remaining for test

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Training loop
    training_examples = []
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            # Convert to PyTorch tensors and move to GPU
            A_tensor = x.view(x.size(0), -1).to(device)
            A_exp_tensor = y.view(y.size(0), -1).to(device)
            
            # Forward pass
            output = model(A_tensor)
            
            # Compute loss
            loss = criterion(output, A_exp_tensor)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation step
        if (epoch + 1) % validation_interval == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}: Current Batch Loss: {loss.item():.4f}")
            model.eval()
            logger.info("*" * 50)
            val_loss = 0
            with torch.no_grad():
                for val_idx, (x, y) in enumerate(val_loader):
                    A_tensor = x.view(x.size(0), -1).to(device)
                    A_exp_tensor = y.view(y.size(0), -1).to(device)
                    val_output = model(A_tensor)
                    val_loss += criterion(val_output, A_exp_tensor).item()
                    # logger.info(f"Validation: Batch {val_idx+1}/{len(val_loader)}: Current Validation Batch Loss: {criterion(val_output, A_exp_tensor).item():.4f}, Cumulative Validation Loss: {val_loss:.4f}")
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Track the number of training examples processed
            training_examples.append((epoch + 1) * len(train_loader.dataset))
            
            # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {log_loss(loss.item()):.4f}, Val Loss: {log_loss(val_loss):.4f}')
            logger.info("*" * 50)
            
    training_val_loss(training_examples, val_losses, save_path=os.path.join(graphs_dir, "training_val_loss.png"), title=f"{op_to_fn[operation]} w. dim={dim}- Val Loss vs. Train Examples")
    training_val_loss(training_examples, val_losses, save_path=os.path.join(graphs_dir, "LogLog - training_val_loss.png"), title=f"{op_to_fn[operation]} w. dim={dim}- Val Loss vs. Train Examples", graph_type="log-log")
    
    # Training loop complete
    logger.info("-" * 50)
    logger.info("Training loop complete")
    logger.info("-" * 50)
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(save_dir, 'matrix_exp_model.pth'))

    # Save loss lists
    with open(os.path.join(save_dir, 'losses.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        
        # Loop through validation losses
        for i, val_loss in enumerate(val_losses):
            epoch = (i + 1) * validation_interval # Calculate the current epoch (i starts at 0)
            train_loss = train_losses[epoch - 1]  # Get the training loss at the corresponding epoch (index is epoch-1)
            writer.writerow([epoch, train_loss, val_loss])
            
    # Training loop complete
    logger.info("-" * 50)
    logger.info("Starting to Test")
    logger.info("-" * 50)

    # Test the model
    model.eval()
    logger.info(f"Testing on {len(test_loader)} batches")
    test_loss = 0
    with torch.no_grad():
        for test_idx, (x, y) in enumerate(test_loader):
            A_tensor = x.view(x.size(0), -1).to(device)
            A_exp_tensor = y.view(y.size(0), -1).to(device)
            test_output = model(A_tensor)
            test_loss += criterion(test_output, A_exp_tensor).item()
            logger.info(f"Test: Batch {test_idx+1}/{len(test_loader)}, Current Test Batch Loss: {criterion(test_output, A_exp_tensor).item():.4f}, Cumulative Test Loss: {test_loss:.4f}")

        # Compare a few results
        for i in range(3):
            logger.info(f"\nExample {i+1}:")
            logger.info("Input A:")
            logger.info(A_tensor[i])
            logger.info(f"Actual {op_to_fn[operation]}:")
            logger.info(A_exp_tensor[i])
            logger.info(f"Predicted {op_to_fn[operation]}):")
            logger.info(test_output[i].view(dim, dim).cpu().numpy())

    test_loss /= len(test_loader)
    logger.info(f'Test Loss: {test_loss:.4f}')