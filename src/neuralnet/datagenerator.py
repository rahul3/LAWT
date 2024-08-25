import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import math

from scipy.linalg import expm, hadamard, signm
import numpy as np

from common import get_logger

logger = get_logger(__name__)

ID = datetime.datetime.strftime(datetime.datetime.now(),'%Y%M%d%s')

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class ExperimentData(Dataset):
    "Dataset for generating various types of matrices"

    def __init__(self, n_examples, dim, distribution="gaussian", matrix_type='', coeff_upper=1, f_type="matrix", operation="square", 
                 wigner_diag_mean=1, wigner_diag_std=2**0.5, **kwargs):
        super().__init__()
        self.n_examples = n_examples
        self.dim = dim
        self.distribution = distribution
        self.matrix_type = matrix_type

        if self.distribution == "gaussian":
            self.data = torch.randn(n_examples, dim, dim)
            if coeff_upper is not None:
                self.data = coeff_upper / math.sqrt(3.0) * self.data
        elif self.distribution == "uniform":
            self.data = torch.rand(n_examples, dim, dim)
            if coeff_upper is not None:
                self.data = torch.tensor(coeff_upper * (2 * self.data - 1))
        else:
            raise TypeError("Unsupported distribution")

        if matrix_type == "wigner":
            A_triu = torch.triu(self.data, 1)
            self.data = A_triu + A_triu.transpose(-2, -1)
            diagonal = torch.diag_embed(torch.randn(n_examples, dim) * wigner_diag_std + wigner_diag_mean)
            self.data = self.data + diagonal
        elif matrix_type == "orthogonal":
            q, _ = torch.linalg.qr(self.data)
            self.data = q
        elif matrix_type == "toeplitz":
            first_row = torch.randn(n_examples, dim)
            self.data = torch.zeros(n_examples, dim, dim)
            for i in range(dim):
                self.data[:, i, i:] = first_row[:, :dim-i]
                self.data[:, i:, i] = first_row[:, :dim-i]
        elif matrix_type == "hankel":
            first_row = torch.randn(n_examples, dim)
            last_col = torch.randn(n_examples, dim-1)
            self.data = torch.zeros(n_examples, dim, dim)
            for i in range(dim):
                self.data[:, i, i:] = first_row[:, :dim-i]
                self.data[:, i:, i] = torch.cat([first_row[:, i:], last_col[:, :i]], dim=1)
        elif matrix_type == "stochastic":
            self.data = torch.abs(self.data)
            self.data = self.data / self.data.sum(dim=-1, keepdim=True)
        elif matrix_type == "circulant":
            first_row = torch.randn(n_examples, dim)
            self.data = torch.zeros(n_examples, dim, dim)
            for i in range(dim):
                self.data[:, i] = torch.roll(first_row, shifts=i, dims=1)
        elif matrix_type == "band":
            bandwidth = kwargs.get("bandwidth") or min(3, dim-1)  # Adjust bandwidth as needed
            self.data = torch.triu(torch.tril(self.data, diagonal=bandwidth), diagonal=-bandwidth)
        elif matrix_type == "positive_definite":
            a = torch.randn(n_examples, dim, dim)
            self.data = a @ a.transpose(-2, -1) + torch.eye(dim).unsqueeze(0) * dim
        elif matrix_type == "m_matrix":
            self.data = torch.abs(self.data)
            self.data = self.data.max(dim=-1, keepdim=True)[0] * torch.eye(dim).unsqueeze(0) - self.data
        elif matrix_type == "p_matrix":
            self.data = torch.abs(self.data) + torch.eye(dim).unsqueeze(0) * dim
        elif matrix_type == "z_matrix":
            self.data = -torch.abs(self.data)
            self.data = self.data + torch.diag_embed(torch.abs(self.data).sum(dim=-1))
        elif matrix_type == "h_matrix":
            a = torch.randn(n_examples, dim, dim)
            h = 0.5 * (a + a.transpose(-2, -1))
            m = torch.abs(h).sum(dim=-1, keepdim=True) - torch.abs(torch.diag_embed(torch.diagonal(h, dim1=-2, dim2=-1)))
            self.data = h + m * torch.eye(dim).unsqueeze(0)
        elif matrix_type == "hadamard":
            if not (dim and (not(dim & (dim - 1)))):
                raise ValueError("Dimension must be a power of 2 for Hadamard matrix")
            h = torch.tensor(hadamard(dim), dtype=torch.float32)
            self.data = h.unsqueeze(0).repeat(n_examples, 1, 1)
            
        if f_type == "scalar" and operation == "exponential":
            self.target = torch.exp(self.data)
        elif f_type == "matrix" and operation == "exponential":
            self.target = torch.tensor(np.array([expm(m.numpy()) for m in self.data]))
        elif f_type == "scalar" and operation == "square":
            self.target = self.data ** 2
        elif f_type == "matrix" and operation == "square":
            self.target = self.data @ self.data
        elif f_type == "scalar" and operation == "sign":
            self.target = torch.sign(self.data)
        elif f_type == "matrix" and operation == "sign":
            self.target = torch.tensor(np.array([signm(m.numpy()) for m in self.data]))
        else:
            raise TypeError("Unsupported function type")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    
if __name__ == "__main__":
    
    # params
    distribution = "gaussian" # or uniform
    matrix_type = "band"
    coeff_lower = -1
    coeff_upper = 2
    batch_size = 256
    num_epochs = 1000

    total_samples = 100000
    validation_interval = 100

    dim = 5

    lr = 1e-3 # learning rate
    
    optional_args = {"bandwidth": 1}
    
    # Create the full dataset
    full_dataset = ExperimentData(n_examples=total_samples,
                                dim=dim, 
                                distribution=distribution,
                                matrix_type=matrix_type,
                                coeff_upper=None,
                                f_type='matrix',
                                operation="square",
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
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"{data.shape=}, {target.shape=}")
        print(f"data:\n{data[0]}\n\ntarget:\n{target[0]}")
        breakpoint()
        break
    
    