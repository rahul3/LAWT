import datetime
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import math

from scipy.linalg import expm, hadamard, signm, cosm, sinm, logm
import numpy as np

from common import get_logger

logger = get_logger(__name__)

ID = datetime.datetime.strftime(datetime.datetime.now(),'%Y%M%d%s')

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def uniform_rand(size, low, high):
    """Generate a random tensor with uniform distribution with coefficients between low and high"""
    return torch.rand(size) * (high - low) + low

def generate_wigner_matrices(num_matrices, matrix_size):
    """Generate random Wigner matrices

    Args:
        num_matrices (int): Number of matrices to generate
        matrix_size (int): Size of the square matrices

    Returns:
        matrices (torch.Tensor): Tensor of shape (num_matrices, matrix_size, matrix_size)
    """
    matrices = torch.zeros(num_matrices, matrix_size, matrix_size)
    
    for i in range(num_matrices):
        # Randomly choose two distributions for each matrix
        dist1, dist2 = random_distribution_pair()
        
        # Generate upper triangular and diagonal elements
        upper = dist1.sample((matrix_size, matrix_size))
        diag = dist2.sample((matrix_size,))
        
        # Make the matrix symmetric
        matrix = upper.triu(1) + upper.triu(1).t() + torch.diag(diag)
        
        matrices[i] = matrix
    
    return matrices

def random_distribution_pair():
    """Randomly choose two distributions from a list of distributions

    Returns:
        dist1 (torch.distributions.Distribution): First distribution
        dist2 (torch.distributions.Distribution): Second distribution
    """
    distributions = [
        lambda: dist.Normal(0, torch.rand(1).item() + 0.5),
        lambda: dist.Uniform(-torch.sqrt(torch.tensor(3.0)), torch.sqrt(torch.tensor(3.0))),
        lambda: Rademacher(),  # Corrected Rademacher distribution
        lambda: ShiftedExponential(rate=1.0)
    ]
    
    dist1 = torch.randint(0, len(distributions), (1,)).item()
    dist2 = torch.randint(0, len(distributions), (1,)).item()
    # logger.debug(f"Chose distributions {dist1} and {dist2}")
    
    return distributions[dist1](), distributions[dist2]()

class ShiftedExponential(dist.Exponential):
    def __init__(self, rate):
        super().__init__(rate)
    
    def sample(self, sample_shape=torch.Size()):
        samples = super().sample(sample_shape)
        return samples - 1/self.rate  # Shift to make mean 0

class Rademacher(dist.Distribution):
    def __init__(self):
        super().__init__(validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        return torch.randint(0, 2, sample_shape).float() * 2 - 1

class ExperimentData(Dataset):
    "Dataset for generating various types of matrices"

    def __init__(self, n_examples, dim, distribution="gaussian", matrix_type='', 
                 coeff_upper=1, f_type="matrix", operation="square", **kwargs):
        super().__init__()
        self.n_examples = n_examples
        self.dim = dim
        self.distribution = distribution
        self.matrix_type = matrix_type
        self.scalable = False

        if self.distribution == "gaussian":
            self.data = torch.randn(n_examples, dim, dim)
            if coeff_upper is not None:
                self.data = coeff_upper / math.sqrt(3.0) * self.data
        elif self.distribution == "gaussian_positive":
            self.data = torch.abs(torch.randn(n_examples, dim, dim))
            if coeff_upper is not None:
                self.data = coeff_upper / math.sqrt(3.0) * self.data
        elif self.distribution == "uniform":
            self.data = torch.rand(n_examples, dim, dim)
            if coeff_upper is not None:
                self.data = torch.tensor(coeff_upper * (2 * self.data - 1))
        else:
            raise TypeError("Unsupported distribution")

        if matrix_type == "wigner":
            self.data = generate_wigner_matrices(n_examples, dim)
        elif matrix_type == "orthogonal":
            q, _ = torch.linalg.qr(self.data)
            self.data = q
        elif matrix_type == "toeplitz":
            first_row = torch.randn(n_examples, dim)
            first_col = torch.randn(n_examples, dim)
            self.data = torch.zeros(n_examples, dim, dim)
            for i in range(dim):
                for j in range(dim):
                    if i <= j:
                        self.data[:, i, j] = first_row[:, j-i]
                    else:
                        self.data[:, i, j] = first_col[:, i-j]
        elif matrix_type == "hankel":
            first_row = torch.randn(n_examples, dim)
            last_col = torch.randn(n_examples, dim)
            # Combine first_row and last_col (excluding duplicate last element of first_row)
            elements = torch.cat([first_row, last_col[:, 1:]], dim=1)  # shape: (n_examples, 2*dim-1)
            self.data = torch.zeros(n_examples, dim, dim)
            for i in range(dim):
                for j in range(dim):
                    self.data[:, i, j] = elements[:, i+j]
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
            self.data = torch.triu(torch.tril(self.data, diagonal=bandwidth-1), diagonal=-bandwidth+1)
        elif matrix_type == "positive_definite":
            a = torch.randn(n_examples, dim, dim)
            self.data = a @ a.transpose(-2, -1) + torch.eye(dim).unsqueeze(0) * dim
        elif matrix_type == "m_matrix":
            # M-matrix: non-positive off-diagonal, positive diagonal, with diagonally dominant
            off_diag = -torch.abs(self.data)
            # Set diagonal to sum of absolute off-diagonal values in each row plus 1
            diag_values = torch.abs(off_diag).sum(dim=-1) + 1
            self.data = off_diag + torch.diag_embed(diag_values)
            # Zero out the diagonal of off_diag part
            mask = torch.eye(dim, device=self.data.device).unsqueeze(0).expand_as(self.data)
            self.data = self.data * (1 - mask) + torch.diag_embed(diag_values) * mask
        elif matrix_type == "p_matrix":
            # P-matrix: all principal minors are positive
            # Sufficient condition: symmetric positive definite with positive entries
            a = torch.abs(torch.randn(n_examples, dim, dim))
            self.data = a @ a.transpose(-2, -1) + torch.eye(dim).unsqueeze(0) * dim
        elif matrix_type == "z_matrix":
            # Z-matrix: non-positive off-diagonal entries
            off_diag = -torch.abs(self.data)
            # Set diagonal to be positive (sum of absolute values of off-diagonals in row)
            # First zero out diagonal before summing
            mask = torch.eye(dim, device=off_diag.device).unsqueeze(0).expand_as(off_diag)
            off_diag_only = off_diag * (1 - mask)
            diag_values = torch.abs(off_diag_only).sum(dim=-1) + torch.rand(n_examples, dim)
            self.data = off_diag_only + torch.diag_embed(diag_values)
        elif matrix_type == "h_matrix":
            # H-matrix: symmetric, positive diagonal, |h_ii| > sum of |h_ij| for j != i
            a = torch.randn(n_examples, dim, dim)
            h = 0.5 * (a + a.transpose(-2, -1))
            # Make diagonally dominant
            off_diag_sum = (torch.abs(h).sum(dim=-1) - torch.abs(torch.diagonal(h, dim1=-2, dim2=-1)))
            # Set diagonal to be larger than off-diagonal sum
            diag_values = off_diag_sum + torch.abs(torch.randn(n_examples, dim)) + 1
            # Keep off-diagonal from h, replace diagonal
            mask = torch.eye(dim, device=h.device).unsqueeze(0).expand_as(h)
            self.data = h * (1 - mask) + torch.diag_embed(diag_values) * mask
        elif matrix_type == "hadamard":
            if not (dim and (not(dim & (dim - 1)))):
                raise ValueError("Dimension must be a power of 2 for Hadamard matrix")
            h = torch.tensor(hadamard(dim), dtype=torch.float32)
            self.data = h.unsqueeze(0).repeat(n_examples, 1, 1)
            # Add variation via random sign flips per row
            row_signs = (torch.randint(0, 2, (n_examples, dim, 1)) * 2 - 1).float()
            col_signs = (torch.randint(0, 2, (n_examples, 1, dim)) * 2 - 1).float()
            self.data = self.data * row_signs * col_signs
            
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
        
        
        if kwargs.get("only_real") is True:
            if self.target.is_complex():
                if len(self.target.shape) == 2: 
                    self.target = torch.real(self.target)
                elif len(self.target.shape) == 3:
                    self.target = torch.stack([torch.real(t) for t in self.target])
                else:
                    raise ValueError("Unsupported target shape")


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    

class SingleDimData(Dataset):
    "Dataset for generating various types of matrices"

    def __init__(self, n_examples, distribution="gaussian", coeff_lower=-1, coeff_upper=1, **kwargs):
        super().__init__()
        self.n_examples = n_examples
        self.distribution = distribution

        if self.distribution == "gaussian":
            self.data = torch.randn(n_examples, 1)
            if coeff_upper is not None:
                self.data = coeff_upper / math.sqrt(3.0) * self.data
        elif self.distribution == "uniform":
            self.data = uniform_rand((n_examples, 1), coeff_lower, coeff_upper)
            
        self.target = torch.exp(self.data)
            
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    
class NNData(Dataset):
    "Dataset for generating various types of matrices"

    def __init__(self, n_examples, distribution="gaussian", dim=1, coeff_lower=-1, coeff_upper=1, **kwargs):
        super().__init__()
        self.n_examples = n_examples
        self.distribution = distribution

        if self.distribution == "gaussian":
            self.data = torch.randn(n_examples, dim, dim)
            if coeff_upper is not None:
                self.data = coeff_upper / math.sqrt(3.0) * self.data
        elif self.distribution == "uniform":
            self.data = uniform_rand((n_examples, dim, dim), coeff_lower, coeff_upper)
        
        if dim == 1:
            self.target = torch.exp(self.data)
        else:
            self.target = torch.tensor(np.array([expm(m.numpy()) for m in self.data]))
            
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    

class NNMatrixData(Dataset):
    "Dataset for generating various types of matrices"

    def __init__(self, n_examples, operation="exponential", distribution="gaussian", dim=1, coeff_lower=-1, coeff_upper=1, **kwargs):
        super().__init__()
        self.n_examples = n_examples
        self.distribution = distribution

        if self.distribution == "gaussian":
            self.data = torch.randn(n_examples, dim, dim)
            if coeff_upper is not None:
                self.data = coeff_upper / math.sqrt(3.0) * self.data
        elif self.distribution == "gaussian_positive":
            self.data = torch.abs(torch.randn(n_examples, dim, dim))
            if coeff_upper is not None:
                self.data = coeff_upper / math.sqrt(3.0) * self.data
        elif self.distribution == "uniform":
            self.data = uniform_rand((n_examples, dim, dim), coeff_lower, coeff_upper)
            
            
        if operation=="exponential":
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
        elif operation=="log":
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
    
    
    
    
if __name__ == "__main__":
    
    # params
    # breakpoint()
    distribution = "uniform"
    matrix_type = "circulent"
    coeff_lower = -1
    coeff_upper = 1
    batch_size = 128
    num_epochs = 1000

    total_samples = 1000
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
    
    full_dataset = SingleDimData(n_examples=total_samples,
                                distribution=distribution,
                                coeff_lower=coeff_lower,
                                coeff_upper=coeff_upper)

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
    
    
