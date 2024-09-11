import torch
import torch.nn as nn
from scipy.linalg import expm
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import random_split

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

    def __init__(self, n_examples, operation="exponential", distribution="gaussian", dim=1, coeff_lower=-1, coeff_upper=1, **kwargs):
        super().__init__()
        self.n_examples = n_examples
        self.distribution = distribution

        if self.distribution == "gaussian":
            self.data = torch.randn(n_examples, dim, dim)
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

    def forward(self, x):
        # x shape: [batch_size, 5, 5]
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [batch_size, 25]
        ff_x = self.encoder(x_flat)  # [batch_size, 200]
        ff_x = ff_x.unsqueeze(1)  # [batch_size, 1, 200]
        tf_x = self.transformer_encoder(ff_x)  # [batch_size, 1, 200]
        tf_x = tf_x.squeeze(1)  # [batch_size, 200]
        output = self.final_linear(tf_x)  # [batch_size, 25]
        return output.view(batch_size, 5, 5)  # [batch_size, 5, 5]
    

# Usage
x = torch.randn(10, 5, 5)
y = torch.stack([torch.from_numpy(expm(x[i].numpy())) for i in range(x.shape[0])])
model = MatrixApproximator(input_dim=25, num_fourier_features=100, d_model=200, nhead=8, num_layers=16)
output = model(y)
print(output.shape)  # Should be torch.Size([10, 5, 5])


# Set up training parameters
num_epochs = 20
batch_size = 32
learning_rate = 0.001

# Create dataset and dataloader
dataset = NNMatrixData(n_examples=10000, distribution="uniform", dim=5, operation="exponential", coeff_lower=-1, coeff_upper=1, only_real=True)
train_dataset, test_dataset = random_split(dataset, [8000, 2000])   
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)


# Initialize model, loss function, and optimizer
model = MatrixApproximator(input_dim=25, num_fourier_features=100, d_model=200, nhead=8, num_layers=2).to(device)
criterion = nn.MSELoss()
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

print("Training completed.")

# Test the model
model.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        predicted = model(batch_x)
        test_loss = criterion(predicted, batch_y)
        total_loss += test_loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss:.4f}') 
    print(f'Test Loss: {test_loss.item():.4f}')
    
# Calculate relative error
relative_error = torch.norm(predicted - batch_y, dim=(1, 2)) / torch.norm(batch_y, dim=(1, 2))
avg_relative_error = relative_error.mean().item()
print(f'Average Relative Error: {avg_relative_error:.4f}')

# Calculate element-wise relative error
element_wise_relative_error = torch.abs(predicted - batch_y) / (torch.abs(batch_y) + 1e-8)  # Add small constant to avoid division by zero
avg_element_wise_relative_error = element_wise_relative_error.mean().item()
print(f'Average Element-wise Relative Error: {avg_element_wise_relative_error:.4f}')

# Calculate percentage of predictions within 5% relative error
within_tolerance = (relative_error <= 0.05).float().mean().item()
print(f'Percentage of predictions within 5% relative error: {within_tolerance*100:.2f}%')

# Save the model
# torch.save(model.state_dict(), 'matrix_approximator_model.pth')
print("Model saved.")


