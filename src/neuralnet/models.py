import torch.nn as nn
import math
import torch

# Define the neural network
class MatrixNet(nn.Module):
    def __init__(self, input_size):
        super(MatrixNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    
class DeepMatrixNet(nn.Module):
    def __init__(self, input_size):
        super(DeepMatrixNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, input_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.dropout(self.relu(self.fc5(x)))
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        x = self.fc8(x)
        return x


class MatrixFunctionTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True),
            num_layers
        )
        self.encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True),
            num_layers
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = x.transpose(-2, -1)
        x = self.encoder2(x)
        return x
    
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
    

class MatrixFunctionFourierTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True),
            num_layers
        )
        self.fourier_features = FourierFeatures(d_model, d_model*2, sigma=1.0)
        self.encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True),
            num_layers
        )

    def forward(self, x):
        x = self.fourier_features(x) # [batch_size, d_model, 2*d_model]
        x = self.encoder1(x) # [batch_size, d_model, d_model]
        x = x.transpose(-2, -1) # [batch_size, d_model, d_model]    
        x = self.fourier_features(x) # [batch_size, d_model, 2*d_model]
        x = self.encoder2(x) # [batch_size, d_model, d_model]
        return x

