import torch
import torch.nn as nn

class FrobeniusNormLoss(nn.Module):
    def __init__(self):
        super(FrobeniusNormLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.shape == target.shape, "Prediction and target must have the same shape"
        return torch.norm(pred - target, p='fro')
    
class MAPE(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = torch.abs((pred - target) / (target + self.epsilon))
        return 100. * torch.mean(diff)
    
class LogFrobeniusNormLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(LogFrobeniusNormLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        frob_norm = torch.norm(pred - target, p='fro')
        return torch.log(frob_norm + self.eps)
    
class RelativeError(nn.Module):
    def __init__(self):
        super(RelativeError, self).__init__()

    def forward(self, pred, target):
        assert pred.shape == target.shape, "Prediction and target must have the same shape"
        error = torch.norm(pred - target, p='fro')
        norm_true = torch.norm(target, p='fro')
        # Avoid division by zero
        return error / (norm_true + 1e-7)  # Adding a small epsilon for stability
    
class SpectralNormError(nn.Module):
    def __init__(self):
        super(SpectralNormError, self).__init__()

    def forward(self, pred, target):
        assert pred.shape == target.shape, "Prediction and target must have the same shape"
        diff = pred - target
        # Compute the largest singular value as an approximation of the 2-norm
        _, S, _ = torch.svd(diff)
        return S.max()
    
class MeanAbsoluteError(nn.Module):
    def __init__(self):
        super(MeanAbsoluteError, self).__init__()

    def forward(self, pred, target):
        assert pred.shape == target.shape, "Prediction and target must have the same shape"
        return torch.mean(torch.abs(pred - target))
    

def frobenius_norm_error(target, actual):
    return torch.norm(target - actual, p='fro')


def relative_error(target, actual):
    error = torch.norm(target - actual, p='fro')
    norm_true = torch.norm(target, p='fro')
    return error / norm_true if norm_true != 0 else torch.tensor(float('inf'))

def spectral_norm_error(target, actual):
    # PyTorch doesn't directly provide a function for 2-norm of the difference,
    # but we can compute it using the largest singular value of the difference
    diff = target - actual
    # Here we use torch.svd to get singular values, then take the max
    _, S, _ = torch.svd(diff)
    return S.max()

def mae_error(target, actual):
    return torch.mean(torch.abs(target - actual))

def mse_error(target, actual):
    return torch.mean((target - actual) ** 2)
