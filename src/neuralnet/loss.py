import torch
import torch.nn as nn

class FrobeniusNormLoss(nn.Module):
    def __init__(self):
        super(FrobeniusNormLoss, self).__init__()

    def forward(self, pred, target):
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