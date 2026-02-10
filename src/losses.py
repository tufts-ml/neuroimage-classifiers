import copy
# PyTorch
import torch
# Importing our custom module(s)
import utils

class ERMLoss(torch.nn.Module):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):
        
        nll = self.criterion(logits, labels)
        
        return {'loss': nll, 'nll': nll}
    
class L1Loss(torch.nn.Module):
    def __init__(self, alpha, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):

        params = kwargs["params"]
        
        nll = self.criterion(logits, labels)
        penalty = (self.alpha/2) * torch.abs(params).sum()
        
        return {'loss': nll + penalty, 'nll': nll}
    
class L2Loss(torch.nn.Module):
    def __init__(self, alpha, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion

    def forward(self, logits, labels, **kwargs):

        params = kwargs["params"]
        
        nll = self.criterion(logits, labels)
        penalty = (self.alpha/2) * (params**2).sum()
        
        return {'loss': nll + penalty, 'nll': nll}
