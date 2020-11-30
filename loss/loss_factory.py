
from torch import nn
import torch
import numpy as np


def QuantileLoss(outputs, targets, quantile_level):
    index = (outputs <= targets).float()
    loss = quantile_level*(targets-outputs)*index+(1-quantile_level)*(outputs-targets)*(1-index)
    return loss.mean()


def MLEGLoss(outputs_mu, outputs_std, targets, lam=1e-5): #1e-5
    loss = torch.log(outputs_std**2+1e-6)/2 + (targets-outputs_mu)**2 / (2*outputs_std**2+1e-6) + lam*outputs_std**4
    return loss.mean()