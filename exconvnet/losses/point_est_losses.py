# implements losses for point estimation
import torch
import numpy as np

__all__ = ['MSELoss']

# we want to use MSELoss here
MSELoss = torch.nn.MSELoss
