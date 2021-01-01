import numpy as np
import torch

def nll_diagonal(target, mu, logvar):
    """Evaluate the NLL for single Gaussian with diagonal covariance matrix
    Parameters
    ----------
    target : torch.Tensor of shape [batch_size, Y_dim]
        Y labels
    mu : torch.Tensor of shape [batch_size, Y_dim]
        network prediction of the mu (mean parameter) of the BNN posterior
    logvar : torch.Tensor of shape [batch_size, Y_dim]
        network prediction of the log of the diagonal elements of the covariance matrix
    Returns
    -------
    torch.Tensor of shape
        NLL values
    """
    precision = torch.exp(-logvar)
    # Loss kernel
    loss = precision * (target - mu)**2.0 + logvar
    # Restore prefactors
    loss += np.log(2.0*np.pi)
    loss *= 0.5
    return torch.mean(torch.sum(loss, dim=-1))