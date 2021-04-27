import torch.nn as nn

__all__ = ['MSELoss']


class MSELoss:
    def __init__(self):
        self.local_mse = nn.MSELoss(reduction='mean')
        self.global_mse = nn.MSELoss(reduction='mean')

    def __call__(self, pred, target):
        pred_local, pred_global = pred
        target_local, target_global = target
        mse = self.local_mse(pred_local, target_local)
        mse += self.global_mse(pred_global, target_global)
        return mse