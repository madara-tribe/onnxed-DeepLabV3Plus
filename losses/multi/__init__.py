import torch.nn as nn
from .focal_loss import FocalLoss


class MultiClassCriterion(nn.Module):
    def __init__(self, loss_type='Lovasz', **kwargs):
        super().__init__()
        if loss_type == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'Focal':
            self.criterion = FocalLoss(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss
