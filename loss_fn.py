import torch
import torch.nn as nn
from torch.autograd import Variable

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class MaskedberHuLoss(nn.Module):
    def __init__(self):
        super(MaskedberHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff=target-pred
        diff=diff[valid_mask]
        c = 0.2 * diff.abs().max()
        l1=diff.abs().mean()
        l2=(diff ** 2).mean()
        if l1 > c:
            return (l2+c**2)/(2*c)
        else:
            return l1