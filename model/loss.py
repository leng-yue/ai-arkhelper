import torch
from torch import nn
from torch.nn import functional as F


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    eps = 1e-7

    pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, regr, gt_regr, mask):
        num = mask.float().sum() * 2
        regr = regr[mask == 1]
        gt_regr = gt_regr[mask == 1]
        regr_loss = F.l1_loss(
            regr, gt_regr, reduction='sum'
        )
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, gt):
        return _neg_loss(pred, gt)
