import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)

        pred = pred.view(-1)

        target = target.view(-1)

        intersection = (pred * target).sum()

        dice = (2. * intersection + 1e-8) / (
            pred.sum() + target.sum() + 1e-8
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, pred, target):

        bce = self.bce(pred, target)

        dice = self.dice(pred, target)

        return bce + dice