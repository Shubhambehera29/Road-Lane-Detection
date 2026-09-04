import torch


def _prepare(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    pred = pred.view(-1)
    target = target.view(-1)

    return pred, target


def dice_score(pred, target):

    pred, target = _prepare(pred, target)

    intersection = (pred * target).sum()

    return ((2 * intersection + 1e-8) /
            (pred.sum() + target.sum() + 1e-8)).item()


def iou_score(pred, target):

    pred, target = _prepare(pred, target)

    intersection = (pred * target).sum()

    union = pred.sum() + target.sum() - intersection

    return ((intersection + 1e-8) /
            (union + 1e-8)).item()


def precision_score(pred, target):

    pred, target = _prepare(pred, target)

    tp = (pred * target).sum()

    fp = (pred * (1 - target)).sum()

    return ((tp + 1e-8) /
            (tp + fp + 1e-8)).item()


def recall_score(pred, target):

    pred, target = _prepare(pred, target)

    tp = (pred * target).sum()

    fn = ((1 - pred) * target).sum()

    return ((tp + 1e-8) /
            (tp + fn + 1e-8)).item()


def f1_score(pred, target):

    p = precision_score(pred, target)

    r = recall_score(pred, target)

    return (2 * p * r) / (p + r + 1e-8)


def pixel_accuracy(pred, target):

    pred, target = _prepare(pred, target)

    correct = (pred == target).sum()

    return (correct / len(pred)).item()