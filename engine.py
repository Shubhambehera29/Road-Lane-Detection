import torch
from tqdm import tqdm

from metrics import dice_score, iou_score


def train_one_epoch(loader, model, optimizer, loss_fn, device):

    model.train()

    total_loss = 0

    for images, masks in tqdm(loader):

        images = images.to(device)

        masks = masks.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(loader, model, loss_fn, device):

    model.eval()

    total_loss = 0

    dice = 0

    iou = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)

            masks = masks.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, masks)

            total_loss += loss.item()

            dice += dice_score(outputs, masks)

            iou += iou_score(outputs, masks)

    return (

        total_loss / len(loader),

        dice / len(loader),

        iou / len(loader)

    )