import torch
from torch.utils.data import DataLoader

import config

from dataset import LaneDataset
from transforms import val_transform
from model import UNet

from metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
    f1_score,
    pixel_accuracy
)

dataset = LaneDataset(
    config.TEST_IMG_DIR,
    config.TEST_MASK_DIR,
    val_transform
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

model = UNet().to(config.DEVICE)

model.load_state_dict(
    torch.load(
        config.MODEL_PATH,
        map_location=config.DEVICE
    )
)

model.eval()

dice = 0
iou = 0
precision = 0
recall = 0
f1 = 0
accuracy = 0

with torch.no_grad():

    for images, masks in loader:

        images = images.to(config.DEVICE)

        masks = masks.to(config.DEVICE)

        outputs = model(images)

        dice += dice_score(outputs, masks)

        iou += iou_score(outputs, masks)

        precision += precision_score(outputs, masks)

        recall += recall_score(outputs, masks)

        f1 += f1_score(outputs, masks)

        accuracy += pixel_accuracy(outputs, masks)

n = len(loader)

print("="*50)

print(f"Dice Score      : {dice/n:.4f}")

print(f"IoU Score       : {iou/n:.4f}")

print(f"Precision       : {precision/n:.4f}")

print(f"Recall          : {recall/n:.4f}")

print(f"F1 Score        : {f1/n:.4f}")

print(f"Pixel Accuracy  : {accuracy/n:.4f}")

print("="*50)