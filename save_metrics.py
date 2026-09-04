import os
import pandas as pd
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

results = pd.DataFrame({

    "Metric":[
        "Dice Score",
        "IoU Score",
        "Precision",
        "Recall",
        "F1 Score",
        "Pixel Accuracy"
    ],

    "Value":[
        dice/n,
        iou/n,
        precision/n,
        recall/n,
        f1/n,
        accuracy/n
    ]

})

os.makedirs("outputs", exist_ok=True)

results.to_csv(
    "outputs/metrics.csv",
    index=False
)

print(results)

print("\nMetrics saved to outputs/metrics.csv")
import os

output_path = os.path.abspath("outputs/metrics.csv")

os.makedirs("outputs", exist_ok=True)

results.to_csv(output_path, index=False)

print(f"Saved to: {output_path}")