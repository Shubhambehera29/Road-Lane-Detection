import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import LaneDataset
from transforms import val_transform
from model import UNet
import config

os.makedirs("comparison_outputs", exist_ok=True)

dataset = LaneDataset(
    config.TEST_IMG_DIR,
    config.TEST_MASK_DIR,
    val_transform
)

model = UNet().to(config.DEVICE)

model.load_state_dict(
    torch.load(config.MODEL_PATH, map_location=config.DEVICE)
)

model.eval()

num_samples = 10

for i in range(num_samples):

    image, mask = dataset[i]

    input_image = image.unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        pred = model(input_image)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    image = image.permute(1,2,0).cpu().numpy()
    image = (image-image.min())/(image.max()-image.min()+1e-8)

    gt = mask.squeeze().cpu().numpy()
    prediction = pred.squeeze().cpu().numpy()

    overlay = image.copy()
    overlay[:,:,1] += prediction*0.6
    overlay = np.clip(overlay,0,1)

    fig, ax = plt.subplots(1,4,figsize=(18,5))

    ax[0].imshow(image)
    ax[0].set_title("Original")

    ax[1].imshow(gt,cmap="gray")
    ax[1].set_title("Ground Truth")

    ax[2].imshow(prediction,cmap="gray")
    ax[2].set_title("Prediction")

    ax[3].imshow(overlay)
    ax[3].set_title("Overlay")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    plt.savefig(f"comparison_outputs/sample_{i}.png")

    plt.close()

print("Comparison images saved.")