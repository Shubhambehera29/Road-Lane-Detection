import torch

from dataset import LaneDataset
from transforms import train_transform
from model import UNet
from loss import BCEDiceLoss

import config

dataset = LaneDataset(
    config.TRAIN_IMG_DIR,
    config.TRAIN_MASK_DIR,
    train_transform
)

image, mask = dataset[0]

image = image.unsqueeze(0).to(config.DEVICE)
mask = mask.unsqueeze(0).to(config.DEVICE)

model = UNet().to(config.DEVICE)

loss_fn = BCEDiceLoss()

output = model(image)

loss = loss_fn(output, mask)

print("Forward Pass Successful")
print("Output Shape :", output.shape)
print("Loss :", loss.item())