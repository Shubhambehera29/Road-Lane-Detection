import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import LaneDataset
from transforms import train_transform, val_transform
from model import UNet
from loss import BCEDiceLoss
from engine import train_one_epoch, validate

os.makedirs("models", exist_ok=True)
os.makedirs("runs", exist_ok=True)

writer = SummaryWriter("runs/lane_segmentation")

train_dataset = LaneDataset(
    config.TRAIN_IMG_DIR,
    config.TRAIN_MASK_DIR,
    train_transform
)

val_dataset = LaneDataset(
    config.VAL_IMG_DIR,
    config.VAL_MASK_DIR,
    val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS
)

model = UNet().to(config.DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

loss_fn = BCEDiceLoss()

best_loss = float("inf")

for epoch in range(config.EPOCHS):

    train_loss = train_one_epoch(
        train_loader,
        model,
        optimizer,
        loss_fn,
        config.DEVICE
    )

    val_loss, dice, iou = validate(
        val_loader,
        model,
        loss_fn,
        config.DEVICE
    )

    scheduler.step(val_loss)

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar("Dice", dice, epoch)
    writer.add_scalar("IoU", iou, epoch)

    print("-"*50)
    print(f"Epoch {epoch+1}/{config.EPOCHS}")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Validation Loss : {val_loss:.4f}")
    print(f"Dice : {dice:.4f}")
    print(f"IoU : {iou:.4f}")

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            config.MODEL_PATH
        )

        print("Best model saved.")

writer.close()