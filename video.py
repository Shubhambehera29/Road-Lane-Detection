import os
import cv2
import torch
import numpy as np
import time

from model import UNet
import config

# ========================================
# SETTINGS
# ========================================

INPUT_VIDEO = "input_videos/road.mp4"
OUTPUT_VIDEO = "outputs/output.mp4"
THRESHOLD = 0.3

os.makedirs("outputs", exist_ok=True)

# ========================================
# DEVICE
# ========================================

DEVICE = config.DEVICE
print(f"Using Device : {DEVICE}")

# ========================================
# LOAD MODEL
# ========================================

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        config.MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

# ========================================
# LOAD VIDEO
# ========================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise Exception("Could not open video!")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 1:
    fps = 30

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps}")
print(f"Frames     : {frame_count}")

# ========================================
# VIDEO WRITER
# ========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise Exception("VideoWriter could not be opened!")

# ========================================
# PROCESS VIDEO
# ========================================

frame_number = 0

start = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = cv2.resize(rgb, (256, 256))

    img = img.astype(np.float32) / 255.0

    img = img.transpose(2, 0, 1)

    img = torch.from_numpy(img).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pred = model(img)

        pred = torch.sigmoid(pred)

        pred = pred.squeeze().cpu().numpy()

    pred = (pred > THRESHOLD).astype(np.uint8)

    pred = cv2.resize(
        pred,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    mask = np.zeros_like(frame)

    mask[:, :, 1] = pred * 255

    overlay = cv2.addWeighted(
        frame,
        0.75,
        mask,
        0.45,
        0
    )

    cv2.putText(
        overlay,
        f"Frame : {frame_number}/{frame_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    writer.write(overlay)

end = time.time()

cap.release()
writer.release()

print("=" * 40)
print("Video processing completed.")
print(f"Frames Processed : {frame_number}")
print(f"Processing Time  : {end-start:.2f} sec")
print(f"Average FPS      : {frame_number/(end-start):.2f}")
print(f"Saved To         : {OUTPUT_VIDEO}")
print("=" * 40)