import cv2
import torch
import numpy as np

from model import UNet
import config

# ------------------------
# Load Model
# ------------------------

device = config.DEVICE

model = UNet().to(device)

model.load_state_dict(
    torch.load(config.MODEL_PATH, map_location=device)
)

model.eval()

# ------------------------
# Open Webcam
# ------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    original = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(rgb, (256,256))

    tensor = resized.astype(np.float32)/255.0

    tensor = tensor.transpose(2,0,1)

    tensor = torch.tensor(tensor).unsqueeze(0).float().to(device)

    with torch.no_grad():

        pred = model(tensor)

        pred = torch.sigmoid(pred)

        pred = pred.squeeze().cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8)

    pred = cv2.resize(
        pred,
        (original.shape[1], original.shape[0])
    )

    mask = np.zeros_like(original)

    mask[:,:,1] = pred * 255

    overlay = cv2.addWeighted(
        original,
        0.7,
        mask,
        0.3,
        0
    )

    cv2.imshow("Lane Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()