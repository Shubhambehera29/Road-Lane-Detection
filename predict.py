import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import UNet
import config


DEVICE = config.DEVICE

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        config.MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


input_folder = "input_images"

output_folder = "outputs"

os.makedirs(output_folder, exist_ok=True)


for file in os.listdir(input_folder):

    image_path = os.path.join(input_folder, file)

    image = cv2.imread(image_path)

    original = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (256,256))

    image = image.astype(np.float32)/255.0

    image = image.transpose(2,0,1)

    image = torch.tensor(image).unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        prediction = model(image)

        prediction = torch.sigmoid(prediction)

        prediction = prediction.squeeze().cpu().numpy()

    prediction = (prediction>0.3).astype(np.uint8)

    prediction = cv2.resize(
        prediction,
        (original.shape[1],original.shape[0])
    )

    color_mask = np.zeros_like(original)

    color_mask[:,:,1] = prediction*255

    overlay = cv2.addWeighted(
        original,
        0.7,
        color_mask,
        0.3,
        0
    )

    save_path = os.path.join(output_folder,file)

    cv2.imwrite(save_path,overlay)

    print(f"Saved : {file}")

print("\nFinished.")