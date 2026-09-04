import cv2
import matplotlib.pyplot as plt
import os

image_path = os.path.join("dataset/images", os.listdir("dataset/images")[0])
mask_path = os.path.join("dataset/masks", os.listdir("dataset/masks")[0])

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, 0)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Road Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Lane Mask")
plt.axis("off")

plt.show()