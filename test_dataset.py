import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import LaneDataset
from transforms import train_transform

import config


dataset = LaneDataset(

    image_dir=config.TRAIN_IMG_DIR,

    mask_dir=config.TRAIN_MASK_DIR,

    transform=train_transform

)

loader = DataLoader(

    dataset,

    batch_size=4,

    shuffle=True

)


images, masks = next(iter(loader))


print("Image Shape :", images.shape)

print("Mask Shape :", masks.shape)


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.imshow(images[0].permute(1,2,0))

plt.title("Image")

plt.axis("off")

plt.subplot(1,2,2)

plt.imshow(masks[0].squeeze(), cmap="gray")

plt.title("Mask")

plt.axis("off")

plt.show()