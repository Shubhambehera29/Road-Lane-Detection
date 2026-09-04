import os
import cv2
import torch
from torch.utils.data import Dataset

class LaneDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]

        image = cv2.imread(
            os.path.join(self.image_dir, img_name)
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            os.path.join(self.mask_dir, img_name),
            cv2.IMREAD_GRAYSCALE
        )

        mask = (mask > 0).astype("float32")

        if self.transform:

            augment = self.transform(
                image=image,
                mask=mask
            )

            image = augment["image"]
            mask = augment["mask"]

        return image, mask.unsqueeze(0)