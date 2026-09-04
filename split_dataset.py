import os
import shutil
from sklearn.model_selection import train_test_split

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

# Create folders
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder, "masks"), exist_ok=True)

# Get image names
images = sorted(os.listdir(IMAGE_DIR))

# Split
train_imgs, temp_imgs = train_test_split(
    images,
    test_size=0.2,
    random_state=42
)

val_imgs, test_imgs = train_test_split(
    temp_imgs,
    test_size=0.5,
    random_state=42
)

def copy_files(file_list, split_dir):

    for file in file_list:

        shutil.copy(
            os.path.join(IMAGE_DIR, file),
            os.path.join(split_dir, "images", file)
        )

        shutil.copy(
            os.path.join(MASK_DIR, file),
            os.path.join(split_dir, "masks", file)
        )

copy_files(train_imgs, TRAIN_DIR)
copy_files(val_imgs, VAL_DIR)
copy_files(test_imgs, TEST_DIR)

print("Done!")

print(f"Train : {len(train_imgs)}")

print(f"Validation : {len(val_imgs)}")

print(f"Test : {len(test_imgs)}")