import torch

# ===========================
# Dataset
# ===========================

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

BATCH_SIZE = 8

NUM_WORKERS = 2

# ===========================
# Training
# ===========================

EPOCHS = 30

LEARNING_RATE = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================
# Paths
# ===========================

TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_MASK_DIR = "dataset/train/masks"

VAL_IMG_DIR = "dataset/val/images"
VAL_MASK_DIR = "dataset/val/masks"

TEST_IMG_DIR = "dataset/test/images"
TEST_MASK_DIR = "dataset/test/masks"

MODEL_PATH = "models/best_model.pth"

OUTPUT_DIR = "outputs"
PIN_MEMORY = True

SAVE_MODEL = True

SEED = 42