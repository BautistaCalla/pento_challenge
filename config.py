import torch

# Set random seed for reproducibility
torch.manual_seed(1234)

# Define constants
DATA_DIR = 'dogs'
NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 70
LEARNING_RATE = 0.001
MOMENTUM = 0.9