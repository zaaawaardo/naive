import os
import random
import numpy as np
import torch

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training.1600000.processed.noemoticon.csv')
SAMPLE_SIZE = 40000  # balanced sample for reasonable compute on CPU
MAX_LEN = 32
VOCAB_SIZE = 10000
EMBED_DIM = 100
BATCH_SIZE = 256
EPOCHS_RNN = 6
EPOCHS_TRANSFORMER = 1
TEST_SIZE = 0.15
VAL_SIZE = 0.15

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)
