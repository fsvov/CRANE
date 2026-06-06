# config.py
import os

# ============================================================
# Paths — update RAW to your local CMU-MOSI raw data directory
# ============================================================
RAW = '../../Raw'  # <-- configure this path

TRANSCRIPT_DIR = f'{RAW}/Transcript/Segmented'
AUDIO_DIR = f'{RAW}/Audio/WAV_16000/Segmented'
VIDEO_DIR = f'{RAW}/Video/Segmented'
LABEL_PATH = "./mosi_data.pkl"

# ============================================
# Model Hyperparameters
# ============================================
TEXT_DIM = 768      # BERT CLS (raw, no PCA)
VISUAL_DIM = 35
ACOUSTIC_DIM = 74
HIDDEN_DIM = 300
DENSE_DIM = 100
NUM_CLASSES = 2
DROPOUT = 0.5
DROPOUT_GRU = 0.5
LR = 1e-3
EPOCHS = 50
MAX_UTT = 63
BATCH_SIZE = 8

# ============================================
# Training
# ============================================
SEED = 42
TRAIN_SPLIT = 0.65
VAL_SPLIT = 0.10

# ============================================
# Save directory
# ============================================
SAVE_DIR = "./saved"