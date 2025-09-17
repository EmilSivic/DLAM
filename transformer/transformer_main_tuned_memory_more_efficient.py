# transformer_main_tuned.py
# ---- speed & stability helpers ----
import os
# helps avoid CUDA fragmentation on long runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math
import time
import hashlib
import random
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
import sentencepiece as spm

from transformer_model_tuned import Seq2SeqTransformerTuned  # same folder

# Speed: allow TF32 on Ampere+
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# -----------------------
# Config (edit as needed)
# -----------------------
@dataclass
class Config:
    # Paths
    DATA_CSV: str = "/content/drive/MyDrive/DLAM_Project/data_processed_new.csv/processed_recipes_multitask_ner.csv"
    SPM_MODEL: str = "/content/drive/MyDrive/DLAM_Project/data/spm_recipes.model"

    # Subset + split
    SUBSET_FRAC: float = 0.05   # 5% for fast runs; use 0.01 for very fast iteration; 1.0 = full dataset
    VAL_FRACTION: float = 0.02  # smaller val to save time
    SHUFFLE_SEED: int = 42

    # Sequence caps (very important for steps; attention is O(T^2))
    MAX_SRC_LEN: int = 64
    MAX_TGT_LEN: int = 256  # raise if VRAM allows

    # Model
    EMB_SIZE: int = 512
    NHEAD: int = 8
    FFN_HID_DIM: int = 512
    NUM_ENCODER_LAYERS: int = 4
    NUM_DECODER_LAYERS: int = 4
    DROPOUT: float = 0.1

    # Optimization
    BATCH_SIZE: int = 8          #_
