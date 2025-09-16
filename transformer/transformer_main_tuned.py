import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# Make sure we can import dataset.py from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_dataloaders
from transformer_model_tuned import Seq2SeqTransformer  # tuned model class

# === Hyperparameters ===
EMB_SIZE = 512
HIDDEN_DIM = 2048
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NHEAD = 8
DROPOUT = 0.2
LR = 5e-4
BATCH_SIZE = 128
EPOCHS = 15
WARMUP_STEPS = 4000

# === Data ===
train_loader, val_loader, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = get_dataloaders(batch_size=BATCH_SIZE)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqTransformer(
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    emb_size=EMB_SIZE,
    nhead=NHEAD,
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    dim_feedforward=HIDDEN_DIM,
    dropout=DROPOUT
).to(device)

# Count parameters
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n=== Transformer Configuration ===")
print(f"Embedding dim: {EMB_SIZE}")
print(f"Hidden dim: {HIDDEN_DIM}")
print(f"Encoder layers: {NUM_ENCODER_LAYERS}")
print(f"Decoder layers: {NUM_DECODER_LAYERS}")
print(f"Attention heads: {NHEAD}")
print(f"Dropout: {DROPOUT}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LR}")
print(f"Parameters: {param_count:,}\n")

# === Optimizer + Scheduler ===
optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

def lr_lambda(step):
    if step == 0:
        step = 1
    return (EMB_SIZE ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# === Loss ===
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

# === Training Loop ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for src, tgt_in, tgt_out in train_loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        optimizer.zero_grad()
        # Forward pass
        output = model(src, tgt_in)

        loss = loss_fn(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid exploding gradients
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in val_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            output = model(src, tgt_in)
            loss = loss_fn(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"[Epoch {epoch:02d}] "
          f"Train Loss: {avg_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"PPL: {math.exp(val_loss):.2f}")
