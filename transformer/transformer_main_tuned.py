import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

# Make sure we can import from project root
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from dataset import RecipeDataset, collate_fn
from transformer.transformer_model_tuned import Seq2SeqTransformerTuned
from logger import evaluate, print_model_info

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
DATA_PATH = os.environ.get("DATA_PATH", "data/processed_recipes.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = RecipeDataset(DATA_PATH)
n_val = int(len(dataset) * 0.2)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

pad_idx = train_loader.dataset.dataset.target_vocab.word2idx["<PAD>"]

# === Model ===
model = Seq2SeqTransformerTuned(
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    emb_size=EMB_SIZE,
    nhead=NHEAD,
    src_vocab_size=len(train_loader.dataset.dataset.input_vocab),
    tgt_vocab_size=len(train_loader.dataset.dataset.target_vocab),
    dim_feedforward=HIDDEN_DIM,
    dropout=DROPOUT,
    pad_idx=pad_idx
).to(DEVICE)


# Count parameters
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n=== Transformer Configuration ===")
print_model_info(model, {
    "embedding_dim": EMB_SIZE,
    "hidden_dim": HIDDEN_DIM,
    "num_layers": NUM_ENCODER_LAYERS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "parameters": param_count
})

# === Optimizer + Scheduler ===
optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

def lr_lambda(step):
    if step == 0:
        step = 1
    return (EMB_SIZE ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# === Loss ===
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

# === Training Loop ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        src = batch["input_ids"].to(DEVICE)
        tgt = batch["target_ids"].to(DEVICE)

        optimizer.zero_grad()
        output = model(src, tgt)

        loss = loss_fn(output[:, 1:, :].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # === Validation ===
    model.eval()
    val_loss, val_ppl, val_acc, *_ = evaluate(model, val_loader, None, DEVICE)

    print(f"[Epoch {epoch:02d}] "
          f"Train Loss: {avg_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"PPL: {val_ppl:.2f} | "
          f"Acc: {val_acc*100:.2f}%")
