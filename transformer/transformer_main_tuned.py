import sys, os, time, torch, random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Seed Fix
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Pfade
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from dataset import RecipeDataset, collate_fn
from transformer.transformer_model_tuned import Seq2SeqTransformerTuned
from logger import (
    log_results, evaluate, print_model_info, log_epoch
)

# Pfade
DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/content/drive/MyDrive/DLAM_Project/data/processed_recipes.csv"
)



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = "/content/drive/MyDrive/DLAM_Project/results"
os.makedirs(CKPT_DIR, exist_ok=True)

# Scheduler
class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = (self.d_model ** -0.5) * min(
            step ** -0.5, step * (self.warmup_steps ** -1.5)
        )
        return [scale for _ in self.base_lrs]

# Training
def train(model, train_loader, val_loader,
          optimizer, criterion, num_epochs, pad_idx,
          scheduler=None, patience=5):
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            src, tgt = batch["input_ids"].to(DEVICE), batch["target_ids"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(src, tgt[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_ppl, val_acc, val_bleu, val_em, val_jacc = evaluate(model, val_loader, None, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"(ppl {val_ppl:.2f}) | acc {val_acc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"best_model_epoch{epoch}.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Plot Loss
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
    plt.close()


if __name__ == "__main__":
    dataset = RecipeDataset(DATA_PATH)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

    pad_idx = train_loader.dataset.dataset.target_vocab.word2idx["<PAD>"]

    model = Seq2SeqTransformerTuned(
        len(train_loader.dataset.dataset.input_vocab),
        len(train_loader.dataset.dataset.target_vocab),
        d_model=512, nhead=8,
        num_encoder_layers=6, num_decoder_layers=6,
        dim_ff=2048, dropout=0.3, pad_idx=pad_idx, tie_weights=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    scheduler = NoamScheduler(optimizer, d_model=model.embedding_dim, warmup_steps=4000)

    train(model, train_loader, val_loader, optimizer, criterion, 40, pad_idx, scheduler=scheduler, patience=7)
