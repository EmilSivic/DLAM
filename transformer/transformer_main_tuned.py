import sys, os, time, torch, random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# paths
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from dataset import RecipeDataset, collate_fn
from transformer.transformer_model_tuned import Seq2SeqTransformerTuned
from logger import log_results, evaluate, print_model_info, log_epoch

DATA_PATH = os.environ.get("DATA_PATH", "data/processed_recipes.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = "/content/drive/MyDrive/DLAM_Project/results"
os.makedirs(CKPT_DIR, exist_ok=True)

def get_model_name_tuned(m, tag=""):
    return f"TRANS_TUNED_d{m.embedding_dim}_layers{m.num_layers}_drop{m.dropout:.1f}{tag}"

# training loop
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, pad_idx, tag=""):
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = None
    train_losses, val_losses = [], []
    start = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    params = {
        "embedding_dim": model.embedding_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "batch_size": train_loader.batch_size,
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
    }
    print_model_info(model, params)

    for epoch in range(1, num_epochs+1):
        model.train()
        total = 0.0
        for batch in train_loader:
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(src, trg)
            logits_flat = logits[:, 1:, :].reshape(-1, logits.size(-1))
            targets_flat = trg[:, 1:].reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        train_loss = total / len(train_loader)
        val_loss, val_ppl, val_acc, val_bleu, val_em, val_jacc = evaluate(model, val_loader, None, DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, f"transformer_tuned{tag}_epoch{epoch}_ppl{val_ppl:.2f}.pt"))

        print(f"[{tag}] Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"(ppl {val_ppl:.2f}) | acc {val_acc*100:.1f}% | BLEU {val_bleu:.3f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        log_epoch(RESULTS_DIR, get_model_name_tuned(model, tag), epoch,
                  train_loss, val_loss, val_ppl, val_acc, val_bleu, val_em, val_jacc, time.time()-start)

    log_results(
        base_dir=RESULTS_DIR,
        model_name=get_model_name_tuned(model, tag),
        params=params,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_ppl=val_ppl,
        best_val_acc=best_val_acc,
        best_val_bleu=val_bleu,
        best_val_em=val_em,
        best_val_jacc=val_jacc,
        gpu_mem=(torch.cuda.max_memory_allocated(DEVICE)//(1024**2)) if torch.cuda.is_available() else 0,
        train_time=time.time()-start,
        train_loss=train_losses[-1],
    )

# grid search
if __name__ == "__main__":
    dataset = RecipeDataset(DATA_PATH)
    n_val = int(len(dataset)*0.2)
    n_train = len(dataset)-n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, collate_fn=collate_fn)
    pad_idx = train_loader.dataset.dataset.target_vocab.word2idx["<PAD>"]

    grid = [
        {"d_model": 256, "nhead": 4, "layers": (3,3), "dim_ff": 1024, "dropout": 0.2, "lr": 5e-4},
        {"d_model": 256, "nhead": 8, "layers": (3,3), "dim_ff": 1024, "dropout": 0.3, "lr": 5e-4},
        {"d_model": 512, "nhead": 8, "layers": (4,4), "dim_ff": 2048, "dropout": 0.3, "lr": 3e-4},
    ]

    for i, cfg in enumerate(grid, 1):
        print(f"\n=== Grid Run {i}: {cfg} ===")
        model = Seq2SeqTransformerTuned(
            len(train_loader.dataset.dataset.input_vocab),
            len(train_loader.dataset.dataset.target_vocab),
            d_model=cfg["d_model"], nhead=cfg["nhead"],
            num_encoder_layers=cfg["layers"][0],
            num_decoder_layers=cfg["layers"][1],
            dim_ff=cfg["dim_ff"], dropout=cfg["dropout"],
            pad_idx=pad_idx,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9,0.98), eps=1e-9, weight_decay=1e-5)

        train(model, train_loader, val_loader, optimizer, criterion, 30, pad_idx, tag=f"_run{i}")
