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
from logger import evaluate, print_model_info

DATA_PATH = os.environ.get("DATA_PATH", "data/processed_recipes.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# training loop (vereinfacht für grid search)
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, pad_idx):
    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(src, trg)
            loss = criterion(
                logits[:, 1:, :].reshape(-1, logits.size(-1)),
                trg[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        train_loss = total / len(train_loader)
        val_loss, val_ppl, val_acc, *_ = evaluate(model, val_loader, None, DEVICE)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch} | train {train_loss:.4f} "
              f"| val {val_loss:.4f} (ppl {val_ppl:.2f}) "
              f"| acc {val_acc*100:.1f}%")

    return best_val_acc


if __name__ == "__main__":
    # data
    dataset = RecipeDataset(DATA_PATH)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, collate_fn=collate_fn)
    pad_idx = train_loader.dataset.dataset.target_vocab.word2idx["<PAD>"]

    # grid
    layers = [3, 6]
    smoothings = [0.0, 0.05, 0.1]

    for num_layers in layers:
        for smoothing in smoothings:
            print(f"\n=== Run: layers={num_layers}, label_smoothing={smoothing} ===\n")

            model = Seq2SeqTransformerTuned(
                len(train_loader.dataset.dataset.input_vocab),
                len(train_loader.dataset.dataset.target_vocab),
                d_model=256, nhead=4,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dim_ff=1024, dropout=0.1, pad_idx=pad_idx
            ).to(DEVICE)

            criterion = nn.CrossEntropyLoss(
                ignore_index=pad_idx,
                label_smoothing=smoothing
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=3e-4,
                betas=(0.9, 0.98),
                eps=1e-9,
                weight_decay=1e-5
            )

            print_model_info(model, {
                "embedding_dim": model.embedding_dim,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "dropout": model.dropout,
                "batch_size": train_loader.batch_size,
                "lr": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0)
            })

            best_val_acc = train(model, train_loader, val_loader, optimizer, criterion, 15, pad_idx)
            print(f">>> Best Val Acc: {best_val_acc*100:.2f}%")
