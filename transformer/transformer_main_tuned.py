import sys, os

# Add project root to sys.path 
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

import sys, os, math, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from dataset import RecipeDataset, collate_fn
from transformer.transformer_model_tuned import Seq2SeqTransformerTuned
from logger import evaluate, print_model_info
from torch.utils.data import DataLoader, random_split

# reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# paths
DATA_PATH = "/content/drive/MyDrive/DLAM_Project/data/processed_recipes.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
EMB_SIZE = 512
HIDDEN_DIM = 2048
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NHEAD = 8
DROPOUT = 0.2
BATCH_SIZE = 128
LR = 3e-4
EPOCHS = 15
WARMUP_STEPS = 4000


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, pad_idx):
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)

            optimizer.zero_grad()
            output = model(src, tgt_in)

            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        val_loss, val_ppl, val_acc, *_ = evaluate(model, val_loader, None, DEVICE)

        print(f"[Epoch {epoch:02d}] Train Loss: {avg_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} | PPL: {val_ppl:.2f} | Acc: {val_acc*100:.2f}%")


if __name__ == "__main__":
    # data
    dataset = RecipeDataset(DATA_PATH)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    pad_idx = train_loader.dataset.dataset.target_vocab.word2idx["<PAD>"]

    # model
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

    # optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    def lr_lambda(step):
        step = max(1, step)
        return (EMB_SIZE ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # loss with label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

    # print config
    print("\n=== Transformer Configuration ===\n")
    print_model_info(model, {
        "embedding_dim": EMB_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_ENCODER_LAYERS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0)
    })

    # train
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, EPOCHS, pad_idx)
