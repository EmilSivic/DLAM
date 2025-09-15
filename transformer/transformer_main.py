import sys, os, time, torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from transformer.transformer_model import Seq2SeqTransformer
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import RecipeDataset, collate_fn

# Logging import
from logger import log_results, evaluate, print_model_info

# === Config ===
DATA_PATH = os.environ.get("DATA_PATH", "data/processed_recipes.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = os.environ.get("CKPT_DIR", "./checkpoints")
RESULTS_DIR = "/content/drive/MyDrive/DLAM_Project/results"
os.makedirs(CKPT_DIR, exist_ok=True)

def get_model_name_t(m):
    return f"TRANS_d{m.embedding_dim}_layers{m.num_layers}_drop{m.dropout:.1f}"

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, pad_idx, scheduler=None):
    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_val_acc = best_val_bleu = best_val_em = best_val_jacc = 0.0
    best_epoch = None
    train_losses, val_losses = [], []
    start = time.time()

    # Reset GPU stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    # Hyperparams dict for logging
    params = {
        "embedding_dim": model.embedding_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "batch_size": train_loader.batch_size,
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0].get("weight_decay",0.0)
    }
    print_model_info(model, params)

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss=0
        for batch in train_loader:
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(src, trg)
            logits_flat = logits[:,1:,:].reshape(-1, logits.size(-1))
            targets_flat = trg[:,1:].reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            total_loss += loss.item()

        train_loss = total_loss/len(train_loader)

        # use logger.evaluate for consistency (returns loss, ppl, acc, bleu, em, jacc)
        val_loss, val_ppl, val_acc, val_bleu, val_em, val_jacc = evaluate(
            model, val_loader, None, DEVICE
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"(ppl {val_ppl:.2f}) | acc {val_acc * 100:.1f}% | "
              f"BLEU {val_bleu:.3f} | EM {val_em * 100:.1f}% | Jacc {val_jacc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss, best_val_ppl = val_loss, val_ppl
            best_val_acc, best_val_bleu, best_val_em, best_val_jacc = val_acc, val_bleu, val_em, val_jacc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"transformer_epoch{epoch}_ppl{val_ppl:.2f}.pt"))

    # plot losses
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig("loss_plot.png"); plt.close()

    train_time = time.time()-start
    gpu_mem = (torch.cuda.max_memory_allocated(DEVICE)//(1024**2)) if torch.cuda.is_available() else 0

    log_results(
        base_dir=RESULTS_DIR,
        model_name=get_model_name_t(model),
        params=params,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_ppl=best_val_ppl,
        best_val_acc=best_val_acc,
        best_val_bleu=best_val_bleu,
        best_val_em=best_val_em,
        best_val_jacc=best_val_jacc,
        gpu_mem=gpu_mem,
        train_time=train_time,
        train_loss=train_losses[-1]
    )

if __name__=="__main__":
    dataset = RecipeDataset(DATA_PATH)
    use_subset=50000
    if use_subset:
        from torch.utils.data import Subset
        dataset=Subset(dataset, range(use_subset))
    n_val=int(len(dataset)*0.2); n_train=len(dataset)-n_val
    train_set, val_set = random_split(dataset, [n_train,n_val])

    from torch.utils.data import Subset
    def base_dataset(ds):
        while isinstance(ds, Subset): ds=ds.dataset
        return ds
    vocab_ds = base_dataset(train_set)

    train_loader = DataLoader(train_set,batch_size=128,shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_set,batch_size=128,shuffle=False,collate_fn=collate_fn)
    pad_idx = vocab_ds.target_vocab.word2idx["<PAD>"]

    model = Seq2SeqTransformer(
        len(vocab_ds.input_vocab), len(vocab_ds.target_vocab),
        d_model=256, nhead=4,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_ff=1024, dropout=0.2, pad_idx=pad_idx
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9,0.98), eps=1e-9, weight_decay=1e-5)

    d_model = model.embedding_dim
    warmup_steps = 4000
    def lr_lambda(step):
        step = max(1, step)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train(model, train_loader, val_loader, optimizer, criterion, 2, pad_idx, scheduler)
