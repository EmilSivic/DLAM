import os, time, csv, torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import RecipeDataset, collate_fn
from transformer_model import Seq2SeqTransformer
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


LOGFILE = "experiment_log.csv"
DATA_PATH = os.environ.get("DATA_PATH", "data/processed_recipes.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = os.environ.get("CKPT_DIR", "./checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

def log_results(model_name, params, best_epoch, best_val_loss, best_val_ppl, best_val_acc, train_time, train_loss):
    file_exists = os.path.isfile(LOGFILE)
    with open(LOGFILE, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "model","embedding_dim","hidden_dim","num_layers","dropout",
                "batch_size","lr","weight_decay",
                "best_epoch","best_val_loss","best_val_ppl","best_val_acc",
                "train_loss","gen_gap","train_time"
            ])
        w.writerow([
            model_name, params["embedding_dim"], params["hidden_dim"],
            params["num_layers"], params["dropout"], params["batch_size"],
            params["lr"], params["weight_decay"], best_epoch,
            round(best_val_loss,4), round(best_val_ppl,2), round(best_val_acc*100,2),
            round(train_loss,4), round(best_val_loss-train_loss,4), round(train_time,2)
        ])

def get_model_name_t(m):
    return f"TRANS_d{m.embedding_dim}_layers{m.num_layers}_drop{m.dropout:.1f}"

@torch.no_grad()
def evaluate(model, loader, criterion, pad_idx):
    model.eval()
    total_loss, total_tokens, total_correct = 0,0,0
    for batch in loader:
        src = batch["input_ids"].to(DEVICE)
        trg = batch["target_ids"].to(DEVICE)
        logits = model(src, trg)
        logits_flat = logits[:,1:,:].reshape(-1, logits.size(-1))
        targets_flat = trg[:,1:].reshape(-1)
        loss = criterion(logits_flat, targets_flat)
        total_loss += loss.item()
        preds = logits[:,1:,:].argmax(-1)
        gold = trg[:,1:]
        mask = (gold != pad_idx)
        total_correct += ((preds==gold)&mask).sum().item()
        total_tokens += mask.sum().item()
    avg_loss = total_loss/len(loader)
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    acc = total_correct/total_tokens if total_tokens>0 else 0
    return avg_loss, ppl, acc

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, pad_idx, scheduler=None):
    best_val_ppl = float("inf")
    best_val_loss=best_val_acc=best_epoch=None
    start = time.time()
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
            total_loss += loss.item()
        train_loss = total_loss/len(train_loader)
        train_ppl = float(torch.exp(torch.tensor(train_loss)))

        val_loss, val_ppl, val_acc = evaluate(model, val_loader, criterion, pad_idx)
        if scheduler: scheduler.step(val_loss)

        if val_ppl < best_val_ppl:
            best_val_ppl, best_val_loss, best_val_acc, best_epoch = val_ppl, val_loss, val_acc, epoch
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"transformer_epoch{epoch}_ppl{val_ppl:.2f}.pt"))
        print(f"Epoch {epoch} | train {train_loss:.4f} (ppl {train_ppl:.2f}) | val {val_loss:.4f} (ppl {val_ppl:.2f}) | acc {val_acc*100:.1f}%")

        sample = batch["input_ids"][0:1].to(DEVICE)
        sos = vocab_ds.target_vocab.word2idx["<SOS>"]; eos = vocab_ds.target_vocab.word2idx["<EOS>"]
        out_ids = model.greedy_or_topk(sample, 15, sos, eos)[0].tolist()
        words = [vocab_ds.target_vocab.idx2word.get(i,"<UNK>") for i in out_ids[1:]]
        print("  Pred:", words[:12])

    train_time = time.time()-start
    params = {
        "embedding_dim": model.embedding_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "batch_size": train_loader.batch_size,
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0].get("weight_decay",0.0)
    }
    log_results(get_model_name_t(model), params, best_epoch, best_val_loss, best_val_ppl, best_val_acc, train_time, train_loss)

if __name__=="__main__":
    dataset = RecipeDataset(DATA_PATH)
    use_subset=50000
    if use_subset: from torch.utils.data import Subset; dataset=Subset(dataset, range(use_subset))
    n_val=int(len(dataset)*0.2); n_train=len(dataset)-n_val
    train_set, val_set = random_split(dataset, [n_train,n_val])

    import torch.utils.data as tud
    def base_dataset(ds):
        while isinstance(ds,tud.Subset): ds=ds.dataset
        return ds
    vocab_ds = base_dataset(train_set)

    train_loader = DataLoader(train_set,batch_size=128,shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_set,batch_size=128,shuffle=False,collate_fn=collate_fn)
    pad_idx = vocab_ds.target_vocab.word2idx["<PAD>"]

    model = Seq2SeqTransformer(len(vocab_ds.input_vocab), len(vocab_ds.target_vocab),
                               d_model=512, nhead=8,
                               num_encoder_layers=4, num_decoder_layers=4,
                               dim_ff=2048, dropout=0.1, pad_idx=pad_idx).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train(model, train_loader, val_loader, optimizer, criterion, 20, pad_idx, scheduler)
