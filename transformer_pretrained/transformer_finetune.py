import os, time, ast
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
import matplotlib.pyplot as plt
from Compare.logger import log_results, evaluate, print_model_info


# --- cfg ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# Dataset
class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, tokenizer, max_len_src=32, max_len_tgt=64):
        df = pd.read_csv(csv_file)
        self.titles = df["input"].tolist()
        self.ingredients = df["target"].tolist()
        self.tokenizer = tokenizer
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        src = self.titles[idx]
        try:
            tgt_list = ast.literal_eval(self.ingredients[idx])
            tgt = ", ".join(tgt_list)
        except Exception:
            tgt = self.ingredients[idx]

        enc = self.tokenizer(src, max_length=self.max_len_src,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        dec = self.tokenizer(tgt, max_length=self.max_len_tgt,
                             padding="max_length", truncation=True,
                             return_tensors="pt")

        labels = dec["input_ids"].squeeze(0)
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels
        }

# Training
def train_model(dataset, batch_size=16, num_epochs=5):
    n_total = len(dataset)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []
    best_val_loss, best_epoch = float("inf"), None
    best_val_acc = best_val_bleu = best_val_em = best_val_jacc = 0.0
    start = time.time()

    # reset memory tracker
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # model info
    cfg = model.config
    params = {
        "embedding_dim": getattr(cfg, "d_model", None),
        "hidden_dim": getattr(cfg, "d_ff", None),
        "num_layers": getattr(cfg, "num_layers", None),
        "dropout": getattr(cfg, "dropout_rate", getattr(cfg, "dropout", None)),
        "batch_size": batch_size,
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0)
    }
    print_model_info(model, params)

    for epoch in range(1, num_epochs+1):
        model.train()
        running = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        train_loss = running/len(train_loader)
        val_loss, val_ppl, val_acc, val_bleu, val_em, val_jacc = evaluate(model, val_loader, tokenizer, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} (ppl {val_ppl:.2f}) | "
              f"acc {val_acc*100:.1f}% | BLEU {val_bleu:.3f} | "
              f"EM {val_em*100:.1f}% | Jacc {val_jacc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc, best_val_bleu, best_val_em, best_val_jacc = val_acc, val_bleu, val_em, val_jacc
            torch.save(model.state_dict(), f"best_model_epoch{epoch}.pt")

    # plot
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig("loss_plot.png"); plt.close()

    train_time = time.time()-start
    gpu_mem = (torch.cuda.max_memory_allocated(device)//(1024**2)) if torch.cuda.is_available() else 0

    # logging
    base_dir = "/content/drive/MyDrive/DLAM_Project/results"
    log_results(model_name=model.name_or_path,
                params=params,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                best_val_ppl=val_ppl,
                best_val_acc=best_val_acc,
                best_val_bleu=best_val_bleu,
                best_val_em=best_val_em,
                best_val_jacc=best_val_jacc,
                gpu_mem=gpu_mem,
                train_time=train_time,
                train_loss=train_losses[-1])

if __name__ == "__main__":
    dataset = RecipeDataset("/content/drive/MyDrive/DLAM_Project/data/processed_recipes.csv", tokenizer)
    train_model(dataset, batch_size=16, num_epochs=5)

    # optional summary
    summary_file = "/content/drive/MyDrive/DLAM_Project/results/results_summary.csv"
    if os.path.exists(summary_file):
        print(pd.read_csv(summary_file).tail())
