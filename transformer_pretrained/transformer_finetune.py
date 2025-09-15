import os, time, torch, ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW

from logger import log_results, evaluate, print_model_info

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# tokenize + model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# dataset
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
            "labels": labels,
            "src_text": src,
            "tgt_text": tgt
        }

# train
def train_model(dataset, batch_size=16, num_epochs=10, results_dir="/content/drive/MyDrive/DLAM_Project/results"):
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

    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

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

        print(f"[epoch {epoch}] train {train_loss:.4f} | "
              f"val {val_loss:.4f} (ppl {val_ppl:.2f}) | "
              f"acc {val_acc*100:.1f}% | BLEU {val_bleu:.3f} | "
              f"EM {val_em*100:.1f}% | Jacc {val_jacc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc, best_val_bleu, best_val_em, best_val_jacc = val_acc, val_bleu, val_em, val_jacc
            ckpt_path = os.path.join(ckpt_dir, f"finetune_best_epoch{epoch}_ppl{val_ppl:.2f}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # plot of losses
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig(os.path.join(results_dir, "finetune_loss_plot.png")); plt.close()

    # sample outputs
    model.eval()
    print("\nSample outputs from validation set:")
    for i, batch in enumerate(val_loader):
        if i >= 3: break
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        gen_ids = model.generate(input_ids, attention_mask=attn_mask, max_length=30)
        for j in range(min(2, input_ids.size(0))):
            src = batch["src_text"][j]
            gold = batch["tgt_text"][j]
            pred = tokenizer.decode(gen_ids[j], skip_special_tokens=True)
            print(f"src: {src}\n → gold: {gold}\n → pred: {pred}\n")

    # confusion matrix (token level)
    from sklearn.metrics import confusion_matrix
    import numpy as np
    all_preds, all_golds = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            gen_ids = model.generate(input_ids, attention_mask=attn_mask, max_length=labels.size(1))
            for p, g in zip(gen_ids, labels):
                p_tokens = tokenizer.decode(p, skip_special_tokens=True).split()
                g_tokens = tokenizer.decode(g.masked_fill(g==-100, tokenizer.pad_token_id), skip_special_tokens=True).split()
                min_len = min(len(p_tokens), len(g_tokens))
                all_preds.extend(p_tokens[:min_len])
                all_golds.extend(g_tokens[:min_len])

    if all_preds:
        vocab = list(set(all_preds + all_golds))
        cm = confusion_matrix(all_golds, all_preds, labels=vocab)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, xticklabels=vocab, yticklabels=vocab, cmap="Blues", cbar=False)
        plt.xlabel("Predicted"); plt.ylabel("Gold")
        plt.title("Token-level Confusion Matrix (subset)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "finetune_confusion_matrix.png"))
        plt.close()

    # log
    train_time = time.time()-start
    gpu_mem = (torch.cuda.max_memory_allocated(device)//(1024**2)) if torch.cuda.is_available() else 0
    log_results(
        base_dir=results_dir,
        model_name=model.name_or_path,
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
        train_loss=train_losses[-1]
    )

if __name__ == "__main__":
    dataset = RecipeDataset("/content/drive/MyDrive/DLAM_Project/data/processed_recipes.csv", tokenizer)
    train_model(dataset, batch_size=16, num_epochs=15)
