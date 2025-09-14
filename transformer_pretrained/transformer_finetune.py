# transformer_finetune.py
import os, time, csv, math
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import ast

# cfg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# data
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
            tgt = ", ".join(tgt_list)  # clearer formatting
        except Exception:
            tgt = self.ingredients[idx]

        enc = self.tokenizer(
            src, max_length=self.max_len_src,
            padding="max_length", truncation=True,
            return_tensors="pt"
        )

        dec = self.tokenizer(
            tgt, max_length=self.max_len_tgt,
            padding="max_length", truncation=True,
            return_tensors="pt"
        )

        labels = dec["input_ids"].squeeze(0)
        labels[labels == tokenizer.pad_token_id] = -100  # ignore pad

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels
        }


# log
def log_results(model_name, params,
                best_epoch, best_val_loss, best_val_ppl,
                best_val_acc, best_val_bleu, best_val_em,
                train_time, train_loss):

    # detailed
    detailed_file = "results_detailed.csv"
    file_exists = os.path.isfile(detailed_file)
    with open(detailed_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "model","embedding_dim","hidden_dim","num_layers","dropout",
                "batch_size","lr","weight_decay",
                "best_epoch","best_val_loss","best_val_ppl",
                "best_val_acc","best_val_bleu","best_val_em",
                "train_loss","gen_gap","train_time"
            ])
        w.writerow([
            model_name, params.get("embedding_dim"), params.get("hidden_dim"),
            params.get("num_layers"), params.get("dropout"), params.get("batch_size"),
            params.get("lr"), params.get("weight_decay"), best_epoch,
            round(best_val_loss,4), round(best_val_ppl,2), round(best_val_acc*100,2),
            round(best_val_bleu,3), round(best_val_em*100,1),
            round(train_loss,4), round(best_val_loss-train_loss,4), round(train_time,2)
        ])

    # compact
    compact_file = "results_compact.csv"
    file_exists = os.path.isfile(compact_file)
    with open(compact_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "model","best_epoch","best_val_loss","best_val_ppl",
                "best_val_acc","best_val_bleu","best_val_em","train_time"
            ])
        w.writerow([
            model_name, best_epoch,
            round(best_val_loss,4), round(best_val_ppl,2),
            round(best_val_acc*100,2), round(best_val_bleu,3),
            round(best_val_em*100,1), round(train_time,2)
        ])


# eval
@torch.no_grad()
def evaluate(model, loader, max_len=20):
    model.eval()
    total_loss, total_correct, total_tokens = 0,0,0
    total_bleu, total_em, n_samples = 0,0,0
    smoothie = SmoothingFunction().method4

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += out.loss.item()

        logits = out.logits
        preds = logits.argmax(-1)
        mask = (labels != -100)
        total_correct += ((preds == labels) & mask).sum().item()
        total_tokens  += mask.sum().item()

        gen_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_length=max_len, num_beams=4
        )
        for i in range(input_ids.size(0)):
            pred_tokens = tokenizer.decode(gen_ids[i], skip_special_tokens=True).split()
            gold_tokens = tokenizer.decode(
                labels[i].masked_fill(labels[i]==-100, tokenizer.pad_token_id),
                skip_special_tokens=True
            ).split()

            bleu = sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smoothie)
            total_bleu += bleu
            total_em += int(pred_tokens == gold_tokens)
            n_samples += 1

    avg_loss = total_loss/len(loader)
    ppl = math.exp(avg_loss)
    acc = total_correct/total_tokens if total_tokens>0 else 0.0
    bleu_score = total_bleu/n_samples if n_samples>0 else 0.0
    em_score = total_em/n_samples if n_samples>0 else 0.0
    return avg_loss, ppl, acc, bleu_score, em_score


# train
def train_model(dataset, batch_size=16, num_epochs=5):
    n_total = len(dataset)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # --- print model info once ---
    cfg = model.config
    print("=== Model configuration ===")
    print("Model:", model.name_or_path)
    print("Embedding dim:", getattr(cfg, "d_model", None))
    print("Hidden dim:", getattr(cfg, "d_ff", None))
    print("Num layers:", getattr(cfg, "num_layers", None))
    print("Dropout:", getattr(cfg, "dropout_rate", getattr(cfg, "dropout", None)))
    print("Batch size:", batch_size)
    print("Learning rate:", optimizer.param_groups[0]["lr"])
    print("Weight decay:", optimizer.param_groups[0].get("weight_decay", 0.0))
    print("===========================")

    train_losses, val_losses = [], []
    best_val_loss, best_epoch = float("inf"), None
    best_val_acc = best_val_bleu = best_val_em = 0.0
    start = time.time()

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip
            optimizer.step()
            running += loss.item()

        train_loss = running/len(train_loader)
        val_loss, val_ppl, val_acc, val_bleu, val_em = evaluate(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} (ppl {val_ppl:.2f}) | "
              f"acc {val_acc*100:.1f}% | BLEU {val_bleu:.3f} | EM {val_em*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc, best_val_bleu, best_val_em = val_acc, val_bleu, val_em
            torch.save(model.state_dict(), f"best_model_epoch{epoch}.pt")

    # plot
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig("loss_plot.png"); plt.close()

    train_time = time.time()-start
    # log real model cfg for T5
    params = {
        "embedding_dim": getattr(cfg, "d_model", None),
        "hidden_dim": getattr(cfg, "d_ff", None),
        "num_layers": getattr(cfg, "num_layers", None),
        "dropout": getattr(cfg, "dropout_rate", getattr(cfg, "dropout", None)),
        "batch_size": batch_size,
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0)
    }
    log_results(model_name=model.name_or_path,
                params=params,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                best_val_ppl=val_ppl,
                best_val_acc=best_val_acc,
                best_val_bleu=best_val_bleu,
                best_val_em=best_val_em,
                train_time=train_time,
                train_loss=train_losses[-1])


if __name__ == "__main__":
    dataset = RecipeDataset("/content/drive/MyDrive/DLAM_Project/data/processed_recipes.csv", tokenizer)
    train_model(dataset, batch_size=16, num_epochs=10)
    if os.path.exists("results_compact.csv"):
        print(pd.read_csv("results_compact.csv").tail())
