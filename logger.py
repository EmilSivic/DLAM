import os, csv, math, time, torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def print_model_info(model, params):
    print("\nModel configuration:")
    print("Model:", getattr(model, "name_or_path", type(model).__name__))
    if hasattr(model, "config"):
        cfg = model.config
        print("Embedding dim:", getattr(cfg, "d_model", None))
        print("Hidden dim:", getattr(cfg, "d_ff", None))
        print("Num layers:", getattr(cfg, "num_layers", None))
        print("Dropout:", getattr(cfg, "dropout_rate", getattr(cfg, "dropout", None)))
    elif hasattr(model, "encoder") and hasattr(model, "decoder"):
        enc, dec = model.encoder, model.decoder
        print("Embedding dim:", getattr(enc, "embedding_dim", None))
        print("Hidden dim:", getattr(enc, "hidden_dim", None))
        print("Encoder layers:", getattr(enc, "num_layers", None))
        print("Decoder layers:", getattr(dec, "num_layers", None))
        print("Dropout:", getattr(enc, "dropout", None))
    print("Batch size:", params.get("batch_size"))
    print("Learning rate:", params.get("lr"))
    print("Weight decay:", params.get("weight_decay"))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total\n")

def log_results(base_dir, model_name, params,
                best_epoch, best_val_loss, best_val_ppl,
                best_val_acc, best_val_bleu, best_val_em,
                best_val_jacc, gpu_mem, train_time, train_loss):

    os.makedirs(base_dir, exist_ok=True)
    detailed_file = os.path.join(base_dir, "results_detailed.csv")
    compact_file  = os.path.join(base_dir, "results_compact.csv")

    file_exists = os.path.isfile(detailed_file)
    with open(detailed_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "model","embedding_dim","hidden_dim","num_layers","dropout",
                "batch_size","lr","weight_decay",
                "best_epoch","best_val_loss","best_val_ppl",
                "best_val_acc","best_val_bleu","best_val_em","best_val_jacc",
                "train_loss","gen_gap","train_time","gpu_mem_MB"
            ])
        w.writerow([
            model_name, params.get("embedding_dim"), params.get("hidden_dim"),
            params.get("num_layers"), params.get("dropout"), params.get("batch_size"),
            params.get("lr"), params.get("weight_decay"), best_epoch,
            round(best_val_loss,4), round(best_val_ppl,2), round(best_val_acc*100,2),
            round(best_val_bleu,3), round(best_val_em*100,1), round(best_val_jacc*100,1),
            round(train_loss,4), round(best_val_loss-train_loss,4),
            round(train_time,2), gpu_mem
        ])

    file_exists = os.path.isfile(compact_file)
    with open(compact_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "model","best_epoch","best_val_loss","best_val_ppl",
                "best_val_acc","best_val_bleu","best_val_em","best_val_jacc",
                "gpu_mem_MB"
            ])
        w.writerow([
            model_name, best_epoch,
            round(best_val_loss,4), round(best_val_ppl,2),
            round(best_val_acc*100,2), round(best_val_bleu,3),
            round(best_val_em*100,1), round(best_val_jacc*100,1),
            gpu_mem
        ])

# epoch logging
def log_epoch(base_dir, model_name, epoch, train_loss, val_loss,
              val_ppl, val_acc, val_bleu, val_em, val_jacc, epoch_time):
    os.makedirs(base_dir, exist_ok=True)
    epoch_file = os.path.join(base_dir, f"{model_name}_epochs.csv")
    file_exists = os.path.isfile(epoch_file)
    with open(epoch_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "epoch","train_loss","val_loss","val_ppl",
                "val_acc","val_bleu","val_em","val_jacc","epoch_time"
            ])
        w.writerow([
            epoch, round(train_loss,4), round(val_loss,4), round(val_ppl,2),
            round(val_acc*100,2), round(val_bleu,3),
            round(val_em*100,1), round(val_jacc*100,1), round(epoch_time,2)
        ])

# examples logging
def log_examples(base_dir, model_name, examples):
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{model_name}_examples.txt")
    with open(path, "w", encoding="utf-8") as f:
        for i, (src, pred, gold) in enumerate(examples):
            f.write(f"Example {i+1}\n")
            f.write(f"SRC : {src}\n")
            f.write(f"PRED: {pred}\n")
            f.write(f"GOLD: {gold}\n\n")

# confusion matrix util (token-level)
def compute_confusion_small(preds, labels, pad_idx=-100, max_labels=30):
    import numpy as np
    import torch as T
    Y = labels.clone()
    P = preds.clone()
    mask = Y != pad_idx
    Y = Y[mask].flatten()
    P = P[mask].flatten()
    if Y.numel() == 0:
        return None

    # coarse: clip ids to < max_labels
    Yc = Y.clamp(min=0, max=max_labels-1)
    Pc = P.clamp(min=0, max=max_labels-1)
    K = max_labels
    cm = T.zeros((K, K), dtype=T.int64, device=Y.device)
    idx = Yc * K + Pc
    binc = T.bincount(idx, minlength=K*K)
    cm.view(-1)[:] = binc
    return cm.cpu().numpy()

def save_confusion_heatmap(cm, out_path, title="Confusion (first 30 ids)"):
    if cm is None:
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("gold")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_len=20, pad_idx=None):
    model.eval()
    total_loss = total_correct = total_tokens = 0
    total_bleu = total_em = total_jacc = 0
    n_samples = 0
    smoothie = SmoothingFunction().method4

    for batch in loader:
        if "attention_mask" in batch and "labels" in batch:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += out.loss.item()

            logits = out.logits
            preds = logits.argmax(-1)
            mask = (labels != -100)
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

            gen_ids = model.generate(input_ids, attention_mask=attention_mask,
                                     max_length=max_len, num_beams=4)
            for i in range(input_ids.size(0)):
                pred_tokens = tokenizer.decode(gen_ids[i], skip_special_tokens=True).split()
                gold_tokens = tokenizer.decode(
                    labels[i].masked_fill(labels[i]==-100, tokenizer.pad_token_id),
                    skip_special_tokens=True
                ).split()
                bleu = sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smoothie)
                total_bleu += bleu
                total_em += int(pred_tokens == gold_tokens)
                sp, sg = set(pred_tokens), set(gold_tokens)
                if sg:
                    total_jacc += len(sp & sg) / len(sp | sg)
                n_samples += 1

        elif "target_ids" in batch:
            src = batch["input_ids"].to(device)
            trg = batch["target_ids"].to(device)
            src_lengths = batch.get("input_lengths", None)
            if src_lengths is not None: src_lengths = src_lengths.to(device)
            out = model(src, trg, src_lengths, teacher_forcing_ratio=0.0) if src_lengths is not None else model(src, trg)
            logits = out[:,1:,:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                trg[:,1:].reshape(-1),
                ignore_index=pad_idx if pad_idx is not None else -100
            )
            total_loss += loss.item()
            preds = logits.argmax(-1)
            gold = trg[:,1:]
            mask = (gold != (pad_idx if pad_idx is not None else -100))
            total_correct += ((preds == gold) & mask).sum().item()
            total_tokens += mask.sum().item()
            n_samples += src.size(0)

        else:
            raise ValueError(f"Unknown batch format: keys={batch.keys()}")

    avg_loss = total_loss/len(loader)
    ppl = math.exp(avg_loss)
    acc = total_correct/total_tokens if total_tokens>0 else 0.0
    bleu_score = total_bleu/n_samples if n_samples>0 else 0.0
    em_score = total_em/n_samples if n_samples>0 else 0.0
    jacc_score = total_jacc/n_samples if n_samples>0 else 0.0
    return avg_loss, ppl, acc, bleu_score, em_score, jacc_score
