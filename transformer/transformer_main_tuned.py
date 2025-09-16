import sys, os, math, torch, csv
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

from dataset import SPRecipeDataset, sp_collate_fn
from transformer.transformer_model_tuned import Seq2SeqTransformerTuned
from logger import print_model_info

# reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# paths
DATA_PATH = "/content/drive/MyDrive/DLAM_Project/data/processed_recipes.csv"
SP_MODEL = "/content/drive/MyDrive/DLAM_Project/data/spm_recipes.model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
EMB_SIZE = 512
HIDDEN_DIM = 1024
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
NHEAD = 8
DROPOUT = 0.2
BATCH_SIZE = 256
LR = 3e-4
EPOCHS = 15
WARMUP_STEPS = 4000


# ----------------- Metrics -----------------
def token_accuracy(preds, targets, pad_idx):
    """Compute token-level accuracy (ignoring PAD)."""
    non_pad = targets != pad_idx
    correct = (preds == targets) & non_pad
    return correct.sum().item() / non_pad.sum().item()


def jaccard_and_em(preds, targets, pad_idx):
    """Compute Jaccard + Exact Match (EM) for batch."""
    batch_size = preds.size(0)
    jaccards, ems = [], []
    for i in range(batch_size):
        pred_tokens = [t for t in preds[i].tolist() if t != pad_idx]
        tgt_tokens = [t for t in targets[i].tolist() if t != pad_idx]

        pred_set, tgt_set = set(pred_tokens), set(tgt_tokens)
        inter = len(pred_set & tgt_set)
        union = len(pred_set | tgt_set)
        jaccard = inter / union if union > 0 else 0.0
        em = 1.0 if pred_tokens == tgt_tokens else 0.0

        jaccards.append(jaccard)
        ems.append(em)

    return sum(jaccards) / batch_size, sum(ems) / batch_size


def decode(token_ids, sp, pad_idx):
    """Decode list of token ids into text using sentencepiece."""
    ids = [i for i in token_ids if i != pad_idx]
    return sp.decode_ids(ids)


# ----------------- Training -----------------
def train(model, train_loader, val_loader, optimizer, scheduler, criterion,
          num_epochs, pad_idx, sp):
    log_path = "/content/drive/MyDrive/DLAM_Project/training_log.csv"

    # CSV header
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "ppl",
                         "acc", "jaccard", "em"])

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, total_acc = 0.0, 0.0

        for batch in train_loader:
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])

            loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            preds = output.argmax(-1)
            total_acc += token_accuracy(preds, trg[:, 1:], pad_idx)

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss, val_acc, jac_list, em_list = 0.0, 0.0, [], []
        sample_outputs = []

        with torch.no_grad():
            for batch in val_loader:
                src = batch["input_ids"].to(DEVICE)
                trg = batch["target_ids"].to(DEVICE)

                output = model(src, trg[:, :-1])
                loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))
                val_loss += loss.item()

                preds = output.argmax(-1)
                val_acc += token_accuracy(preds, trg[:, 1:], pad_idx)

                jac, em = jaccard_and_em(preds, trg[:, 1:], pad_idx)
                jac_list.append(jac)
                em_list.append(em)

                if len(sample_outputs) < 3:
                    sample_outputs.append((
                        decode(preds[0].tolist(), sp, pad_idx),
                        decode(trg[0].tolist(), sp, pad_idx)
                    ))

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        avg_jac = sum(jac_list) / len(jac_list)
        avg_em = sum(em_list) / len(em_list)

        ppl = math.exp(val_loss)

        # print
        print(f"[Epoch {epoch:02d}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"PPL: {ppl:.2f} | "
              f"Acc: {val_acc*100:.2f}% | "
              f"Jaccard: {avg_jac*100:.2f}% | EM: {avg_em*100:.2f}%")

        for i, (pred, gold) in enumerate(sample_outputs, 1):
            print(f"  Example {i}:")
            print(f"    Pred: {pred}")
            print(f"    Gold: {gold}")

        # save to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, val_loss, ppl,
                             val_acc, avg_jac, avg_em])


# ----------------- Main -----------------
if __name__ == "__main__":
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)

    # dataset
    dataset = SPRecipeDataset(DATA_PATH, SP_MODEL)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=sp_collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=sp_collate_fn)

    pad_idx = sp.piece_to_id("<pad>")

    # model
    model = Seq2SeqTransformerTuned(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=EMB_SIZE,
        nhead=NHEAD,
        src_vocab_size=sp.get_piece_size(),
        tgt_vocab_size=sp.get_piece_size(),
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
    train(model, train_loader, val_loader, optimizer, scheduler, criterion,
          EPOCHS, pad_idx, sp)
