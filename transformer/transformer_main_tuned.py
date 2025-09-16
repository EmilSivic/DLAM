import sys, os, math, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

# Project imports
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from dataset import RecipeDataset, collate_fn
from transformer.transformer_model_tuned import Seq2SeqTransformerTuned
from logger import print_model_info

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


# ---- Jaccard + Exact Match ----
def jaccard_and_em(preds, targets, pad_idx):
    jac_scores, em_scores = [], []
    for p, t in zip(preds.tolist(), targets.tolist()):
        p = [tok for tok in p if tok != pad_idx]
        t = [tok for tok in t if tok != pad_idx]
        p_set, t_set = set(p), set(t)
        inter = len(p_set & t_set)
        union = len(p_set | t_set) if len(p_set | t_set) > 0 else 1
        jac_scores.append(inter / union)
        em_scores.append(1.0 if p == t else 0.0)
    return sum(jac_scores) / len(jac_scores), sum(em_scores) / len(em_scores)


# ---- Decode tokens -> words ----
def decode(tokens, vocab, pad_idx):
    toks = [t for t in tokens if t != pad_idx]
    return " ".join(vocab.idx2word[t] for t in toks)


# ---- Training Loop ----
def train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, pad_idx,
          input_vocab, target_vocab):
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])  # teacher forcing (shifted input)

            loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss, jac_list, em_list = 0.0, [], []
        sample_outputs = []

        with torch.no_grad():
            for batch in val_loader:
                src = batch["input_ids"].to(DEVICE)
                trg = batch["target_ids"].to(DEVICE)

                output = model(src, trg[:, :-1])
                loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))
                val_loss += loss.item()

                # greedy predictions
                pred_tokens = output.argmax(-1)
                jac, em = jaccard_and_em(pred_tokens, trg[:, 1:], pad_idx)
                jac_list.append(jac)
                em_list.append(em)

                # collect some samples
                if len(sample_outputs) < 3:  # show 3 examples per epoch
                    sample_outputs.append((
                        decode(pred_tokens[0].tolist(), target_vocab, pad_idx),
                        decode(trg[0].tolist(), target_vocab, pad_idx)
                    ))

        val_loss /= len(val_loader)
        avg_jac = sum(jac_list) / len(jac_list)
        avg_em = sum(em_list) / len(em_list)

        print(f"[Epoch {epoch:02d}] Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"PPL: {math.exp(val_loss):.2f} | "
              f"Jaccard: {avg_jac*100:.2f}% | EM: {avg_em*100:.2f}%")

        # print sample predictions
        for i, (pred, gold) in enumerate(sample_outputs, 1):
            print(f"  Example {i}:")
            print(f"    Pred: {pred}")
            print(f"    Gold: {gold}")


if __name__ == "__main__":
    dataset = RecipeDataset(DATA_PATH)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    pad_idx = train_loader.dataset.dataset.target_vocab.word2idx["<PAD>"]

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

    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)

    def lr_lambda(step):
        step = max(1, step)
        return (EMB_SIZE ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

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

    # pass vocabs so we can decode predictions
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, EPOCHS, pad_idx,
          train_loader.dataset.dataset.input_vocab, train_loader.dataset.dataset.target_vocab)
