import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import RecipeDataset, collate_fn
from model import EncoderRNN, DecoderRNN

import csv
import time

LOGFILE = "experiment_log.csv"

def log_results(model_name, params, best_epoch, best_val_loss, best_val_ppl, best_val_acc, train_time, train_loss):
    """
    Schreibt Ergebnisse in eine CSV-Datei.
    """
    file_exists = os.path.isfile(LOGFILE)
    with open(LOGFILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model", "embedding_dim", "hidden_dim", "num_layers", "dropout",
                "batch_size", "lr", "weight_decay",
                "best_epoch", "best_val_loss", "best_val_ppl", "best_val_acc",
                "train_loss", "gen_gap", "train_time"
            ])

        gen_gap = best_val_loss - train_loss
        writer.writerow([
            model_name,
            params.get("embedding_dim"),
            params.get("hidden_dim"),
            params.get("num_layers"),
            params.get("dropout"),
            params.get("batch_size"),
            params.get("lr"),
            params.get("weight_decay"),
            best_epoch,
            round(best_val_loss, 4),
            round(best_val_ppl, 2),
            round(best_val_acc * 100, 2),
            round(train_loss, 4),
            round(gen_gap, 4),
            round(train_time, 2)
        ])

# function to name the model
def get_model_name(enc, dec):
    return f"LSTM_{enc.embedding_dim}emb_{enc.hidden_dim}hid_{enc.num_layers}ly_{enc.dropout:.1f}drop"

# configurations
DEFAULT_DATA_PATH = "data/processed_recipes.csv"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAD_CLIP = 1.0
CKPT_DIR = os.environ.get("CKPT_DIR", "./checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


# train encoder and decoder end to end
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_idx, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        self.pad_idx = pad_idx

    def forward(self, src, trg, src_length, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size()
        vocab_size = self.decoder.fc_out.out_features

        # Encoder
        encoder_outputs, hidden, cell = self.encoder(src, src_length)
        # fixed bug with calling
        enc_layers = hidden.size(0)
        dec_layers = self.decoder.num_layers
        if enc_layers != dec_layers:
            # nur die letzten Schichten des Encoders verwenden
            hidden = hidden[-dec_layers:]
            cell = cell[-dec_layers:]

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        input_token = trg[:, 0]  # <SOS>

        for t in range(1, trg_len):
            step_logits, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = step_logits

            use_tf = (torch.rand(1).item() < teacher_forcing_ratio)
            input_token = trg[:, t] if use_tf else step_logits.argmax(dim=1)

        return outputs


@torch.no_grad()
def greedy_decode_one(model, dataset, title_ids, title_len, max_len=15):
    model.eval()

    # Encoder liefert outputs, hidden, cell
    encoder_outputs, hidden, cell = model.encoder(
        title_ids.unsqueeze(0).to(DEVICE),
        title_len.unsqueeze(0)
    )

    sos_idx = dataset.target_vocab.word2idx["<SOS>"]
    eos_idx = dataset.target_vocab.word2idx["<EOS>"]
    input_token = torch.tensor([sos_idx], device=DEVICE)

    out_tokens = []
    for _ in range(max_len):
        # Decoder erwartet auch encoder_outputs
        logits, hidden, cell, _ = model.decoder(
            input_token, hidden, cell, encoder_outputs
        )
        next_id = logits.argmax(dim=1)
        tok = int(next_id.item())
        if tok == eos_idx:
            break
        word = dataset.target_vocab.idx2word.get(tok, "<UNK>")
        out_tokens.append(word)
        input_token = next_id

    return out_tokens


@torch.no_grad()
def evaluate(model, loader, criterion, pad_idx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in loader:
        src = batch["input_ids"].to(DEVICE)
        trg = batch["target_ids"].to(DEVICE)
        src_lengths = batch["input_lengths"]

        logits = model(src, trg, src_lengths, teacher_forcing_ratio=0.0)

        logits_flat  = logits[:, 1:, :].contiguous().view(-1, logits.size(-1))
        targets_flat = trg[:, 1:].contiguous().view(-1)
        loss = criterion(logits_flat, targets_flat)
        total_loss += loss.item()

        preds = logits[:, 1:, :].argmax(dim=-1)
        gold  = trg[:, 1:]
        mask  = (gold != pad_idx)
        correct = (preds == gold) & mask
        total_correct += correct.sum().item()
        total_tokens  += mask.sum().item()

    avg_loss = total_loss / max(1, len(loader))
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    return avg_loss, ppl, acc

train_losses_all = []
val_losses_all = []

def train(model, train_loader, val_loader, optimizer, criterion, dataset,
          num_epochs=10, pad_idx=0, teacher_forcing_ratio=0.5):
    best_val_ppl = float("inf")
    best_val_loss = None
    best_val_acc = None
    best_epoch = None
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        teacher_forcing_ratio = max(0.5 * (0.95 ** epoch), 0.1)
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)
            src_lengths = batch["input_lengths"]

            optimizer.zero_grad()
            logits = model(src, trg, src_lengths, teacher_forcing_ratio=teacher_forcing_ratio)

            logits_flat  = logits[:, 1:, :].contiguous().view(-1, logits.size(-1))
            targets_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_ppl = float(torch.exp(torch.tensor(train_loss)))

        val_loss, val_ppl, val_acc = evaluate(model, val_loader, criterion, pad_idx)
        scheduler.step(val_loss)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            ckpt_path = os.path.join(CKPT_DIR, f"best_epoch{epoch}_ppl{val_ppl:.2f}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        print(f"Epoch {epoch:02d} | "
              f"Train loss {train_loss:.4f} (ppl {train_ppl:.2f})  | "
              f"Val loss {val_loss:.4f} (ppl {val_ppl:.2f})  | "
              f"Val token-acc {val_acc*100:.1f}%")

        sample_title = batch["input_ids"][0]
        sample_len   = batch["input_lengths"][0]
        prediction = greedy_decode_one(model, dataset, sample_title, sample_len)
        print("  Predicted:", prediction[:12])

        train_losses_all.append(train_loss)
        val_losses_all.append(val_loss)

    train_time = time.time() - start_time
    params = {
        "embedding_dim": model.encoder.embedding_dim,
        "hidden_dim": model.encoder.hidden_dim,
        "num_layers": model.encoder.num_layers,
        "dropout": model.encoder.dropout,
        "batch_size": train_loader.batch_size,
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
    }
    model_name = get_model_name(model.encoder, model.decoder)
    log_results(model_name, params, best_epoch, best_val_loss, best_val_ppl, best_val_acc, train_time, train_loss)

if __name__ == "__main__":
    dataset = RecipeDataset(DATA_PATH)

    use_subset = 50000  # None für Vollgröße
    if use_subset is not None:
        dataset = torch.utils.data.Subset(dataset, range(use_subset))

    val_ratio = 0.2
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    import torch.utils.data as tud
    def base_dataset(ds):
        while isinstance(ds, tud.Subset):
            ds = ds.dataset
        return ds

    vocab_ds = base_dataset(train_set)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set, batch_size=256, shuffle=False, collate_fn=collate_fn)

    enc = EncoderRNN(len(vocab_ds.input_vocab), 512, 256, 2, bidirectional=False)
    dec = DecoderRNN(
        len(vocab_ds.target_vocab),
        512,
        256,
        num_layers=2)

    model = Seq2Seq(enc, dec, DEVICE,
                    sos_idx=vocab_ds.target_vocab.word2idx["<SOS>"],
                    pad_idx=vocab_ds.target_vocab.word2idx["<PAD>"]).to(DEVICE)

    pad_idx = vocab_ds.target_vocab.word2idx["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2)

    train(model, train_loader, val_loader, optimizer, criterion,
          dataset=vocab_ds,
          num_epochs=20,
          pad_idx=pad_idx,
          teacher_forcing_ratio=0.5)

    import matplotlib.pyplot as plt
    plt.plot(train_losses_all, label="Train Loss")
    plt.plot(val_losses_all, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()
