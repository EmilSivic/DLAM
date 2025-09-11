import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import RecipeDataset, collate_fn
from model import EncoderRNN, DecoderRNN

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
        """
        src:  [B, src_len]   (Titel-Token-IDs)
        trg:  [B, trg_len]   (Zutaten-Token-IDs, inkl. <SOS>/<EOS>)
        src_length: [B]      (reale Längen der Titel, vor Padding)
        """
        batch_size, trg_len = trg.size(0), trg.size(1)
        vocab_size = self.decoder.fc_out.out_features

        # Encoder reads titles
        hidden, cell = self.encoder(src, src_length)

        # reserviere speicher für logits
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        # start decoder with <SOS> liegt als trg[:,0] vor
        input_token = trg[:, 0].to(self.device)

        # Schrittweise über die Ziel-Länge laufen (Auto-Regressiv)
        for t in range(1, trg_len):
            # 1) einen Schritt decoden
            step_logits, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = step_logits  # Logits für Zeitpunkt t speichern

            # 2) Teacher Forcing: manchmal nehmen wir das echte nächste Token,
            #    manchmal die eigene Vorhersage (argmax)
            use_tf = (torch.rand(1).item() < teacher_forcing_ratio)
            if use_tf:
                input_token = trg[:, t].long()              # Ground Truth einspeisen
            else:
                input_token = step_logits.argmax(dim=1)     # Eigene Vorhersage einspeisen

        return outputs


# -----------------------------
# Greedy Decoding (nur für Debug/Inference)
# -----------------------------
@torch.no_grad()
def greedy_decode_one(model, dataset, title_ids, title_len, max_len=15):
    """
    Ich gebe dem Decoder nur <SOS> und die Encoder-States.
    Danach immer das zuletzt vorhergesagte Token wieder rein.
    """
    model.eval()

    # Encoder “liest” den Titel
    hidden, cell = model.encoder(title_ids.unsqueeze(0).to(DEVICE),
                                 title_len.unsqueeze(0))

    sos_idx = dataset.target_vocab.word2idx["<SOS>"]
    eos_idx = dataset.target_vocab.word2idx["<EOS>"]
    input_token = torch.tensor([sos_idx], device=DEVICE)

    out_tokens = []
    for _ in range(max_len):
        logits, hidden, cell = model.decoder(input_token, hidden, cell)
        next_id = logits.argmax(dim=1)              # Top-1 wählen
        tok = int(next_id.item())
        if tok == eos_idx:                          # Ende der Sequenz
            break
        word = dataset.target_vocab.idx2word.get(tok, "<UNK>")
        out_tokens.append(word)
        input_token = next_id                       # Autoregressiv füttern

    return out_tokens


# -----------------------------
# Evaluation: Loss, PPL, Token-Accuracy
# -----------------------------
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

        # Im Eval keine Teacher-Forcing-„Schummelei“
        logits = model(src, trg, src_lengths, teacher_forcing_ratio=0.0)

        # Loss berechnen (klassisches next-token, ohne t=0)
        logits_flat  = logits[:, 1:, :].contiguous().view(-1, logits.size(-1))
        targets_flat = trg[:, 1:].contiguous().view(-1)
        loss = criterion(logits_flat, targets_flat)
        total_loss += loss.item()

        # Token-Accuracy: Vorhersage vs. Gold, PAD ignorieren
        preds = logits[:, 1:, :].argmax(dim=-1)   # [B, T-1]
        gold  = trg[:, 1:]                        # [B, T-1]
        mask  = (gold != pad_idx)                 # nur echte Tokens
        correct = (preds == gold) & mask
        total_correct += correct.sum().item()
        total_tokens  += mask.sum().item()

    avg_loss = total_loss / max(1, len(loader))
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    return avg_loss, ppl, acc



# train with validations and checkpoints
def train(model, train_loader, val_loader, optimizer, criterion, dataset,
          num_epochs=10, pad_idx=0, teacher_forcing_ratio=0.5):
    best_val_ppl = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)
            src_lengths = batch["input_lengths"]

            optimizer.zero_grad()
            logits = model(src, trg, src_lengths, teacher_forcing_ratio=teacher_forcing_ratio)

            # Loss über t>=1 (klassischer Shift)
            logits_flat  = logits[:, 1:, :].contiguous().view(-1, logits.size(-1))
            targets_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()

        # Train-Metriken
        train_loss = total_loss / len(train_loader)
        train_ppl = float(torch.exp(torch.tensor(train_loss)))

        # Validation
        val_loss, val_ppl, val_acc = evaluate(model, val_loader, criterion, pad_idx)

        print(f"Epoch {epoch:02d} | "
              f"Train loss {train_loss:.4f} (ppl {train_ppl:.2f})  | "
              f"Val loss {val_loss:.4f} (ppl {val_ppl:.2f})  | "
              f"Val token-acc {val_acc*100:.1f}%")

        # Bestes Modell speichern (nach Val-PPL)
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            ckpt_path = os.path.join(CKPT_DIR, f"best_epoch{epoch}_ppl{val_ppl:.2f}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Kleine Debug-Prediction (ein Beispiel aus letztem Batch der Epoche)
        sample_title = batch["input_ids"][0]
        sample_len   = batch["input_lengths"][0]
        prediction = greedy_decode_one(model, dataset, sample_title, sample_len)
        print("  Predicted:", prediction[:12])


# entrypoint
if __name__ == "__main__":
    dataset = RecipeDataset(DATA_PATH)

    # SUBSET of data
    use_subset = 20000  # setze None für Vollgröße
    if use_subset is not None:
        dataset = torch.utils.data.Subset(dataset, range(use_subset))

    #Train/Val-Split 90/10
    val_ratio = 0.1
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # baseset
    import torch.utils.data as tud
    def base_dataset(ds):
        # ent-nestet Subsets bis zur echten RecipeDataset
        while isinstance(ds, tud.Subset):
            ds = ds.dataset
        return ds

    vocab_ds = base_dataset(train_set)

    #dataloader
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, collate_fn=collate_fn)

    #models
    enc = EncoderRNN(len(vocab_ds.input_vocab), 128, 256, 1)
    dec = DecoderRNN(len(vocab_ds.target_vocab), 128, 256, 1)
    model = Seq2Seq(enc, dec, DEVICE,
                    sos_idx=vocab_ds.target_vocab.word2idx["<SOS>"],
                    pad_idx=vocab_ds.target_vocab.word2idx["<PAD>"]).to(DEVICE)

    # train
    pad_idx = vocab_ds.target_vocab.word2idx["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, train_loader, val_loader, optimizer, criterion,
          dataset=vocab_ds,
          num_epochs=10,
          pad_idx=pad_idx,
          teacher_forcing_ratio=0.5)

import matplotlib.pyplot as plt
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.show()
