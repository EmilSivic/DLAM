import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import RecipeDataset, collate_fn
from model import EncoderRNN, DecoderRNN
from logger import log_results, evaluate, print_model_info

# configs
DEFAULT_DATA_PATH = "data/processed_recipes.csv"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAD_CLIP = 1.0
CKPT_DIR = os.environ.get("CKPT_DIR", "./checkpoints")
RESULTS_DIR = "/content/drive/MyDrive/DLAM_Project/results"
os.makedirs(CKPT_DIR, exist_ok=True)

# seq2seq wrapper
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

        encoder_outputs, hidden, cell = self.encoder(src, src_length)

        seq_len = encoder_outputs.size(1)
        src_mask = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        src_mask = src_mask < src_length.unsqueeze(1).to(self.device)

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)
        input_token = trg[:, 0]  # <SOS>

        for t in range(1, trg_len):
            step_logits, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask=src_mask
            )
            outputs[:, t, :] = step_logits

            use_tf = (torch.rand(1).item() < teacher_forcing_ratio)
            input_token = trg[:, t] if use_tf else step_logits.argmax(dim=1)

        return outputs

# name for logging
def get_model_name(enc, dec):
    return f"LSTM_{enc.embedding_dim}emb_{enc.hidden_dim}hid_{enc.num_layers}ly_{enc.dropout:.1f}drop"

# training loop
train_losses_all = []
val_losses_all = []

def train(model, train_loader, val_loader, optimizer, criterion, dataset,
          num_epochs=10, pad_idx=0, teacher_forcing_ratio=0.5):
    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_val_acc = best_val_bleu = best_val_em = best_val_jacc = 0.0
    best_epoch = None
    start_time = time.time()

    # params for logging
    params = {
        "embedding_dim": model.encoder.embedding_dim,
        "hidden_dim": model.encoder.hidden_dim,
        "num_layers": model.encoder.num_layers,
        "dropout": model.encoder.dropout,
        "batch_size": train_loader.batch_size,
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
    }
    print_model_info(model, params)

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

        # use shared logger.evaluate
        val_loss, val_ppl, val_acc, val_bleu, val_em, val_jacc = evaluate(
            model, val_loader, None, DEVICE, pad_idx=pad_idx
        )

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_bleu, best_val_em, best_val_jacc = val_bleu, val_em, val_jacc
            best_epoch = epoch
            ckpt_path = os.path.join(CKPT_DIR, f"best_epoch{epoch}_ppl{val_ppl:.2f}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        print(f"Epoch {epoch:02d} | "
              f"Train loss {train_loss:.4f} | Val loss {val_loss:.4f} (ppl {val_ppl:.2f}) | "
              f"Val token-acc {val_acc*100:.1f}% | BLEU {val_bleu:.3f} | EM {val_em*100:.1f}% | Jacc {val_jacc*100:.1f}%")

        train_losses_all.append(train_loss)
        val_losses_all.append(val_loss)

    train_time = time.time() - start_time
    log_results(
        base_dir=RESULTS_DIR,
        model_name=get_model_name(model.encoder, model.decoder),
        params=params,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_ppl=best_val_ppl,
        best_val_acc=best_val_acc,
        best_val_bleu=best_val_bleu,
        best_val_em=best_val_em,
        best_val_jacc=best_val_jacc,
        gpu_mem=(torch.cuda.max_memory_allocated(DEVICE)//(1024**2)) if torch.cuda.is_available() else 0,
        train_time=train_time,
        train_loss=train_losses_all[-1]
    )

if __name__ == "__main__":
    dataset = RecipeDataset(DATA_PATH)
    use_subset = 50000
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

    enc = EncoderRNN(
        len(vocab_ds.input_vocab), 512, 256,
        num_layers=2, dropout=0.3, bidirectional=True
    )
    dec = DecoderRNN(
        len(vocab_ds.target_vocab), 512, 256,
        enc_dim=1024,  # bidirectional => 2*hidden
        num_layers=2, dropout=0.3
    )

    model = Seq2Seq(
        enc, dec, DEVICE,
        sos_idx=vocab_ds.target_vocab.word2idx["<SOS>"],
        pad_idx=vocab_ds.target_vocab.word2idx["<PAD>"]
    ).to(DEVICE)

    pad_idx = vocab_ds.target_vocab.word2idx["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2)

    train(model, train_loader, val_loader, optimizer, criterion,
          dataset=vocab_ds,
          num_epochs=20,
          pad_idx=pad_idx,
          teacher_forcing_ratio=0.5)

    # loss curve speichern
    import matplotlib.pyplot as plt
    plt.plot(train_losses_all, label="Train Loss")
    plt.plot(val_losses_all, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()
