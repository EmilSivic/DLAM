# main.py
import os, time, torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import RecipeDataset, collate_fn
from model import EncoderRNN, DecoderRNN
from logger import log_results, evaluate, print_model_info, log_epoch, log_examples, compute_confusion_small, save_confusion_heatmap
import matplotlib.pyplot as plt

DEFAULT_DATA_PATH = "data/processed_recipes.csv"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAD_CLIP = 1.0
CKPT_DIR = os.path.join(".", "checkpoints")
RESULTS_DIR = "/content/drive/MyDrive/DLAM_Project/results"
os.makedirs(CKPT_DIR, exist_ok=True)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_idx, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
    def forward(self, src, trg, src_length, teacher_forcing_ratio=0.5):
        b, T = trg.size()
        vocab = self.decoder.fc_out.out_features
        enc_out, h, c = self.encoder(src, src_length)
        L = enc_out.size(1)
        mask = torch.arange(L, device=self.device).unsqueeze(0).expand(b, -1) < src_length.unsqueeze(1).to(self.device)
        outputs = torch.zeros(b, T, vocab, device=self.device)
        token = trg[:,0]
        for t in range(1, T):
            step_logits, h, c, _ = self.decoder(token, h, c, enc_out, mask=mask)
            outputs[:,t,:] = step_logits
            use_tf = (torch.rand(1).item() < teacher_forcing_ratio)
            token = trg[:,t] if use_tf else step_logits.argmax(-1)
        return outputs

def get_model_name(enc, dec):
    return f"LSTM_{enc.embedding_dim}emb_{enc.hidden_dim}hid_{enc.num_layers}ly_{enc.dropout:.1f}drop"

def train(model, train_loader, val_loader, optimizer, criterion, dataset,
          model_tag, num_epochs=10, pad_idx=0, teacher_forcing_ratio=0.5):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_val_acc = best_val_bleu = best_val_em = best_val_jacc = 0.0
    best_epoch = None
    start_time = time.time()

    params = {
        "embedding_dim": model.encoder.embedding_dim,
        "hidden_dim": model.encoder.hidden_dim,
        "num_layers": model.encoder.num_layers,
        "dropout": model.encoder.dropout,
        "batch_size": train_loader.batch_size,
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
    }
    print("\nTraining", model_tag)
    print_model_info(model, params)

    for epoch in range(1, num_epochs+1):
        epoch_start = time.time()
        model.train()
        total = 0.0
        for batch in train_loader:
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)
            src_lengths = batch["input_lengths"]
            optimizer.zero_grad()
            logits = model(src, trg, src_lengths, teacher_forcing_ratio=teacher_forcing_ratio)
            logits_flat = logits[:,1:,:].reshape(-1, logits.size(-1))
            tgt_flat = trg[:,1:].reshape(-1)
            loss = criterion(logits_flat, tgt_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item()

        train_loss = total/len(train_loader)
        val_loss, val_ppl, val_acc, val_bleu, val_em, val_jacc = evaluate(
            model, val_loader, None, DEVICE, pad_idx=pad_idx
        )

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_val_loss = val_loss
            best_val_acc, best_val_bleu, best_val_em, best_val_jacc = val_acc, val_bleu, val_em, val_jacc
            best_epoch = epoch
            ckpt = os.path.join(CKPT_DIR, f"{model_tag}_epoch{epoch}_ppl{val_ppl:.2f}.pt")
            torch.save(model.state_dict(), ckpt)

        print(f"[{model_tag}] Epoch {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} (ppl {val_ppl:.2f}) | Acc {val_acc*100:.1f}% | BLEU {val_bleu:.3f} | EM {val_em*100:.1f}% | Jacc {val_jacc*100:.1f}%")

        train_losses.append(train_loss); val_losses.append(val_loss)
        epoch_time = time.time() - epoch_start
        model_name_for_epochs = f"{model_tag}_{get_model_name(model.encoder, model.decoder)}"
        log_epoch(RESULTS_DIR, model_name_for_epochs, epoch, train_loss, val_loss, val_ppl, val_acc, val_bleu, val_em, val_jacc, epoch_time)

    # examples (3 samples)
    examples = []
    val_iter = iter(val_loader)
    for _ in range(3):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        src = batch["input_ids"][:1].to(DEVICE)
        src_len = batch["input_lengths"][:1]
        trg = batch["target_ids"][:1].to(DEVICE)
        with torch.no_grad():
            out = model(src, trg, src_len, teacher_forcing_ratio=0.0)
            ids = out.argmax(-1)[0].tolist()
        # simple decode via idx2word (cut at EOS if exists)
        base_ds = dataset
        sos = base_ds.target_vocab.word2idx["<SOS>"]; eos = base_ds.target_vocab.word2idx["<EOS>"]
        pred_ids = ids[1:ids.index(eos)] if eos in ids[1:] else ids[1:]
        gold_ids = trg[0].tolist()
        gold_ids = gold_ids[1:gold_ids.index(eos)] if eos in gold_ids[1:] else gold_ids[1:]
        pred = " ".join([base_ds.target_vocab.idx2word.get(i, "<UNK>") for i in pred_ids])
        gold = " ".join([base_ds.target_vocab.idx2word.get(i, "<UNK>") for i in gold_ids])
        # source decode
        src_ids = src[0].tolist()
        eos_in = base_ds.input_vocab.word2idx.get("<EOS>", None)
        src_ids = src_ids[1:src_ids.index(eos_in)] if (eos_in and eos_in in src_ids[1:]) else src_ids[1:]
        src_txt = " ".join([base_ds.input_vocab.idx2word.get(i, "<UNK>") for i in src_ids])
        print("SRC :", src_txt); print("PRED:", pred); print("GOLD:", gold)
        examples.append((src_txt, pred, gold))
    log_examples(RESULTS_DIR, model_tag, examples)

    # confusion matrix on a small batch
    with torch.no_grad():
        try:
            batch = next(iter(val_loader))
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)
            src_lengths = batch["input_lengths"]
            out = model(src, trg, src_lengths, teacher_forcing_ratio=0.0)
            logits = out[:,1:,:]
            preds = logits.argmax(-1).cpu()
            gold = trg[:,1:].cpu()
            cm = compute_confusion_small(preds, gold, pad_idx=pad_idx, max_labels=30)
            save_confusion_heatmap(cm, os.path.join(RESULTS_DIR, f"{model_tag}_confusion.png"))
        except Exception:
            pass

    # loss curve
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"loss_{model_tag}.png")); plt.close()

    train_time = time.time()-start_time
    log_results(
        base_dir=RESULTS_DIR,
        model_name=f"{model_tag}_{get_model_name(model.encoder, model.decoder)}",
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
        train_loss=train_losses[-1]
    )

if __name__ == "__main__":
    dataset = RecipeDataset(DATA_PATH)
    use_subset = None
    if use_subset is not None:
        dataset = torch.utils.data.Subset(dataset, range(use_subset))
    n_val = int(0.2 * len(dataset)); n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    import torch.utils.data as tud
    def base_dataset(ds):
        while isinstance(ds, tud.Subset): ds = ds.dataset
        return ds
    vocab_ds = base_dataset(train_set)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set, batch_size=256, shuffle=False, collate_fn=collate_fn)

    configs = {
        "small":  {"emb": 256, "hid": 256, "layers": 2, "dropout": 0.2, "bidir": True},
        "medium": {"emb": 512, "hid": 512, "layers": 2, "dropout": 0.3, "bidir": True},
        "large":  {"emb": 512, "hid": 512, "layers": 4, "dropout": 0.3, "bidir": True},
        "xl":     {"emb": 1024, "hid": 512, "layers": 4, "dropout": 0.4, "bidir": True},
    }

    for tag, cfg in configs.items():
        enc = EncoderRNN(len(vocab_ds.input_vocab), cfg["emb"], cfg["hid"],
                         num_layers=cfg["layers"], dropout=cfg["dropout"], bidirectional=cfg["bidir"])
        enc_dim = enc.hidden_dim * (2 if enc.bidirectional else 1)
        dec = DecoderRNN(len(vocab_ds.target_vocab), cfg["emb"], cfg["hid"],
                         enc_dim=enc_dim, num_layers=cfg["layers"], dropout=cfg["dropout"])
        model = Seq2Seq(enc, dec, DEVICE,
                        sos_idx=vocab_ds.target_vocab.word2idx["<SOS>"],
                        pad_idx=vocab_ds.target_vocab.word2idx["<PAD>"]).to(DEVICE)

        pad_idx = vocab_ds.target_vocab.word2idx["<PAD>"]
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        train(model, train_loader, val_loader, optimizer, criterion,
              dataset=vocab_ds, model_tag=tag, num_epochs=2, pad_idx=pad_idx, teacher_forcing_ratio=0.5)
