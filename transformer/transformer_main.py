import sys, os, time, torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from dataset import RecipeDataset, collate_fn
# from transformer.transformer_model import Seq2SeqTransformer
from transformer.transformer_model_tuned import Seq2SeqTransformerTuned as Seq2SeqTransformer

from logger import log_results, evaluate, print_model_info, log_epoch, log_examples, compute_confusion_small, save_confusion_heatmap

DATA_PATH = os.environ.get("DATA_PATH", "data/processed_recipes.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = "/content/drive/MyDrive/DLAM_Project/results"
os.makedirs(CKPT_DIR, exist_ok=True)

def get_model_name_t(m):
    return f"TRANS_d{m.embedding_dim}_layers{m.num_layers}_drop{m.dropout:.1f}"

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, pad_idx, scheduler=None):
    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_val_acc = best_val_bleu = best_val_em = best_val_jacc = 0.0
    best_epoch = None
    train_losses, val_losses = [], []
    start = time.time()

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(DEVICE)

    params = {
        "embedding_dim": model.embedding_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "batch_size": train_loader.batch_size,
        "lr": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0].get("weight_decay",0.0)
    }
    print_model_info(model, params)

    for epoch in range(1, num_epochs+1):
        epoch_start = time.time()
        model.train()
        total = 0.0
        for batch in train_loader:
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(src, trg)
            logits_flat = logits[:,1:,:].reshape(-1, logits.size(-1))
            targets_flat = trg[:,1:].reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            total += loss.item()

        train_loss = total/len(train_loader)
        val_loss, val_ppl, val_acc, val_bleu, val_em, val_jacc = evaluate(model, val_loader, None, DEVICE)

        if val_loss < best_val_loss:
            best_val_loss, best_val_ppl = val_loss, val_ppl
            best_val_acc, best_val_bleu, best_val_em, best_val_jacc = val_acc, val_bleu, val_em, val_jacc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"transformer_epoch{epoch}_ppl{val_ppl:.2f}.pt"))

        print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} (ppl {val_ppl:.2f}) | acc {val_acc*100:.1f}% | BLEU {val_bleu:.3f} | EM {val_em*100:.1f}% | Jacc {val_jacc*100:.1f}%")

        train_losses.append(train_loss); val_losses.append(val_loss)
        epoch_time = time.time() - epoch_start
        log_epoch(RESULTS_DIR, get_model_name_t(model), epoch, train_loss, val_loss, val_ppl, val_acc, val_bleu, val_em, val_jacc, epoch_time)

    # examples
    examples = []
    base_ds = None
    try:
        from torch.utils.data import Subset
        def base_dataset(ds):
            while isinstance(ds, Subset): ds=ds.dataset
            return ds
        base_ds = base_dataset(train_loader.dataset)
    except Exception:
        pass

    val_iter = iter(val_loader)
    for _ in range(3):
        try: batch = next(val_iter)
        except StopIteration: break
        src = batch["input_ids"][:1].to(DEVICE)
        trg = batch["target_ids"][:1]
        with torch.no_grad():
            out = model.greedy_or_topk(src, max_len=20, sos=base_ds.target_vocab.word2idx["<SOS>"], eos=base_ds.target_vocab.word2idx["<EOS>"])
            pred_ids = out[0].tolist()[1:]
        gold_ids = trg[0].tolist()[1:]
        eos = base_ds.target_vocab.word2idx["<EOS>"]
        if eos in pred_ids: pred_ids = pred_ids[:pred_ids.index(eos)]
        if eos in gold_ids: gold_ids = gold_ids[:gold_ids.index(eos)]
        pred = " ".join([base_ds.target_vocab.idx2word.get(i,"<UNK>") for i in pred_ids])
        gold = " ".join([base_ds.target_vocab.idx2word.get(i,"<UNK>") for i in gold_ids])
        src_ids = src[0].tolist()[1:]
        eos_in = base_ds.input_vocab.word2idx.get("<EOS>", None)
        if eos_in in src_ids: src_ids = src_ids[:src_ids.index(eos_in)]
        src_txt = " ".join([base_ds.input_vocab.idx2word.get(i,"<UNK>") for i in src_ids])
        print("SRC :", src_txt); print("PRED:", pred); print("GOLD:", gold)
        examples.append((src_txt, pred, gold))
    log_examples(RESULTS_DIR, get_model_name_t(model), examples)

    # confusion
    with torch.no_grad():
        try:
            batch = next(iter(val_loader))
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)
            logits = model(src, trg)[:,1:,:]
            preds = logits.argmax(-1).cpu()
            gold = trg[:,1:].cpu()
            cm = compute_confusion_small(preds, gold, pad_idx=base_ds.target_vocab.word2idx["<PAD>"], max_labels=30)
            save_confusion_heatmap(cm, os.path.join(RESULTS_DIR, f"{get_model_name_t(model)}_confusion.png"))
        except Exception:
            pass

    # loss curve
    plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_transformer.png")); plt.close()

    train_time = time.time()-start
    gpu_mem = (torch.cuda.max_memory_allocated(DEVICE)//(1024**2)) if torch.cuda.is_available() else 0

    log_results(
        base_dir=RESULTS_DIR, model_name=get_model_name_t(model), params=params,
        best_epoch=best_epoch, best_val_loss=best_val_loss, best_val_ppl=best_val_ppl,
        best_val_acc=best_val_acc, best_val_bleu=best_val_bleu, best_val_em=best_val_em,
        best_val_jacc=best_val_jacc, gpu_mem=gpu_mem, train_time=train_time,
        train_loss=train_losses[-1]
    )

if __name__=="__main__":
    dataset = RecipeDataset(DATA_PATH)
    use_subset=None
    if use_subset:
        from torch.utils.data import Subset
        dataset=Subset(dataset, range(use_subset))
    n_val=int(len(dataset)*0.2); n_train=len(dataset)-n_val
    train_set, val_set = random_split(dataset, [n_train,n_val])

    from torch.utils.data import Subset
    def base_dataset(ds):
        while isinstance(ds, Subset): ds=ds.dataset
        return ds
    vocab_ds = base_dataset(train_set)

    train_loader = DataLoader(train_set,batch_size=128,shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_set,batch_size=128,shuffle=False,collate_fn=collate_fn)
    pad_idx = vocab_ds.target_vocab.word2idx["<PAD>"]

    model = Seq2SeqTransformer(
        len(vocab_ds.input_vocab), len(vocab_ds.target_vocab),
        d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
        dim_ff=1024, dropout=0.2, pad_idx=pad_idx
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9,0.98), eps=1e-9, weight_decay=1e-5)

    d_model = model.embedding_dim; warmup_steps = 4000
    def lr_lambda(step):
        step = max(1, step)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train(model, train_loader, val_loader, optimizer, criterion, 15, pad_idx, scheduler)
