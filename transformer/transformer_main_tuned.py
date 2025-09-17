import os
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import sentencepiece as spm

from transformer_model_tuned import Seq2SeqTransformerTuned  # same folder
# If your project uses a package like "from transformer.transformer_model_tuned import ...",
# change the import above accordingly.

# -----------------------
# Config (edit as needed)
# -----------------------
@dataclass
class Config:
    TRAIN_CSV: str = "train.csv"     # expects columns: input, target
    VAL_CSV: str = "val.csv"
    SPM_MODEL: str = "/content/drive/MyDrive/DLAM_Project/data/spm_recipes.model"     # SentencePiece model path

    # Model
    EMB_SIZE: int = 512
    NHEAD: int = 8
    FFN_HID_DIM: int = 512
    NUM_ENCODER_LAYERS: int = 4
    NUM_DECODER_LAYERS: int = 4
    DROPOUT: float = 0.1

    # Optimization
    BATCH_SIZE: int = 32
    EPOCHS: int = 8
    BASE_LR: float = 5e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_STEPS: int = 1000
    CLIP_NORM: float = 1.0
    LABEL_SMOOTHING: float = 0.1

    # Generation/logging
    MAX_NEW_TOKENS: int = 128
    LOG_EXAMPLES: int = 3
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# -----------------------
# Data
# -----------------------
class RecipeDatasetSP(Dataset):
    def __init__(self, csv_path: str, sp: spm.SentencePieceProcessor):
        self.df = pd.read_csv(csv_path)
        assert {"input", "target"}.issubset(self.df.columns), "CSV must have 'input' and 'target' columns"
        self.sp = sp

        # id safety
        self.pad_id = self._safe_piece("<pad>", default=0)
        self.bos_id = self._safe_piece("<s>", default=1)
        self.eos_id = self._safe_piece("</s>", default=2)

    def _safe_piece(self, piece: str, default: int) -> int:
        pid = self.sp.piece_to_id(piece)
        return pid if pid is not None and pid >= 0 else default

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        src_text = self.df.iloc[idx]["input"]
        tgt_text = self.df.iloc[idx]["target"]
        src_ids = [self.bos_id] + self.sp.encode(str(src_text), out_type=int) + [self.eos_id]
        tgt_ids = [self.bos_id] + self.sp.encode(str(tgt_text), out_type=int) + [self.eos_id]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long), str(src_text), str(tgt_text)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str, str]], pad_id: int):
    srcs, tgts, src_texts, tgt_texts = zip(*batch)
    srcs_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_id)
    tgts_padded = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_id)
    return srcs_padded, tgts_padded, list(src_texts), list(tgt_texts)


# -----------------------
# Metrics & helpers
# -----------------------
def token_accuracy(logits: torch.Tensor, tgt_out: torch.Tensor, ignore_index: int) -> float:
    """
    logits: [B, T, V], tgt_out: [B, T]
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = tgt_out.ne(ignore_index)
        correct = (preds.eq(tgt_out) & mask).sum().item()
        total = mask.sum().item()
        return (correct / max(total, 1)) * 100.0


def perplexity(loss_value: float) -> float:
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float("inf")


def parse_ingredients(text: str) -> List[str]:
    """
    Extract entries like "['clam broth', 'milk', ...]" -> ["clam broth", "milk", ...]
    If not in that format, fall back to comma-split.
    """
    text = str(text)
    out: List[str] = []
    buf = ""
    in_quote = False
    quote_char = None
    for ch in text:
        if ch in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote_char = ch
                buf = ""
            else:
                if ch == quote_char:
                    in_quote = False
                    if buf.strip():
                        out.append(buf.strip())
                    buf = ""
                else:
                    buf += ch
        elif in_quote:
            buf += ch
    if out:
        return out
    # fallback
    return [p.strip() for p in text.split(",") if p.strip()]


def jaccard_from_texts(pred_text: str, gold_text: str) -> float:
    a = set(parse_ingredients(pred_text))
    b = set(parse_ingredients(gold_text))
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return 100.0 * len(a & b) / len(a | b)


# -----------------------
# Build model / optimizer
# -----------------------
def build_model_and_optim(sp: spm.SentencePieceProcessor) -> Tuple[nn.Module, AdamW, LambdaLR, int]:
    pad_id = sp.piece_to_id("<pad>")
    if pad_id is None or pad_id < 0:
        pad_id = 0
    vocab_size = sp.get_piece_size()

    model = Seq2SeqTransformerTuned(
        num_encoder_layers=cfg.NUM_ENCODER_LAYERS,
        num_decoder_layers=cfg.NUM_DECODER_LAYERS,
        emb_size=cfg.EMB_SIZE,
        nhead=cfg.NHEAD,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dim_feedforward=cfg.FFN_HID_DIM,
        dropout=cfg.DROPOUT,
        pad_idx=pad_id,
    ).to(cfg.DEVICE)

    optimizer = AdamW(model.parameters(), lr=cfg.BASE_LR, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.WEIGHT_DECAY)

    def inv_sqrt_lambda(step: int):
        step = max(step, 1)
        if step <= cfg.WARMUP_STEPS:
            return step / cfg.WARMUP_STEPS  # linear warmup to 1.0
        return (cfg.WARMUP_STEPS ** 0.5) / (step ** 0.5)

    scheduler = LambdaLR(optimizer, lr_lambda=inv_sqrt_lambda)
    return model, optimizer, scheduler, pad_id


# -----------------------
# Generation
# -----------------------
@torch.no_grad()
def greedy_generate(model: nn.Module, sp: spm.SentencePieceProcessor, text: str, device, max_new_tokens: int = 128) -> str:
    model.eval()
    bos = sp.piece_to_id("<s>") if sp.piece_to_id("<s>") >= 0 else 1
    eos = sp.piece_to_id("</s>") if sp.piece_to_id("</s>") >= 0 else 2
    pad = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") >= 0 else 0

    src_ids = [bos] + sp.encode(text, out_type=int) + [eos]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, S]
    src_key_mask = (src == pad)

    memory = model.encode(src, src_key_padding_mask=src_key_mask)

    ys = torch.tensor([[bos]], dtype=torch.long, device=device)  # [1, 1]
    for _ in range(max_new_tokens):
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1), device=device)
        out = model.decode(ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_mask)
        logits = model.generator(out[:, -1])  # [1, V]
        next_id = int(logits.argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos:
            break

    gen_ids = ys[0, 1:]
    if len(gen_ids) and gen_ids[-1].item() == eos:
        gen_ids = gen_ids[:-1]
    return sp.decode(gen_ids.tolist())


# -----------------------
# Training / Evaluation
# -----------------------
def train_one_epoch(model, loader, optimizer, scheduler, criterion, pad_id) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for src, tgt, _, _ in loader:
        src = src.to(cfg.DEVICE)
        tgt = tgt.to(cfg.DEVICE)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:].contiguous()

        tgt_mask = model.generate_square_subsequent_mask(tgt_in.size(1), device=src.device)
        src_pad_mask = src.eq(pad_id)
        tgt_pad_mask = tgt_in.eq(pad_id)

        logits = model(
            src=src,
            tgt_in=tgt_in,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask,
        )

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.CLIP_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += token_accuracy(logits, tgt_out, ignore_index=pad_id)
        n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, criterion, pad_id) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for src, tgt, _, _ in loader:
        src = src.to(cfg.DEVICE)
        tgt = tgt.to(cfg.DEVICE)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:].contiguous()

        tgt_mask = model.generate_square_subsequent_mask(tgt_in.size(1), device=src.device)
        src_pad_mask = src.eq(pad_id)
        tgt_pad_mask = tgt_in.eq(pad_id)

        logits = model(
            src=src,
            tgt_in=tgt_in,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        total_loss += loss.item()
        total_acc += token_accuracy(logits, tgt_out, ignore_index=pad_id)
        n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def fit():
    # Load SentencePiece
    if not os.path.exists(cfg.SPM_MODEL):
        raise FileNotFoundError(f"SentencePiece model not found at '{cfg.SPM_MODEL}'")
    sp = spm.SentencePieceProcessor(model_file=cfg.SPM_MODEL)

    # Data
    train_ds = RecipeDatasetSP(cfg.TRAIN_CSV, sp)
    val_ds = RecipeDatasetSP(cfg.VAL_CSV, sp)
    pad_id = train_ds.pad_id

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    # Model / Optim
    model, optimizer, scheduler, pad_id_m = build_model_and_optim(sp)
    assert pad_id_m == pad_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=cfg.LABEL_SMOOTHING)

    global_step = 0
    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, pad_id)
        val_loss, val_acc = evaluate(model, val_loader, criterion, pad_id)
        dt = time.time() - t0

        ppl = perplexity(val_loss)

        # LR logging
        lr_now = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f} | Acc: {val_acc:.2f}% | Time: {dt:.1f}s | LR: {lr_now:.6g}")

        # Print sample generations
        try:
            for i in range(min(cfg.LOG_EXAMPLES, len(val_ds))):
                sample_inp = val_ds.df.iloc[i]["input"]
                gold_text = val_ds.df.iloc[i]["target"]
                pred_text = greedy_generate(model, sp, sample_inp, cfg.DEVICE, max_new_tokens=cfg.MAX_NEW_TOKENS)
                jac = jaccard_from_texts(pred_text, gold_text)
                em = 100.0 if pred_text.strip() == str(gold_text).strip() else 0.0
                print(f"  Example {i+1}:\n    Pred: {pred_text}\n    Gold: {gold_text}\n    Jaccard: {jac:.2f}% | EM: {em:.2f}%")
        except Exception as e:
            print(f"(Skipping example print due to error: {e})")

    return model


if __name__ == "__main__":
    fit()
