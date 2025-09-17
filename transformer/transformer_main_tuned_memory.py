# transformer_main_tuned.py

# --- put this BEFORE importing torch so it's effective in a fresh process ---
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math
import time
import hashlib
import random
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
import sentencepiece as spm

from transformer_model_tuned import Seq2SeqTransformerTuned  # same folder

# Enable TF32 on A100/V100 etc (faster matmul, same-ish quality)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass


# -----------------------
# Config (edit as needed)
# -----------------------
@dataclass
class Config:
    # Single dataset to load and split
    DATA_CSV: str = "/content/drive/MyDrive/DLAM_Project/data_processed_new.csv/processed_recipes_multitask_ner.csv"
    SPM_MODEL: str = "/content/drive/MyDrive/DLAM_Project/data/spm_recipes.model"

    # Data split
    VAL_FRACTION: float = 0.05
    SHUFFLE_SEED: int = 42

    # Sequence caps (CRUCIAL for steps)
    MAX_SRC_LEN: int = 64
    MAX_TGT_LEN: int = 256   # bump to 320–512 later if VRAM allows

    # Model
    EMB_SIZE: int = 512
    NHEAD: int = 8
    FFN_HID_DIM: int = 512
    NUM_ENCODER_LAYERS: int = 4
    NUM_DECODER_LAYERS: int = 4
    DROPOUT: float = 0.1

    # Optimization
    BATCH_SIZE: int = 8      # was 32; much safer with long steps
    EPOCHS: int = 8
    BASE_LR: float = 5e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_STEPS: int = 1000
    CLIP_NORM: float = 1.0
    LABEL_SMOOTHING: float = 0.1

    # Gradient accumulation + AMP
    GRAD_ACCUM_STEPS: int = 2
    USE_AMP: bool = True

    # Dataloader perf
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True

    # Generation/logging
    MAX_NEW_TOKENS: int = 256   # steps are longer
    LOG_EXAMPLES: int = 3
    LOG_EVERY_N_STEPS: int = 200
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Epoch-level text-metric evaluation (generation-based)
    METRIC_EVAL_SAMPLES: int = 256      # set -1 to evaluate all val rows
    METRIC_EVAL_SEED: int = 123

    # (Optional) subsample for quick smoke tests; 0 = all
    MAX_TRAIN_SAMPLES: int = 0
    MAX_VAL_SAMPLES: int = 0


cfg = Config()


# -----------------------
# Data
# -----------------------
class RecipeDatasetSP(Dataset):
    """Takes a pre-sliced DataFrame (train or val) and a SentencePiece processor."""
    def __init__(self, df: pd.DataFrame, sp: spm.SentencePieceProcessor):
        assert {"input", "target"}.issubset(df.columns), "CSV must have 'input' and 'target' columns"
        self.df = df[["input", "target"]].dropna().reset_index(drop=True)
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

        # tokenize
        src_ids = [self.bos_id] + self.sp.encode(str(src_text), out_type=int) + [self.eos_id]
        tgt_ids = [self.bos_id] + self.sp.encode(str(tgt_text), out_type=int) + [self.eos_id]

        # truncate (keep EOS)
        if len(src_ids) > cfg.MAX_SRC_LEN:
            src_ids = src_ids[:cfg.MAX_SRC_LEN]
            src_ids[-1] = self.eos_id
        if len(tgt_ids) > cfg.MAX_TGT_LEN:
            tgt_ids = tgt_ids[:cfg.MAX_TGT_LEN]
            tgt_ids[-1] = self.eos_id

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
            str(src_text),
            str(tgt_text),
        )


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str, str]], pad_id: int):
    srcs, tgts, src_texts, tgt_texts = zip(*batch)
    srcs_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_id)
    tgts_padded = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_id)
    return srcs_padded, tgts_padded, list(src_texts), list(tgt_texts)


# -----------------------
# Metrics & helpers
# -----------------------
def token_accuracy(logits: torch.Tensor, tgt_out: torch.Tensor, ignore_index: int) -> float:
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
    text = str(text)
    out: List[str] = []
    buf = ""
    in_quote = False
    quote_char = None
    for ch in text:
        if ch in ("'", '"'):
            if not in_quote:
                in_quote = True; quote_char = ch; buf = ""
            else:
                if ch == quote_char:
                    in_quote = False
                    if buf.strip(): out.append(buf.strip())
                    buf = ""
                else:
                    buf += ch
        elif in_quote:
            buf += ch
    if out: return out
    return [p.strip() for p in text.split(",") if p.strip()]


def _fallback_word_tokens(text: str) -> List[str]:
    return [t for t in str(text).replace("\n", " ").split(" ") if t.strip()]


def tokens_for_metrics(pred_text: str, gold_text: str) -> Tuple[List[str], List[str]]:
    pred_ing = parse_ingredients(pred_text)
    gold_ing = parse_ingredients(gold_text)
    if pred_ing or gold_ing:
        return pred_ing, gold_ing
    return _fallback_word_tokens(pred_text), _fallback_word_tokens(gold_text)


def jaccard_from_texts(pred_text: str, gold_text: str) -> float:
    a = set(parse_ingredients(pred_text)); b = set(parse_ingredients(gold_text))
    if not a and not b: return 100.0
    if not a or not b: return 0.0
    return 100.0 * len(a & b) / len(a | b)


def cosine_similarity_from_texts(pred_text: str, gold_text: str) -> float:
    hyp, ref = tokens_for_metrics(pred_text, gold_text)
    if not hyp and not ref: return 100.0
    if not hyp or not ref: return 0.0
    ch, cr = Counter(hyp), Counter(ref)
    dot = sum(ch[t] * cr.get(t, 0) for t in ch)
    nh = math.sqrt(sum(v * v for v in ch.values()))
    nr = math.sqrt(sum(v * v for v in cr.values()))
    if nh == 0 or nr == 0: return 0.0
    return 100.0 * (dot / (nh * nr))


def _ngram_counts(seq: List[str], n: int) -> Counter:
    if n <= 0 or len(seq) < n: return Counter()
    return Counter(tuple(seq[i:i+n]) for i in range(len(seq) - n + 1))


def bleu_from_texts(pred_text: str, gold_text: str, max_n: int = 4) -> float:
    hyp, ref = tokens_for_metrics(pred_text, gold_text)
    len_h, len_r = len(hyp), len(ref)
    if len_h == 0 and len_r == 0: return 100.0
    if len_h == 0: return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ch = _ngram_counts(hyp, n); cr = _ngram_counts(ref, n)
        if sum(ch.values()) == 0: precisions.append(0.0); continue
        overlap = sum(min(cnt, cr.get(ng, 0)) for ng, cnt in ch.items())
        p_n = (overlap + 1.0) / (sum(ch.values()) + 1.0)  # add-1 smoothing
        precisions.append(p_n)
    gm = math.exp(sum(math.log(p) for p in precisions if p > 0) / max(1, len([p for p in precisions if p > 0])))
    bp = 1.0 if len_h > len_r else math.exp(1.0 - (len_r / max(1, len_h)))
    return 100.0 * bp * gm


def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0: return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l_f1_from_texts(pred_text: str, gold_text: str) -> float:
    hyp, ref = tokens_for_metrics(pred_text, gold_text)
    if not hyp and not ref: return 100.0
    if not hyp or not ref: return 0.0
    lcs = _lcs_len(hyp, ref)
    prec = lcs / len(hyp) if len(hyp) else 0.0
    rec  = lcs / len(ref) if len(ref) else 0.0
    if prec + rec == 0: return 0.0
    return 100.0 * (2 * prec * rec) / (prec + rec)


def stable_anchor_indices(df: pd.DataFrame, k=3) -> List[int]:
    titles = df["input"].astype(str).tolist()
    keyed = [(int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16), i) for i, t in enumerate(titles)]
    keyed.sort()
    return [i for _, i in keyed[:min(k, len(keyed))]]


# -----------------------
# Build model / optimizer
# -----------------------
def build_model_and_optim(sp: spm.SentencePieceProcessor) -> Tuple[nn.Module, AdamW, LambdaLR, int]:
    pad_id = sp.piece_to_id("<pad>")
    if pad_id is None or pad_id < 0: pad_id = 0
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
            return step / cfg.WARMUP_STEPS
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
    if len(src_ids) > cfg.MAX_SRC_LEN:
        src_ids = src_ids[:cfg.MAX_SRC_LEN]; src_ids[-1] = eos

    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, S]
    src_key_mask = (src == pad)

    memory = model.encode(src, src_key_padding_mask=src_key_mask)

    ys = torch.tensor([[bos]], dtype=torch.long, device=device)  # [1, 1]
    for _ in range(max_new_tokens):
        tgt_mask_raw = model.generate_square_subsequent_mask(ys.size(1), device=device)
        tgt_mask = tgt_mask_raw if tgt_mask_raw.dtype == torch.bool else (tgt_mask_raw == float("-inf"))
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
def train_one_epoch(model, loader, optimizer, scheduler, criterion, pad_id, scaler: Optional[GradScaler]) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (src, tgt, _, _) in enumerate(loader, 1):
        src = src.to(cfg.DEVICE, non_blocking=True)
        tgt = tgt.to(cfg.DEVICE, non_blocking=True)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:].contiguous()

        tgt_mask_raw = model.generate_square_subsequent_mask(tgt_in.size(1), device=src.device)
        tgt_mask = tgt_mask_raw if tgt_mask_raw.dtype == torch.bool else (tgt_mask_raw == float("-inf"))
        src_pad_mask = src.eq(pad_id)
        tgt_pad_mask = tgt_in.eq(pad_id)

        with autocast(enabled=cfg.USE_AMP):
            logits = model(
                src=src,
                tgt_in=tgt_in,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                tgt_mask=tgt_mask,
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss_accum = loss / cfg.GRAD_ACCUM_STEPS

        if scaler and cfg.USE_AMP:
            scaler.scale(loss_accum).backward()
        else:
            loss_accum.backward()

        if step % cfg.GRAD_ACCUM_STEPS == 0:
            if scaler and cfg.USE_AMP:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.CLIP_NORM)

            if scaler and cfg.USE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_acc += token_accuracy(logits, tgt_out, ignore_index=pad_id)
        n_batches += 1

        if cfg.LOG_EVERY_N_STEPS and (step % cfg.LOG_EVERY_N_STEPS == 0):
            print(f"  [train] step {step}/{len(loader)}  loss {loss.item():.4f}")

    # if last micro-batch didn't hit optimizer.step(), step the scheduler anyway (optional)

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, criterion, pad_id) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for src, tgt, _, _ in loader:
        src = src.to(cfg.DEVICE, non_blocking=True)
        tgt = tgt.to(cfg.DEVICE, non_blocking=True)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:].contiguous()

        tgt_mask_raw = model.generate_square_subsequent_mask(tgt_in.size(1), device=src.device)
        tgt_mask = tgt_mask_raw if tgt_mask_raw.dtype == torch.bool else (tgt_mask_raw == float("-inf"))
        src_pad_mask = src.eq(pad_id)
        tgt_pad_mask = tgt_in.eq(pad_id)

        with autocast(enabled=cfg.USE_AMP):
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


@torch.no_grad()
def evaluate_text_metrics_epoch(model, sp, df: pd.DataFrame, device, max_new_tokens: int,
                                sample_size: int, seed: int) -> dict:
    n = len(df)
    if n == 0:
        return {"Jaccard": 0.0, "Cosine": 0.0, "ROUGE-L": 0.0, "BLEU-4": 0.0, "EM": 0.0}

    if sample_size is None or sample_size < 0 or sample_size >= n:
        indices = list(range(n))
    else:
        rng = random.Random(seed)
        indices = rng.sample(range(n), k=min(sample_size, n))

    sums = {"Jaccard": 0.0, "Cosine": 0.0, "ROUGE-L": 0.0, "BLEU-4": 0.0, "EM": 0.0}
    count = 0

    for i in indices:
        title = df.iloc[i]["input"]
        gold_text = df.iloc[i]["target"]
        try:
            pred_text = greedy_generate(model, sp, str(title), device, max_new_tokens=max_new_tokens)

            jac  = jaccard_from_texts(pred_text, gold_text)
            cos  = cosine_similarity_from_texts(pred_text, gold_text)
            rl   = rouge_l_f1_from_texts(pred_text, gold_text)
            bleu = bleu_from_texts(pred_text, gold_text, max_n=4)
            em   = 100.0 if str(pred_text).strip() == str(gold_text).strip() else 0.0

            sums["Jaccard"] += jac; sums["Cosine"] += cos; sums["ROUGE-L"] += rl; sums["BLEU-4"] += bleu; sums["EM"] += em
            count += 1
        except Exception:
            continue

    if count == 0:
        return {"Jaccard": 0.0, "Cosine": 0.0, "ROUGE-L": 0.0, "BLEU-4": 0.0, "EM": 0.0}
    return {k: v / count for k, v in sums.items()}


def fit():
    # Load SentencePiece
    if not os.path.exists(cfg.SPM_MODEL):
        raise FileNotFoundError(f"SentencePiece model not found at '{cfg.SPM_MODEL}'")
    sp = spm.SentencePieceProcessor(model_file=cfg.SPM_MODEL)

    # Load single dataset and split
    if not os.path.exists(cfg.DATA_CSV):
        raise FileNotFoundError(f"Data CSV not found at '{cfg.DATA_CSV}'")
    full_df = pd.read_csv(cfg.DATA_CSV)[["input", "target"]].dropna()
    full_df = full_df.sample(frac=1.0, random_state=cfg.SHUFFLE_SEED).reset_index(drop=True)

    # Safe bounds on val fraction
    val_frac = min(max(cfg.VAL_FRACTION, 0.0), 0.5)
    split_idx = int(len(full_df) * (1.0 - val_frac))
    train_df = full_df.iloc[:split_idx].copy()
    val_df   = full_df.iloc[split_idx:].copy()

    # Optional quick-smoke subsample
    if cfg.MAX_TRAIN_SAMPLES > 0:
        train_df = train_df.iloc[:cfg.MAX_TRAIN_SAMPLES].copy()
    if cfg.MAX_VAL_SAMPLES > 0:
        val_df = val_df.iloc[:cfg.MAX_VAL_SAMPLES].copy()

    # Datasets
    train_ds = RecipeDatasetSP(train_df, sp)
    val_ds   = RecipeDatasetSP(val_df, sp)
    pad_id = train_ds.pad_id

    # Show composition
    n_steps = int(val_ds.df["input"].astype(str).str.startswith("<STEPS>").sum())
    n_ingr  = int(val_ds.df["input"].astype(str).str.startswith("<INGR>").sum())
    print(f"Val split rows -> INGREDIENTS: {n_ingr} | STEPS: {n_steps}")

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )

    # Model / Optim
    model, optimizer, scheduler, pad_id_m = build_model_and_optim(sp)
    assert pad_id_m == pad_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=cfg.LABEL_SMOOTHING)
    scaler = GradScaler(enabled=cfg.USE_AMP)

    # Choose fixed anchors for per-epoch example prints
    anchor_idx = stable_anchor_indices(val_ds.df, k=cfg.LOG_EXAMPLES)

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, pad_id, scaler)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, pad_id)
        dt = time.time() - t0

        ppl = perplexity(val_loss)
        lr_now = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f} | "
              f"Acc: {val_acc:.2f}% | Time: {dt:.1f}s | LR: {lr_now:.6g}")

        # ----- Epoch-level text metrics (generation-based) -----
        metrics_avg = evaluate_text_metrics_epoch(
            model=model, sp=sp, df=val_ds.df, device=cfg.DEVICE,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            sample_size=cfg.METRIC_EVAL_SAMPLES,
            seed=cfg.METRIC_EVAL_SEED + epoch
        )
        print("  Epoch Text Metrics (avg over sampled val): "
              f"Jaccard: {metrics_avg['Jaccard']:.2f}% | Cosine: {metrics_avg['Cosine']:.2f}% | "
              f"ROUGE-L(F1): {metrics_avg['ROUGE-L']:.2f}% | BLEU-4: {metrics_avg['BLEU-4']:.2f}% | "
              f"EM: {metrics_avg['EM']:.2f}%")

        # ----- Fixed anchors for qualitative sanity -----
        try:
            indices = list(anchor_idx)
            pool = [i for i in range(len(val_ds)) if i not in anchor_idx]
            rot_needed = max(0, cfg.LOG_EXAMPLES - len(indices))
            rng = random.Random(cfg.SHUFFLE_SEED + epoch)
            indices += rng.sample(pool, k=min(rot_needed, len(pool)))

            for j, i in enumerate(indices, 1):
                title = val_ds.df.iloc[i]["input"]
                gold_text = val_ds.df.iloc[i]["target"]
                pred_text = greedy_generate(model, sp, title, cfg.DEVICE, max_new_tokens=cfg.MAX_NEW_TOKENS)

                jac  = jaccard_from_texts(pred_text, gold_text)
                cos  = cosine_similarity_from_texts(pred_text, gold_text)
                rl   = rouge_l_f1_from_texts(pred_text, gold_text)
                bleu = bleu_from_texts(pred_text, gold_text, max_n=4)
                em   = 100.0 if pred_text.strip() == str(gold_text).strip() else 0.0

                print(f"  Example {j}:")
                print(f"    Title: {title}")
                print(f"    Pred:  {pred_text}")
                print(f"    Gold:  {gold_text}")
                print(f"    Metrics -> Jaccard: {jac:.2f}% | Cosine: {cos:.2f}% | ROUGE-L(F1): {rl:.2f}% | BLEU-4: {bleu:.2f}% | EM: {em:.2f}%")
        except Exception as e:
            print(f"(Skipping example print due to error: {e})")

    return model


if __name__ == "__main__":
    fit()
