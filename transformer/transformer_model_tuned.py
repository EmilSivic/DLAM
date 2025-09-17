import math
import torch
import torch.nn as nn
from typing import Optional


class Seq2SeqTransformer(nn.Module):
    """
    A thin, practical wrapper around nn.Transformer with:
      - token + positional embeddings
      - convenience encode/decode
      - final generator (linear to vocab)
      - batch_first=True everywhere
    """
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.model_type = "Transformer"
        self.pad_idx = pad_idx

        # Embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)

        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output head
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    # ---- Convenience helpers ----
    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Standard masked (no-peek) causal mask for decoder inputs."""
        mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
        return mask if device is None else mask.to(device)

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out  # return hidden states (apply generator outside)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [B, S]
            tgt_in: [B, T] (teacher-forced input, shifted right)
        Returns:
            logits over tgt vocab: [B, T, V]
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_in))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        return self.generator(out)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000) -> None:
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * (math.log(10000.0) / emb_size))
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # [1, L, D]
        self.register_buffer("pos_embedding", pos_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        # token_emb: [B, L, D]
        return self.dropout(token_emb + self.pos_embedding[:, : token_emb.size(1), :])


# Backward-compatible alias
Seq2SeqTransformerTuned = Seq2SeqTransformer
