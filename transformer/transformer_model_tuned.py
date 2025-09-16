import torch
import torch.nn as nn
import math


class Seq2SeqTransformerTuned(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward=512,
        dropout=0.1,
        tie_weights=False,
        pad_idx=0,
    ):
        super().__init__()

        self.src_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=pad_idx)

        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        self.pos_decoder = PositionalEncoding(emb_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(emb_size, tgt_vocab_size, bias=False)

        # ---- Nur Decoder-Embeddings ↔ Output teilen ----
        if tie_weights:
            self.fc_out.weight = self.tgt_emb.weight

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.pos_encoder(self.src_emb(src))
        return self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.pos_decoder(self.tgt_emb(tgt))
        return self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                teacher_forcing_ratio=None):  # kompatibel mit logger
        memory = self.encode(src, src_mask, src_key_padding_mask)
        outs = self.decode(tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.fc_out(outs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
