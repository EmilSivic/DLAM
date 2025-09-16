import torch
import torch.nn as nn
from torch.nn import Transformer

class Seq2SeqTransformerTuned(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        src_pad_idx=0,
        tgt_pad_idx=0,
        tie_weights=False
    ):
        super().__init__()
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)

        self.pos_encoder = nn.Embedding(5000, d_model)
        self.pos_decoder = nn.Embedding(5000, d_model)

        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        if tie_weights:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("tie_weights=True nur möglich, wenn src_vocab_size == tgt_vocab_size")
            self.generator.weight = self.tgt_embedding.weight

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):

        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)

        src_pos = torch.arange(0, src_seq_len, device=src.device).unsqueeze(0).expand_as(src)
        tgt_pos = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(0).expand_as(tgt)

        src_emb = self.src_embedding(src) + self.pos_encoder(src_pos)
        tgt_emb = self.tgt_embedding(tgt) + self.pos_decoder(tgt_pos)

        memory = self.transformer.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask
        )
        outs = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.generator(outs)
