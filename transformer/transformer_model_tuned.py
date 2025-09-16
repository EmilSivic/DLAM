import torch
import torch.nn as nn
from torch.nn import Transformer

class Seq2SeqTransformerTuned(nn.Module):
    def __init__(self, src_vocab, tgt_vocab,
                 d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_ff=2048, dropout=0.3,
                 pad_idx=0, tie_weights=True):
        super().__init__()

        self.embedding_dim = d_model
        self.hidden_dim = dim_ff
        self.num_layers = num_encoder_layers
        self.dropout = dropout
        self.pad_idx = pad_idx

        # gemeinsame Embedding-Matrix für Encoder & Decoder
        self.shared_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)

        self.pos_encoder = nn.Embedding(5000, d_model)  # einfache Positional Embedding
        self.pos_decoder = nn.Embedding(5000, d_model)

        self.transformer = Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )

        # Decoder Output Layer
        self.generator = nn.Linear(d_model, tgt_vocab, bias=False)

        if tie_weights:
            # Embedding-Weights mit Output teilen
            self.generator.weight = self.shared_emb.weight

    def forward(self, src, tgt,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_mask=None):

        src_pos = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(0, tgt.size(1), device=tgt.device).unsqueeze(0)

        src_emb = self.shared_emb(src) + self.pos_encoder(src_pos)
        tgt_emb = self.shared_emb(tgt) + self.pos_decoder(tgt_pos)

        out = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask
        )
        return self.generator(out)
