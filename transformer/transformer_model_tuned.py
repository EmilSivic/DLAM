import torch
import torch.nn as nn


class Seq2SeqTransformerTuned(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_ff=1024,
        dropout=0.1,
        pad_idx=0,
        tie_weights=False   # <--- hinzugefügt
    ):
        super().__init__()
        self.embedding_dim = d_model
        self.hidden_dim = d_model
        self.num_layers = num_encoder_layers
        self.dropout = dropout

        # Embeddings
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # Encoder & Decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output Layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Gewichtssharing
        if tie_weights:
            if src_vocab_size == tgt_vocab_size:
                self.fc_out.weight = self.tgt_emb.weight
            else:
                raise ValueError("tie_weights=True nur möglich, wenn src_vocab_size == tgt_vocab_size")

        self.pad_idx = pad_idx

    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=None):
        # teacher_forcing_ratio ignoriert, nur für Kompatibilität
        src_emb = self.src_emb(src)
        tgt_emb = self.tgt_emb(tgt)

        # Encoder
        memory = self.encoder(src_emb)

        # Decoder
        out = self.decoder(tgt_emb, memory)

        # Projection to vocab
        logits = self.fc_out(out)
        return logits
