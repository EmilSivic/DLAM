import torch
import torch.nn as nn
import math


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers,
                 emb_size, nhead, src_vocab_size, tgt_vocab_size,
                 dim_feedforward=512, dropout=0.1, pad_idx=0):
        super(Seq2SeqTransformer, self).__init__()

        self.model_type = "Transformer"
        self.pad_idx = pad_idx

        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output layer
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

        # Embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=pad_idx)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

        # Initialize
        self._reset_parameters()

        # Store hyperparams for logging
        self.embedding_dim = emb_size
        self.hidden_dim = dim_feedforward
        self.num_layers = num_encoder_layers
        self.dropout = dropout

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int, device):
        """Generate a triangular causal mask for the decoder."""
        mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, *args, **kwargs):
        device = src.device
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        tgt_seq_len = tgt.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)

        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)

        outs = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        return self.generator(outs)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])


# === Alias for tuned variant ===
Seq2SeqTransformerTuned = Seq2SeqTransformer
