import math
import torch
import torch.nn as nn

def generate_square_subsequent_mask(sz, device):
    # mask future
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

def key_padding_mask(tokens, pad_idx):
    # true for pads
    return tokens.eq(pad_idx)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Seq2SeqTransformerTuned(nn.Module):
    def __init__(self, src_vocab, tgt_vocab,
                 d_model=256, nhead=4,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_ff=1024, dropout=0.2, pad_idx=0):
        super().__init__()

        # Embed + positional
        self.src_tok_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )

        # Output layer (weight tying mit tgt embedding)
        self.generator = nn.Linear(d_model, tgt_vocab, bias=False)
        self.generator.weight = self.tgt_tok_emb.weight

        self.embedding_dim = d_model
        self.hidden_dim = d_model
        self.num_layers = num_encoder_layers + num_decoder_layers
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.d_model = d_model

    def _encode(self, src):
        src_key_mask = key_padding_mask(src, self.pad_idx)
        src_emb = self.pos_enc(self.src_tok_emb(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src=src_emb,
            src_key_padding_mask=src_key_mask
        )
        return memory, src_key_mask

    def _decode(self, tgt_inp, memory, src_key_mask):
        tgt_key_mask = key_padding_mask(tgt_inp, self.pad_idx)
        tgt_mask = generate_square_subsequent_mask(tgt_inp.size(1), tgt_inp.device)
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt_inp) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt=tgt_emb, memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_mask,
            memory_key_padding_mask=src_key_mask
        )
        return self.generator(out)

    def forward(self, src, trg, src_lengths=None, teacher_forcing_ratio=0.0):
        memory, src_key_mask = self._encode(src)
        trg_in = trg[:, :-1]
        logits = self._decode(trg_in, memory, src_key_mask)
        B, Tm1, V = logits.shape
        out = logits.new_zeros((B, Tm1+1, V))
        out[:, 1:, :] = logits
        return out

    @torch.no_grad()
    def greedy_or_topk(self, src, max_len=20, sos=None, eos=None, k=1):
        device = src.device
        batch_size = src.size(0)

        if sos is None:
            sos = getattr(self, "sos_idx", None)
            if sos is None:
                raise ValueError("SOS-Index fehlt. Bitte sos übergeben oder self.sos_idx setzen.")
        if eos is None:
            eos = getattr(self, "eos_idx", None)

        ys = torch.full((batch_size, 1), sos, dtype=torch.long, device=device)

        for _ in range(max_len-1):
            out = self.forward(src, ys)
            next_token_logits = out[:, -1, :]

            if k > 1:
                topk_probs, topk_idx = torch.topk(torch.softmax(next_token_logits, dim=-1), k)
                next_tokens = topk_idx[torch.arange(batch_size), torch.multinomial(topk_probs, 1).squeeze(-1)]
            else:
                next_tokens = next_token_logits.argmax(dim=-1)

            ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)

            if eos is not None and (ys[:, -1] == eos).all():
                break

        return ys
