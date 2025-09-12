import math
import torch
import torch.nn as nn

def generate_square_subsequent_mask(sz, device):
    # maskiert zukunft im decoder
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

def key_padding_mask(tokens, pad_idx):
    # true bei pads
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

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4, dim_ff=2048,
                 dropout=0.1, pad_idx=0):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )
        self.generator = nn.Linear(d_model, tgt_vocab)

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
    def greedy_or_topk(self, src, max_len, sos_idx, eos_idx, k=1):
        memory, src_key_mask = self._encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), sos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits = self._decode(ys, memory, src_key_mask)
            step_logits = logits[:, -1, :]
            if k == 1:
                next_token = step_logits.argmax(-1)
            else:
                probs = torch.softmax(step_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(eos_idx)
            if finished.all():
                break
        return ys
