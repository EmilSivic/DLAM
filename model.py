import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------
# Encoder
# --------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.3, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # falls bidirectional: hidden_dim verdoppelt -> reduzieren
        if bidirectional:
            self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim)
            self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim)

        # Hyperparams speichern
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, input_ids, input_lengths):
        embedded = self.emb_dropout(self.embedding(input_ids))  # [B, T, E]

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # [B, T, H*num_directions]

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [B, 2H]
            cell = torch.cat((cell[-2], cell[-1]), dim=1)        # [B, 2H]
            hidden = self.reduce_h(hidden).unsqueeze(0)          # [1, B, H]
            cell = self.reduce_c(cell).unsqueeze(0)              # [1, B, H]

            # bei num_layers > 1 einfach wiederholen
            hidden = hidden.repeat(self.num_layers, 1, 1)
            cell = cell.repeat(self.num_layers, 1, 1)

        return outputs, hidden, cell  # outputs wichtig für Attention


# --------------------
# Luong Attention
# --------------------
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim, enc_dim=None):
        super().__init__()
        if enc_dim is None:
            enc_dim = hidden_dim
        self.attn = nn.Linear(enc_dim, hidden_dim, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: [B, H]
        # encoder_outputs: [B, src_len, enc_dim]
        proj_enc = self.attn(encoder_outputs)  # [B, src_len, H]

        scores = torch.bmm(proj_enc, decoder_hidden.unsqueeze(2)).squeeze(2)  # [B, src_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=1)  # [B, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), proj_enc).squeeze(1)  # [B, H]

        return context, attn_weights



# decoder with Attention
class DecoderRNN(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, enc_dim=None, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = LuongAttention(hidden_dim, enc_dim=enc_dim or hidden_dim)

        self.fc_out = nn.Linear(hidden_dim * 2, output_vocab_size)
        self.out_dropout = nn.Dropout(dropout)


        # Hyperparams speichern
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        embedded = self.emb_dropout(self.embedding(input_token).unsqueeze(1))  # [B, 1, E]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # [B, 1, H]
        output = output.squeeze(1)  # [B, H]

        context, attn_weights = self.attention(output, encoder_outputs, mask)  # [B, H], [B, src_len]

        combined = torch.cat((output, context), dim=1)  # [B, 2H]
        combined = self.out_dropout(combined)

        predictions = self.fc_out(combined)  # [B, vocab_size]
        return predictions, hidden, cell, attn_weights
