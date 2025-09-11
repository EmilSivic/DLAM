# model.py

import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, input_ids, input_lengths):
        """
        input_ids: Tensor of shape (batch_size, seq_len)
        input_lengths: Tensor of shape (batch_size,)
        """
        # Embed tokens
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        outputs, (hidden, cell) = self.lstm(packed)
        return hidden, cell


import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.4):
        super().__init__()

        # Embedding (traininng of word representation)
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)

        # LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # projection onto output_vocab_size
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, input_token, hidden, cell):
        """
        input_token: (batch_size)       - one token per sample
        hidden: (num_layers, batch_size, hidden_dim)
        cell:   (num_layers, batch_size, hidden_dim)
        """


        embedded = self.embedding(input_token).unsqueeze(1)
        output, (hidden,cell) = self.lstm(embedded, (hidden, cell))
        output = output.squeeze(1)
        predictions = self.fc_out(output)
        return predictions, hidden, cell
