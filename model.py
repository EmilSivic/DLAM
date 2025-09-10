# model.py

import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, input_ids, input_lengths):
        """
        input_ids: Tensor of shape (batch_size, seq_len)
        input_lengths: Tensor of shape (batch_size,)
        """
        # 1. Embed tokens
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        # 2. Pack padded sequence for efficient LSTM
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # 3. Pass through LSTM
        outputs, (hidden, cell) = self.lstm(packed)

        # 4. Return final hidden and cell states
        return hidden, cell


import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()

        # 1. Embedding layer: map token indices to embeddings
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)

        # 2. LSTM: takes embeddings + hidden/cell states, returns new hidden/cell
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # 3. Linear layer: project hidden state → vocab size
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, input_token, hidden, cell):
        """
        input_token: (batch_size)       - one token per sample
        hidden: (num_layers, batch_size, hidden_dim)
        cell:   (num_layers, batch_size, hidden_dim)
        """

        # TODO 1: Embed the input token (hint: shape goes from [batch] → [batch, 1, embedding_dim])
        embedded = self.embedding(input_token).unsqueeze(1)

        # TODO 2: Pass embedding + hidden, cell into the LSTM
        output, (hidden,cell) = self.lstm(embedded, (hidden, cell))

        # TODO 3: Take the output from LSTM (not hidden), pass through fc_out to get vocab logits
        output = output.squeeze(1)
        predictions = self.fc_out(output)

        # TODO 4: Return predictions, hidden, cell
        return predictions, hidden, cell
