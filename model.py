import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)   # dropout on embeddings
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

        # falls bidirectional: Linear-Layer um hidden_dim*2 -> hidden_dim zu reduzieren
        if bidirectional:
            self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim)
            self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim)

        # save hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, input_ids, input_lengths):
        embedded = self.embedding(input_ids)
        embedded = self.emb_dropout(embedded)    # apply dropout
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.lstm(packed)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            cell = torch.cat((cell[-2], cell[-1]), dim=1)
            hidden = self.reduce_h(hidden).unsqueeze(0)
            cell = self.reduce_c(cell).unsqueeze(0)

            # expand if more than one layer
            hidden = hidden.repeat(self.num_layers, 1, 1)
            cell = cell.repeat(self.num_layers, 1, 1)
        return hidden, cell


class DecoderRNN(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)   # dropout on embeddings
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)
        self.out_dropout = nn.Dropout(dropout)   # dropout on projection

        # save hyperparams
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, input_token, hidden, cell):
        embedded = self.embedding(input_token).unsqueeze(1)
        embedded = self.emb_dropout(embedded)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = output.squeeze(1)
        output = self.out_dropout(output)        # dropout before final projection
        predictions = self.fc_out(output)
        return predictions, hidden, cell
