class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.emb_dropout = nn.Dropout(dropout)   # dropout on embeddings
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, input_ids, input_lengths):
        embedded = self.embedding(input_ids)
        embedded = self.emb_dropout(embedded)    #apply dropout
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.lstm(packed)
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

    def forward(self, input_token, hidden, cell):
        embedded = self.embedding(input_token).unsqueeze(1)
        embedded = self.emb_dropout(embedded)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = output.squeeze(1)
        output = self.out_dropout(output)        # dropout before final projection
        predictions = self.fc_out(output)
        return predictions, hidden, cell
