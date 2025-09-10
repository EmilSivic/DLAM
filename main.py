import pandas as pd
from ast import literal_eval
from dataset import RecipeDataset, collate_fn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import EncoderRNN, DecoderRNN
import os
from dataset import RecipeDataset, collate_fn

DEFAULT_DATA_PATH = "data/processed_recipes.csv"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)

dataset = RecipeDataset(DATA_PATH)

# dataset = RecipeDataset("data/processed_recipes.csv")

# Subset für schnelles Training
subset = torch.utils.data.Subset(dataset, range(2000))
dataloader = DataLoader(subset, batch_size=8, shuffle=True, collate_fn=collate_fn)

print(f"Dataset size: {len(dataset)}")
print("Input vocab size:", len(dataset.input_vocab))
print("Target vocab size:", len(dataset.target_vocab))

# -------------------------------
# Encoder & Decoder
# -------------------------------
encoder = EncoderRNN(
    input_vocab_size=len(dataset.input_vocab),
    embedding_dim=128,
    hidden_dim=256,
    num_layers=1
)

decoder = DecoderRNN(
    output_vocab_size=len(dataset.target_vocab),
    embedding_dim=128,
    hidden_dim=256,
    num_layers=1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Greedy Decode Funktion
# -------------------------------
def greedy_decode_one(model, title_ids, title_len, max_len=15):
    model.eval()
    with torch.no_grad():
        hidden, cell = model.encoder(title_ids.unsqueeze(0).to(device),
                                     title_len.unsqueeze(0))

        sos_idx = dataset.target_vocab.word2idx["<SOS>"]
        eos_idx = dataset.target_vocab.word2idx["<EOS>"]
        input_token = torch.tensor([sos_idx], device=device)

        out_tokens = []
        for _ in range(max_len):
            logits, hidden, cell = model.decoder(input_token, hidden, cell)
            next_id = logits.argmax(dim=1)  # [1]
            tok = int(next_id.item())

            if tok == eos_idx:
                break

            word = dataset.target_vocab.idx2word.get(tok, "<UNK>")
            out_tokens.append(word)

            input_token = next_id

    return out_tokens

# -------------------------------
# Seq2Seq Modell
# -------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_idx, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        self.pad_idx = pad_idx

    def forward(self, src, trg, src_length, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.size(0), trg.size(1)
        vocab_size = self.decoder.fc_out.out_features

        # Encode
        hidden, cell = self.encoder(src, src_length)

        # Output-Container
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        # Start-Token
        input_token = trg[:, 0]  # <SOS>
        input_token = input_token.to(self.device)

        # Decoder Loop
        for t in range(1, trg_len):
            step_logits, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = step_logits

            use_tf = (torch.rand(1).item() < teacher_forcing_ratio)
            if use_tf:
                input_token = trg[:, t].to(self.device).long()
            else:
                input_token = step_logits.argmax(dim=1)

        return outputs

# -------------------------------
# Training Setup
# -------------------------------
model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    device=device,
    sos_idx=dataset.target_vocab.word2idx["<SOS>"],
    pad_idx=dataset.target_vocab.word2idx["<PAD>"]
).to(device)

pad_idx = dataset.target_vocab.word2idx["<PAD>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
grad_clip = 1.0

# -------------------------------
# Training Loop
# -------------------------------
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch["input_ids"].to(device)
        trg = batch["target_ids"].to(device)
        src_lengths = batch["input_lengths"]

        optimizer.zero_grad()
        logits = model(src, trg, src_lengths, teacher_forcing_ratio=0.5)

        logits_flat  = logits[:, 1:, :].contiguous().view(-1, logits.size(-1))
        targets_flat = trg[:, 1:].contiguous().view(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    ppl = torch.exp(torch.tensor(avg_loss))
    print(f"Epoch {epoch+1}, avg loss {avg_loss:.4f}, perplexity {ppl:.2f}")

    #Prediction vs Ground Truth nach jeder Epoche
    sample_title = batch["input_ids"][0]
    sample_len = batch["input_lengths"][0]
    prediction = greedy_decode_one(model, sample_title, sample_len)

    ground_truth = batch["target_ids"][0].tolist()
    ground_truth_words = [
        dataset.target_vocab.idx2word[idx]
        for idx in ground_truth
        if idx not in [dataset.target_vocab.word2idx["<PAD>"]]
    ]

    print("Predicted:", prediction)
    print("Ground truth:", ground_truth_words)