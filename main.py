import pandas as pd
from ast import literal_eval
from dataset import RecipeDataset, collate_fn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import EncoderRNN, DecoderRNN
import os

# ----------------------
# Config
# ----------------------
DEFAULT_DATA_PATH = "data/processed_recipes.csv"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAD_CLIP = 1.0


# ----------------------
# Seq2Seq Wrapper
# ----------------------
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

        hidden, cell = self.encoder(src, src_length)
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        input_token = trg[:, 0].to(self.device)  # <SOS>

        for t in range(1, trg_len):
            step_logits, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = step_logits

            use_tf = (torch.rand(1).item() < teacher_forcing_ratio)
            if use_tf:
                input_token = trg[:, t].long()
            else:
                input_token = step_logits.argmax(dim=1)

        return outputs


# ----------------------
# Greedy Decoding
# ----------------------
def greedy_decode_one(model, dataset, title_ids, title_len, max_len=15):
    model.eval()
    with torch.no_grad():
        hidden, cell = model.encoder(title_ids.unsqueeze(0).to(DEVICE),
                                     title_len.unsqueeze(0))

        sos_idx = dataset.target_vocab.word2idx["<SOS>"]
        eos_idx = dataset.target_vocab.word2idx["<EOS>"]
        input_token = torch.tensor([sos_idx], device=DEVICE)

        out_tokens = []
        for _ in range(max_len):
            logits, hidden, cell = model.decoder(input_token, hidden, cell)
            next_id = logits.argmax(dim=1)
            tok = int(next_id.item())

            if tok == eos_idx:
                break
            word = dataset.target_vocab.idx2word.get(tok, "<UNK>")
            out_tokens.append(word)

            input_token = next_id

    return out_tokens


# ----------------------
# Training Loop
# ----------------------
def train(model, dataloader, optimizer, criterion, dataset, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            src = batch["input_ids"].to(DEVICE)
            trg = batch["target_ids"].to(DEVICE)
            src_lengths = batch["input_lengths"]

            optimizer.zero_grad()
            logits = model(src, trg, src_lengths, teacher_forcing_ratio=0.5)

            logits_flat = logits[:, 1:, :].contiguous().view(-1, logits.size(-1))
            targets_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        ppl = torch.exp(torch.tensor(avg_loss))
        print(f"Epoch {epoch+1}, avg loss {avg_loss:.4f}, perplexity {ppl:.2f}")

        # Debug prediction
        sample_title = batch["input_ids"][0]
        sample_len = batch["input_lengths"][0]
        prediction = greedy_decode_one(model, dataset, sample_title, sample_len)

        ground_truth = batch["target_ids"][0].tolist()
        ground_truth_words = [
            dataset.target_vocab.idx2word[idx]
            for idx in ground_truth
            if idx not in [dataset.target_vocab.word2idx["<PAD>"]]
        ]

        print("Predicted:", prediction)
        print("Ground truth:", ground_truth_words)


# ----------------------
# Main entrypoint
# ----------------------
if __name__ == "__main__":
    dataset = RecipeDataset(DATA_PATH)

    subset = torch.utils.data.Subset(dataset, range(2000))  # nur 2k Beispiele
    dataloader = DataLoader(subset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    encoder = EncoderRNN(len(dataset.input_vocab), 128, 256, 1)
    decoder = DecoderRNN(len(dataset.target_vocab), 128, 256, 1)

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=DEVICE,
        sos_idx=dataset.target_vocab.word2idx["<SOS>"],
        pad_idx=dataset.target_vocab.word2idx["<PAD>"]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.target_vocab.word2idx["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, dataloader, optimizer, criterion, dataset, num_epochs=10)
