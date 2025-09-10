import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from nltk.tokenize import word_tokenize

class Vocabulary:
    def __init__(self, specials=["<PAD>", "<SOS>", "<EOS>", "<UNK>"]):
        self.word2idx = {}
        self.idx2word = {}
        self.specials = specials
        self.build_done = False
    def build(self, token_lists):
        all_tokens = set(token for tokens in token_lists for token in tokens)
        for i, token in enumerate(self.specials +sorted(all_tokens)):
            self.word2idx[token] = i
            self.idx2word[i] = token
        self.build_done = True


    def encode(self, tokens):
        return [self.word2idx.get(tok, self.word2idx["<UNK>"]) for tok in tokens]

    def decode(self, indices):
        return [self.idx2word.get(idx, "<UNK>") for idx in indices]

    def __len__(self):
        return len(self.word2idx)

class RecipeDataset(Dataset):
    def __init__(self, csv_path, input_vocab=None, target_vocab=None, max_len=30):
        df = pd.read_csv(csv_path)
        df["target"] = df["target"].apply(literal_eval)

        self.inputs = [title.lower().split() for title in df["input"]]
        self.targets = [ingredients for ingredients in df["target"]]

        # Build vocab if not provided
        if input_vocab is None:
            self.input_vocab = Vocabulary()
            self.input_vocab.build(self.inputs)
        else:
            self.input_vocab = input_vocab

        if target_vocab is None:
            self.target_vocab = Vocabulary()
            self.target_vocab.build(self.targets)
        else:
            self.target_vocab = target_vocab

        self.max_len = max_len  # for padding

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tokens = self.inputs[idx][:self.max_len]
        target_tokens = self.targets[idx][:self.max_len]

        # Add SOS and EOS for target
        target_tokens = ["<SOS>"] + target_tokens + ["<EOS>"]

        input_ids = self.input_vocab.encode(input_tokens)
        target_ids = self.target_vocab.encode(target_tokens)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]

    # Pad sequences to max length in batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)   # 0 = <PAD>
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)

    input_lengths = torch.tensor([len(seq) for seq in input_ids], dtype=torch.long)
    target_lengths = torch.tensor([len(seq) for seq in target_ids], dtype=torch.long)

    return {
        "input_ids": input_ids_padded,
        "input_lengths": input_lengths,
        "target_ids": target_ids_padded,
        "target_lengths": target_lengths
    }

