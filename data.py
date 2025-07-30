import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
  def __init__(self, data, vocab, max_len, encode_fn):
    self.data = data
    self.vocab = vocab
    self.max_len = max_len
    self.encode = encode_fn

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    tokens, label = self.data[index]
    token_ids = self.encode(tokens, self.vocab, self.max_len)
    return torch.tensor(token_ids), torch.tensor(label, dtype=torch.long)