import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
  def __init__(self, embedded, out_channels, num_class, dropout_rate=0.5):
    super(TextCNN, self).__init__()
    vocab_size, embedding_dim = embedded.shape

    self.embedding = nn.Embedding.from_pretrained(
        embedded, freeze=False, padding_idx=0
    )

    self.conv2 = nn.ModuleList([nn.Conv2d(1, out_channels, kernel_size=(k, embedding_dim)) for k in [3, 4, 5]])

    self.dropout = nn.Dropout(dropout_rate)

    self.fc = nn.Linear(len([3, 4, 5]) * out_channels, num_class)

  def forward(self, x):

    x = self.embedding(x)

    x = x.unsqueeze(1)

    conv_x = [F.relu(conv(x)).squeeze(3) for conv in self.conv2]

    pool_x = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in conv_x]

    x = torch.cat(pool_x, 1)

    x = self.dropout(x)

    logit = self.fc(x)

    return logit