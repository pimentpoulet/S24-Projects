import torch as t
import torch.nn as nn

from torch.nn import functional

from get_data import *
from functions import *


# encode the entire text dataset and store it in a torch.Tensor
data = t.tensor(encode(text), dtype=t.long)

# split data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

t.manual_seed(1652)
BATCH_SIZE = 4
BLOCK_SIZE = 8

xb, yb = get_batch('train',
                   train_data=train_data,
                   val_data=val_data,
                   block_size=BLOCK_SIZE,
                   batch_size=BATCH_SIZE)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx)    # (B,T,C) --> Batch (4), Time (8), Channel (65)

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = functional.cross_entropy(logits, targets)

        return logits, loss


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)




