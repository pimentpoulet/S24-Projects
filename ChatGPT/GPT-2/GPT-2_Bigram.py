import torch as t
import torch.nn as nn

from torch.nn import functional

from gpt2_functions import *
from get_data import *


""" HYPER PARAMETERS """

batch_size = 4
block_size = 8
max_iters = 10_000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if t.cuda.is_available() else "cpu"
eval_iters = 200

t.manual_seed(1652)


""" GET DATA """

with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# get all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping characters <--> integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


""" TRAIN AND TEST SPLITS """

# encode the entire text dataset and store it in a torch.Tensor
data = t.tensor(encode(text), dtype=t.long)

# split data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


""" BIGRAM MODEL CLASS"""


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # B, T, C

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample from the distribution
            idx_next = t.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = t.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)


""" OPTIMIZER """

optimizer = t.optim.AdamW(m.parameters(), lr=1e-3)


""" TRAINING LOOP """

for iter in range(max_iters):

    # evaluate the loss on train and val splits
    if iter % eval_interval == 0:
        losses = estimate_loss(train_data=train_data,
                               val_data=val_data,
                               block_size=block_size,
                               batch_size=batch_size,
                               model=m, eval_iters=eval_iters)
        print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train",
                       train_data=train_data,
                       val_data=val_data,
                       block_size=block_size,
                       batch_size=batch_size,
                       device=device)

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


""" GENERATE FROM THE MODEL """

context = t.zeros((1, 1), dtype=t.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
