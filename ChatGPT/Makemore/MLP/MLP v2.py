import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from functions import *
from datasets import *


# get data
input_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'datasets', 'names.txt')
words = open(input_file_path, 'r').read().splitlines()

# lookup tables for s --> i || i --> s
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s,i in stoi.items()}
vocab_size = len(itos)

# build the dataset
block_size = 3

random.seed(1789)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1], stoi=stoi, itos=itos, block_size=block_size)      # 80%
Xdev, Ydev = build_dataset(words[n1:n2], stoi=stoi, itos=itos, block_size=block_size)    # 10%
Xte,  Yte  = build_dataset(words[n2:], stoi=stoi, itos=itos, block_size=block_size)      # 10%

# MLP revisited
n_emb = 10        # dimensionality of the character embedding vectors
n_hidden = 200    # number of neurons in the hidden layer of the MLP

g  = t.Generator().manual_seed(1789)
C  = t.randn((vocab_size, n_emb),            generator=g)
w1 = t.randn((n_emb * block_size, n_hidden), generator=g)
b1 = t.randn(n_hidden,                       generator=g)
w2 = t.randn((n_hidden, vocab_size),         generator=g)
b2 = t.randn(vocab_size,                     generator=g)

parameters = [C, w1, b1, w2, b2]
print(f"number of parameters in total: {sum(p.nelement() for p in parameters)}")

for p in parameters:
    p.requires_grad = True

# optimization
max_steps = 10000
batch_size = 32
lossi = []
for i in range(max_steps):

    # minibatch construct
    ix = t.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb]                            # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1)    # concatenate the vectors
    hpreact = embcat @ w1 + b1             # hidden layer pre-activation
    h = t.tanh(hpreact)                    # hidden layer
    logits = h @ w2 + b2                  # output layer
    loss = F.cross_entropy(logits, Yb)     # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 5000 else 0.01    # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 500 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
    lossi.append(loss.log10().item())

plt.plot(lossi)
# plt.show()

split_loss("train",
           Xtr=Xtr, Ytr=Ytr,
           Xdev=Xdev, Ydev=Ydev,
           Xte=Xte, Yte=Yte,
           C=C,
           w1=w1, b1=b1, w2=w2, b2=b2)
split_loss("val",
           Xtr=Xtr, Ytr=Ytr,
           Xdev=Xdev, Ydev=Ydev,
           Xte=Xte, Yte=Yte,
           C=C,
           w1=w1, b1=b1, w2=w2, b2=b2)


































