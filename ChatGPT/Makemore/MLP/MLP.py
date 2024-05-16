import os
import requests
import tiktoken
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import torch.nn.functional as F


input_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'names.txt')
words = open(input_file_path, 'r').read().splitlines()

# lookup tables for s --> i || i --> s
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s,i in stoi.items()}

# build the dataset
BLOCK_SIZE = 3    # how many characters do we take to predict the next one ?
X, Y = [], []
for w in words:
    # print(w)
    context = [0] * BLOCK_SIZE
    # print(context)
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix]    # crop and append

# datasets
X = t.tensor(X)
Y = t.tensor(Y)

g = t.Generator().manual_seed(1789)
C = t.randn((27, 2))
w1 = t.randn((6, 100))
b1 = t.randn(100)
w2 = t.rand((100, 27))
b2 = t.randn(27)

parameters = [C, w1, b1, w2, b2]
num = sum(p.nelement() for p in parameters)

# set gradient requirements
for p in parameters:
    p.requires_grad = True

for _ in range(10):
    # forward pass
    emb = C[X]    # (32, 3, 2)
    h = t.tanh(emb.view(-1, 6) @ w1 + b1)    # (32, 100)
    logits = h @ w2 + b2    # (32, 27)
    loss = F.cross_entropy(logits, Y)
    print(loss.item())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    for p in parameters:
        p.data += -1 * p.grad
