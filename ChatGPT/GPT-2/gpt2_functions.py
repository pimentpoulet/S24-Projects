import torch as t
import requests
import numpy as np
import torch.nn.functional as F
import os

from graphviz import Digraph

from datasets import *


device = "cuda" if t.cuda.is_available() else "cpu"

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()


def encode(s):
    """
    takes a string as input, outputs a list of integers
    :param s: string
    :return: list of integers
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return [stoi[c] for c in s]


def decode(l):
    """
    takes a list as input, outputs a string
    :param l: list of integers
    :return: string
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return ''.join([itos[i] for i in l])


def get_batch(split,
              train_data,
              val_data,
              block_size,
              batch_size,
              device):
    """
    generates a small batch of data of input x and targets y
    :param split: train or validation dataset
    :param train_data: training dataset
    :param val_data: validation dataset
    :param block_size: int
    :param batch_size: int
    :param device: device to train with
    :return: tuple (input data, target data)
    """
    data = train_data if split == 'train' else val_data
    ix = t.randint(len(data) - block_size, (batch_size,))
    x = t.stack([data[i:i + block_size] for i in ix])
    y = t.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@t.no_grad()
def estimate_loss(train_data,
                  val_data,
                  block_size,
                  batch_size,
                  model,
                  eval_iters):
    """
    estimates loss of a language model
    :param eval_iters: number of evaluating iterations
    :param model: model's name
    :param train_data: training dataset
    :param val_data: validation dataset
    :param block_size: int
    :param batch_size: int
    :return: mean loss for a split
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = t.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split=split,
                             train_data=train_data,
                             val_data=val_data,
                             block_size=block_size,
                             batch_size=batch_size,
                             device=device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out
