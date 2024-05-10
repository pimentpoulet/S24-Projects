import torch as t

from get_data import *


def encode(s):
    """
    takes a string as input, outputs a list of integers
    :param s: string
    :return: list of integers
    """
    # create a mapping | characters <--> integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return [stoi[c] for c in s]


def decode(l):
    """
    takes a list as input, outputs a string
    :param l: list of integers
    :return: string
    """
    # create a mapping | characters <--> integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return ''.join([itos[i] for i in l])


def get_batch(split,
              train_data,
              val_data,
              block_size,
              batch_size):
    """
    generates a small batch of data of input x and targets y
    :param split: train or validation dataset
    :param train_data: training dataset
    :param val_data: validation dataset
    :param block_size: int
    :param batch_size: int
    :return: tuple (x, y)
    """
    data = train_data if split == 'train' else val_data
    ix = t.randint(len(data) - block_size, (batch_size,))
    x = t.stack([data[i:i+block_size] for i in ix])
    y = t.stack([data[i+1:i+block_size] for i in ix])

    return x, y
