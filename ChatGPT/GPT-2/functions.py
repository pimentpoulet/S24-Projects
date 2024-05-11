import torch as t

from graphviz import Digraph

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
    :return: tuple (input data, target data)
    """
    data = train_data if split == 'train' else val_data
    ix = t.randint(len(data) - block_size, (batch_size,))
    x = t.stack([data[i:i + block_size] for i in ix])
    y = t.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x, y


def trace(root):
    """
    builds a set of all nodes and edges in a graph
    :param root:
    :return:
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    """

    :param root:
    :return:
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR: Left --> Right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{%s | data %.4f }" % (n.label, n.data), shape='record')
        if n.op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n.op, label=n.op)
            # and connect this node to it
            dot.edge(uid + n.op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot
