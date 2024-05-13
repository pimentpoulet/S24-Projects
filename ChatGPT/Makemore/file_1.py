import os
import requests
import tiktoken
import numpy as np
import torch as t


# download the names dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'names.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

words = open(input_file_path, 'r').read().splitlines()
# print(words[:10])
# print(min(len(w) for w in words))
# print(max(len(w) for w in words))

b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    # print(chs)
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

most_occuring = sorted(b.items(), key=lambda kv: -kv[1])
# print(most_occuring)
