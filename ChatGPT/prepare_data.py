import torch as t

from get_data import *
from functions import *


# encode the entire text dataset and store it in a torch.Tensor
data = t.tensor(encode(text), dtype=t.long)

# split data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

t.manual_seed(344)
BATCH_SIZE = 4
BLOCK_SIZE = 8

xb, yb = get_batch('train',
                   train_data=train_data,
                   val_data=val_data,
                   block_size=BLOCK_SIZE,
                   batch_size=BATCH_SIZE)

print(xb)
print(yb)









