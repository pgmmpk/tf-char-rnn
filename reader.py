from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import codecs

import numpy as np
import tensorflow as tf

from vocab import Vocab
import random


def read_datasets(input_data, train_fraction=0.95, valid_fraction=0.05, vocab=None, vocab_size=128, shuffle_lines=True):

    print('Reading data from', input_data, '...')

    with open(input_data, 'r', encoding='utf-8') as f:
        data = f.read()

    if shuffle_lines:
        data = data.split('\n')
        random.shuffle(data)
        data = '\n'.join(data)

    if vocab is None:
        vocab = Vocab.from_data(data, vocab_size=vocab_size)

    train_size = int(math.floor(len(data) * train_fraction))
    valid_size = int(math.floor(len(data) * valid_fraction))
    train_data = data[:train_size]

    valid_data = data[train_size:train_size+valid_size]
    test_data = data[train_size+valid_size:]

    return [vocab.encode(c) for c in train_data], [vocab.encode(c) for c in valid_data], [vocab.encode(c) for c in test_data], vocab


def dataset_iterator(dataset, batch_size, num_steps):

    raw_data = np.array(dataset, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
      data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield x, y


def next_batch(dataset, batch_size, num_steps):

    while True:
        for x, y in dataset_iterator(dataset, batch_size, num_steps):
            yield x, y
