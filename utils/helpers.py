import json
from typing import List
import numpy as np
import math
import sys
import time
import os
import errno

import torch

from configuration import GPU


def prepare_sentences(sentences: List[List[str]], vocab):
    lengths = [len(sentence) for sentence in sentences]
    mask = [[0 for _ in range(length)] + [1 for _ in range(max(lengths) - length)] for length in lengths]
    mask = torch.BoolTensor(mask)

    np_sents = [np.array([vocab[word] for word in sent]) for sent in sentences]
    if GPU:
        mask = mask.cuda()
        tensor_sents = [torch.LongTensor(s).cuda() for s in np_sents]
    else:
        tensor_sents = [torch.LongTensor(s) for s in np_sents]

    packed_sents = torch.nn.utils.rnn.pad_sequence(tensor_sents, padding_value=0)

    return packed_sents, mask


def batch_iter(data, batch_size):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    Shuffle not supported for generator objects
    """

    assert hasattr(data, '__len__')
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


def makedirs(name):
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            pass
        else:
            raise
