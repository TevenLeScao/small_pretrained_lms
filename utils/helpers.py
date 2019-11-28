import json
from typing import List
import numpy as np
import math
import sys
import time
import os
import os.path as osp
import errno
import pickle
import re
import inspect

import torch
from torch import optim

from configuration import GPU, VocabConfig as vconfig
from utils import subwords, vocab


def makedirs(name):
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and osp.isdir(name):
            pass
        else:
            raise


def prepare_sentences(sentences: List[List[str]], vocab):
    lengths = [len(sentence) for sentence in sentences]
    mask = [[0 for _ in range(length)] + [1 for _ in range(max(lengths) - length)] for length in lengths]
    mask = torch.FloatTensor(mask)

    np_sents = [np.array([vocab[word] for word in sent]) for sent in sentences]
    if GPU:
        mask = mask.cuda()
        tensor_sents = [torch.LongTensor(s).cuda() for s in np_sents]
    else:
        tensor_sents = [torch.LongTensor(s) for s in np_sents]

    packed_sents = torch.nn.utils.rnn.pad_sequence(tensor_sents, padding_value=0, batch_first=True)

    return packed_sents, mask


def batch_iter(data, batch_size):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    Shuffle not supported for generator objects
    """
    # gotta check the number of items
    assert hasattr(data, '__len__')
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))
    np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)

        yield ([e[i] for e in examples] for i in range(len(examples[0])))


def word_lists_to_lines(sentences):
    # TODO: take in lines instead of list of lists
    for sentence in sentences:
        yield " ".join(sentence) + "\n"


# If subwords already exist, load them, create them otherwise
def create_subwords(lines, task_folder, task_name):
    subwords_folder = osp.join(task_folder, "vocab")
    print(subwords_folder)
    sub_model_path = osp.join(subwords_folder, "{}.{}.model".format(task_name, vconfig.subwords_model_type))
    try:
        reader = subwords.SubwordReader(sub_model_path)
    except (FileNotFoundError, OSError):
        makedirs(subwords_folder)
        subwords_data_path = osp.join(subwords_folder, "{}_subwords_source.txt".format(task_name))
        if not osp.exists(subwords_data_path):
            print("writing out subwords source file")
            with open(subwords_data_path, "w") as f:
                for line in lines:
                    f.write(line)
        subwords.train(subwords_folder, subwords_data_path, prefix=task_name + ".")
        reader = subwords.SubwordReader(sub_model_path)
    return reader


# If a vocabulary already exists, load it, create it otherwise
def create_vocabulary(lines, task_folder, task_name):
    vocab_folder = osp.join(task_folder, "vocab")
    vocab_file_path = osp.join(vocab_folder, "{}.vocab".format(task_name))
    # use vocab class
    try:
        task_vocab = pickle.load(open(vocab_file_path, 'rb'))
    except FileNotFoundError:
        makedirs(vocab_folder)
        task_vocab = vocab.Vocab.from_corpus(lines, vconfig.vocab_size, vconfig.freq_cutoff)
        pickle.dump(task_vocab, open(vocab_file_path, 'wb'))
    return task_vocab


def create_dictionary(sentences):
    words = {}
    for s in sentences:
        for word in s:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2
    # words['<UNK>'] = 1e9 + 1
    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params