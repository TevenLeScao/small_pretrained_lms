import sys
import io
import numpy as np
import logging
from os import path as osp
import pickle

from configuration import GPU, VocabConfig as vconfig, Paths
from utils.helpers import word_lists_to_lines, create_vocabulary, create_subwords
from models.sentence_encoders import *
from models.bert_based_models import *

import senteval


# SentEval prepare and batcher

def prepare(params, samples):
    # run vocab + subwords
    # TODO: take in lines instead of list of lists
    lines = word_lists_to_lines(samples)
    if vconfig.subwords:
        reader = create_subwords(lines, params.task_path, params.current_task)
        params.reader = reader
        # need to re-create the generator
        lines = word_lists_to_lines(samples)
        lines = reader.lines_to_subwords(lines)
    params.vocab = create_vocabulary(lines, params.task_path, params.current_task)
    return


def batcher(params, batch):
    # TODO: write batcher for transfer eval
    return


# Set params for SentEval
params_senteval = {'task_path': Paths.semeval_data_path, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    word_embedder = TransformerWordEmbedder()
    print(word_embedder)
    sentence_embedder = RandomLSTM()
    if GPU:
        word_embedder = word_embedder.cuda()
        sentence_embedder = sentence_embedder.cuda()
    te = senteval.train_engine.TrainEngine(params_senteval, word_embedder, sentence_embedder, prepare)
    training_tasks = ['EmoContext']
    results = te.train(training_tasks)
    print(results)
