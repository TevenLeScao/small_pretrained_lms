import sys
import io
import numpy as np
import logging
from os import path as osp
import pickle

from configuration import GPU, VocabConfig as vconfig, Paths
from utils.helpers import word_lists_to_lines, lines_to_word_lists,\
    create_vocabulary, create_subwords, prepare_sentences
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
    # batch is an array of (string) sentences or list of word lists
    if not isinstance(batch[0], str):
        # normalize the batch
        batch = word_lists_to_lines(batch)

    if vconfig.subwords:
        #TODO: cleanup the reader loading. This code might be risky, as we use small batches
        if not params.reader:
            reader = create_subwords(batch, params.task_path, params.current_task)
            params.reader = reader
        lines = list(params.reader.lines_to_subwords(batch))
    else:
        #TODO: apply a vocabulary
        lines = lines_to_word_lists(batch)
    if not params.vocab:
        params.vocab = create_vocabulary(lines, params.task_path, params.current_task)
    feature_vectors, masks = prepare_sentences(lines, params.vocab)

    with torch.no_grad():
        sentence_vectors = params.sentence_encoder(params.word_embedder(feature_vectors, masks), masks)

    return sentence_vectors


# Set params for SentEval
params_senteval = {'task_path': Paths.semeval_data_path, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    word_embedder = TransformerWordEmbedder()
    print(word_embedder)
    sentence_encoder = RandomLSTM()
    if GPU:
        word_embedder = word_embedder.cuda()
        sentence_encoder = sentence_encoder.cuda()
    params_senteval["sentence_encoder"] = sentence_encoder
    params_senteval["word_embedder"] = word_embedder
    te = senteval.train_engine.TrainEngine(params_senteval, prepare)
    training_tasks = ['EmoContext']
    testing_tasks = ['EmoContext']
    ee = senteval.eval_engine.SE(params_senteval, batcher)
    train_results = te.train(training_tasks)
    test_results = ee.eval(testing_tasks)
    print("train results:")
    print(train_results)
    print("test_results:")
    print(test_results)
