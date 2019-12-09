import sys
import io
import numpy as np
import logging
from os import path as osp
import pickle

import torch

from configuration import GPU, VocabConfig as vconfig, Paths
from utils.helpers import word_lists_to_lines, lines_to_word_lists, \
    create_vocabulary, create_subwords, prepare_sentences
from utils import subwords
from models.sentence_encoders import SentenceEncoder, BOREP, RandomLSTM
from models.bert_based_models import TransformerWordEmbedder

import senteval


# SentEval prepare and batcher

def train_prepare(params, samples):
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


def eval_prepare(params, samples):
    # TODO: add new vocab with Vocab.add_corpus in fine-tune mode
    if vconfig.subwords:
        subwords_folder = osp.join(params.task_path, "vocab")
        sub_model_path = osp.join(subwords_folder, "{}.{}.model".format(params.current_task, vconfig.subwords_model_type))
        reader = subwords.SubwordReader(sub_model_path)
        params.reader = reader
    vocab_folder = osp.join(params.task_path, "vocab")
    vocab_file_path = osp.join(vocab_folder, "{}.vocab".format(params.current_task))
    task_vocab = pickle.load(open(vocab_file_path, 'rb'))
    params.vocab = task_vocab
    return


def tokenize(params, dataset):
    sub_reader = params.get("reader")
    if sub_reader is not None:
        return list(sub_reader.lines_to_subwords(word_lists_to_lines(dataset)))
    else:
        return dataset


def batcher(params, batch):
    # batch is an array of (string) sentences or list of word lists
    if not isinstance(batch[0], str):
        # normalize the batch
        batch = word_lists_to_lines(batch)

    if vconfig.subwords:
        # TODO: cleanup the reader loading. This code might be risky, as we use small batches
        if not params.reader:
            reader = create_subwords(batch, params.task_path, params.current_task)
            params.reader = reader
        lines = list(params.reader.lines_to_subwords(batch))
    else:
        # TODO: apply a vocabulary
        lines = lines_to_word_lists(batch)
    if not params.vocab:
        raise KeyError("no vocab in params !")
    feature_vectors, masks = prepare_sentences(lines, params.vocab)

    with torch.no_grad():
        sentence_vectors = params.sentence_encoder(params.word_embedder(feature_vectors, masks), masks)

    return sentence_vectors


# Set params for SentEval
base_params = {'base_path': Paths.semeval_data_path, 'usepytorch': True, 'kfold': 5}
base_params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                             'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    word_embedder = TransformerWordEmbedder()
    sentence_encoder = RandomLSTM()
    try:
        word_embedder.load_params(osp.join(Paths.direct_reload_path, "embedder"))
        sentence_encoder.load_params(osp.join(Paths.direct_reload_path, "encoder"))
    except (AttributeError, FileNotFoundError):
        pass
    if GPU:
        word_embedder = word_embedder.cuda()
        sentence_encoder = sentence_encoder.cuda()

    base_params["sentence_encoder"] = sentence_encoder
    base_params["word_embedder"] = word_embedder
    training_tasks = ['SNLI']
    testing_tasks = ['EmoContext', 'SNLI', 'HatEval']

    if training_tasks:
        te = senteval.train_engine.TrainEngine(base_params, train_prepare)
        train_results = te.train(training_tasks)
        print("train results:")
        print(train_results)

    if testing_tasks:
        ee = senteval.eval_engine.SE(base_params, batcher, eval_prepare)
        test_results = ee.eval(testing_tasks)
        print("test_results:")
        print(test_results)
