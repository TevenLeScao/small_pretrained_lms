import sys
import io
import numpy as np
import logging
from os import path as osp
import pickle
from importlib import reload

import torch
from transformers import BertTokenizer

import configuration
from configuration import GPU, VocabConfig as vconfig, ModelConfig as mconfig
import paths
from utils.helpers import word_lists_to_lines, lines_to_word_lists, \
    create_vocabulary, create_subwords, prepare_sentences, make_masks
from models.sentence_encoders import SentenceEncoder, BOREP, RandomLSTM
from models.word_embedders import TransformerWordEmbedder, MODEL_CLASSES

import senteval

preloaded_model = "bert-base-uncased"
configuration.EXPERIMENT_NAME = preloaded_model
reload(paths)

# SentEval prepare and batcher

def prepare(params, samples):
    pass


def pretrained_tokenize(params, dataset):
    return [params.pretrained_tokenizer.encode(line) for line in word_lists_to_lines(dataset)]
    # return [["[CLS]"] + params.pretrained_tokenizer.tokenize(line) + ["[SEP]"] for line in word_lists_to_lines(dataset)]


def batcher(params, batch):
    # batch is an array of (string) sentences or list of word lists

    tokenized_sentences = pretrained_tokenize(params, batch)

    if not params.vocab:
        raise KeyError("no vocab in params !")
    feature_vectors, masks = make_masks(tokenized_sentences)

    with torch.no_grad():
        word_vectors = params.word_embedder(feature_vectors, masks)
        sentence_vectors = params.sentence_encoder(word_vectors, masks).cpu().numpy()

    return sentence_vectors


# Set params for SentEval
base_params = {'base_path': paths.senteval_data_path, 'usepytorch': True, 'kfold': 5, "train_encoder":True}
base_params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 64,
                             'tenacity': 3, 'epoch_size': 2}
base_params.update({"semeval_path": paths.semeval_data_path, "others_path": paths.others_data_path})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    word_embedder = TransformerWordEmbedder(pretrained=preloaded_model)
    sentence_encoder = RandomLSTM(word_dim=word_embedder.embedding_size)
    mconfig.width = word_embedder.embedding_size
    # sentence_encoder = BOREP(word_dim=word_embedder.embedding_size)
    tokenizer = MODEL_CLASSES[mconfig.model]["tokenizer"].from_pretrained(preloaded_model)
    if GPU:
        word_embedder = word_embedder.cuda()
        sentence_encoder = sentence_encoder.cuda()

    base_params["sentence_encoder"] = sentence_encoder
    base_params["word_embedder"] = word_embedder
    base_params["pretrained_tokenizer"] = tokenizer
    base_params["tokenize"] = pretrained_tokenize
    try:
        base_params["vocab"] = tokenizer.vocab
    except AttributeError:
        # the XLM attribute is called encoder instead
        base_params["vocab"] = tokenizer.encoder

    ee = senteval.eval_engine.SE(base_params, batcher)
    testing_tasks = ['EmoContext', 'HatEval']

    if testing_tasks:
        test_results = ee.eval(testing_tasks)
        print("test_results:")
        print(test_results)
