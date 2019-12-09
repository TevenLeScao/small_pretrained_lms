import sys
import io
import numpy as np
import logging
from os import path as osp
import pickle

import torch

from transformers import BertConfig, BertModel, BertForSequenceClassification, BertTokenizer

from configuration import GPU, VocabConfig as vconfig, Paths
from utils.helpers import word_lists_to_lines, lines_to_word_lists,\
    create_vocabulary, create_subwords, prepare_sentences
from models.sentence_encoders import SentenceEncoder, BOREP, RandomLSTM
from models.bert_based_models import TransformerWordEmbedder

import senteval


# SentEval prepare and batcher

def prepare(params, samples):
    pass


def tokenize(params, dataset):
    return [params.tokenizer.tokenize(line) for line in word_lists_to_lines(dataset)]


def batcher(params, batch):
    # batch is an array of (string) sentences or list of word lists
    if not isinstance(batch[0], str):
        # normalize the batch
        batch = word_lists_to_lines(batch)

    sents = [params.tokenizer.tokenize(line) for line in batch]

    if not params.vocab:
        raise KeyError("no vocab in params !")
    feature_vectors, masks = prepare_sentences(sents, params.vocab)

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

    word_embedder = TransformerWordEmbedder(load_bert=True)
    sentence_encoder = RandomLSTM(word_dim=word_embedder.embedding_size)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if GPU:
        word_embedder = word_embedder.cuda()
        sentence_encoder = sentence_encoder.cuda()

    base_params["sentence_encoder"] = sentence_encoder
    base_params["word_embedder"] = word_embedder
    base_params["tokenizer"] = tokenizer
    base_params["vocab"] = tokenizer.vocab

    ee = senteval.eval_engine.SE(base_params, batcher)
    testing_tasks = ['EmoContext', 'HatEval', 'SNLI']

    if testing_tasks:
        test_results = ee.eval(testing_tasks)
        print("test_results:")
        print(test_results)
