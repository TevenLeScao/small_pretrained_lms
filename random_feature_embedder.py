import sys
import io
import numpy as np
import logging
from os import path as osp
import pickle

import torch

from configuration import GPU, VocabConfig as vconfig
import paths
from utils.helpers import word_lists_to_lines, lines_to_word_lists, \
    create_vocabulary, create_subwords, make_masks
from utils import subwords
from models.sentence_encoders import SentenceEncoder, BOREP, RandomLSTM
from models.word_embedders import BertWordEmbedder
from models.structure import SimpleDataParallel

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
        sub_model_path = osp.join(subwords_folder,
                                  "{}.{}.model".format(params.current_task, vconfig.subwords_model_type))
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
        return [[params.vocab[subword] for subword in subworded_line] for subworded_line in
                sub_reader.lines_to_subwords(word_lists_to_lines(dataset))]
    else:
        return [[params.vocab[word] for word in line] for line in dataset]


def batcher(params, batch):
    params.word_embedder.eval()
    params.sentence_encoder.eval()
    # batch is an array of (string) sentences or list of word lists
    tokenized_sentences = tokenize(params, batch)

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

    word_embedder = BertWordEmbedder()
    sentence_encoder = RandomLSTM()
    try:
        word_embedder.load_params(osp.join(paths.direct_reload_path, "embedder"))
        if not base_params.get("train_encoder"):
            sentence_encoder.load_params(osp.join(paths.direct_reload_path, "encoder"))
    except (AttributeError, FileNotFoundError):
        pass
    if GPU:
        word_embedder = word_embedder.cuda()
        sentence_encoder = sentence_encoder.cuda()
        if torch.cuda.device_count() > 1:
            print("%s GPUs found, using parallel data"%torch.cuda.device_count())
            word_embedder = SimpleDataParallel(word_embedder, dim=0)
            sentence_encoder = SimpleDataParallel(sentence_encoder, dim=0)
    base_params["sentence_encoder"] = sentence_encoder
    base_params["word_embedder"] = word_embedder
    base_params["tokenize"] = tokenize
    training_tasks = ['Sentiment']
    testing_tasks = ['Sentiment', 'EmoContext']

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
