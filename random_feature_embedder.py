import sys
import io
import numpy as np
import logging
from os import path as osp
import pickle

from configuration import VocabConfig as vconfig
from utils import subwords, helpers, vocab

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def lines_from_list(sentences):
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
        helpers.makedirs(subwords_folder)
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
def create_dictionary(lines, task_folder, task_name):
    vocab_folder = osp.join(task_folder, "vocab")
    vocab_file_path = osp.join(vocab_folder, "{}.vocab".format(task_name))
    # use vocab class
    try:
        task_vocab = pickle.load(open(vocab_file_path, 'rb'))
    except FileNotFoundError:
        helpers.makedirs(vocab_folder)
        task_vocab = vocab.Vocab.from_corpus(lines, vconfig.vocab_size, vconfig.freq_cutoff)
        pickle.dump(task_vocab, open(vocab_file_path, 'wb'))
    return task_vocab


# SentEval prepare and batcher

def prepare(params, samples):
    # run vocab + subwords
    # TODO: take in lines instead of list of lists
    if vconfig.subwords:
        lines = lines_from_list(samples)
        reader = create_subwords(lines, params.task_path, params.current_task)
        params.reader = reader
    # reset generator in case subwords exhausted it
    lines = lines_from_list(samples)
    if vconfig.subwords:
        lines = reader.lines_to_subwords(lines)
    params.vocab = create_dictionary(lines, params.task_path, params.current_task)
    return


def batcher(params, batch):
    # 1-hot embedding
    return


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, prepare)
    training_tasks = ['SNLI']
    results = se.eval(training_tasks)
    print(results)
