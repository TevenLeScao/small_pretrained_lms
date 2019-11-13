import numpy as np
import pickle

import torch
from torch import nn

import paths
import utils
from configuration import GPU, ModelConfig as mconfig, TrainConfig as tconfig
from models.general_model import GeneralModel
from models.regressor import SentimentTransformer
from vocab import Vocab, VocabEntry

data_vocab = pickle.load(open(paths.vocab_path, 'rb')).src


def get_data():
    train_data_src = utils.read_corpus(paths.get_data_path(chunk="train", origin="src"))
    train_data_tgt = np.load(open(paths.get_data_path(chunk="train", origin="tgt"), "rb"))
    train_data = zip(train_data_src, train_data_tgt)
    valid_data_src = utils.read_corpus(paths.get_data_path(chunk="valid", origin="src"))
    valid_data_tgt = np.load(open(paths.get_data_path(chunk="valid", origin="tgt"), "rb"))
    valid_data = zip(valid_data_src, valid_data_tgt)
    return train_data, valid_data


def epoch_loop(model: GeneralModel, data, validation=False):
    for source, target in utils.batch_iter(data, tconfig.batch_size):
        sentences, mask = utils.prepare_sentences(source, data_vocab)
        if GPU:
            target = torch.Tensor(target).cuda()
        else:
            target = torch.Tensor(target)
        loss = model.decode_to_loss(sentences, mask, target)
        if not validation:
            model.step(loss)


if __name__ == "__main__":
    train_data, valid_data = get_data()
    sentiment_transformer = SentimentTransformer(depth=mconfig.depth, width=mconfig.width, d_ff=mconfig.d_ff, n_head=mconfig.n_head)
    if GPU:
        sentiment_transformer = sentiment_transformer.cuda()
    for epoch in range(tconfig.max_epoch):
        epoch_loop(sentiment_transformer, train_data, validation=False)
        epoch_loop(sentiment_transformer, valid_data, validation=True)