from os import path as osp

import sentencepiece as spm
import youtokentome as yttm

from utils.helpers import makedirs
from configuration import VocabConfig as vconfig


def train_model(train_source_path, target):
    vocab_size = str(vconfig.subwords_vocab_size)
    model_type = vconfig.subwords_model_type
    print("training subwords model")

    print(target)

    spm.SentencePieceTrainer.Train('--input=' + train_source_path +
                                   ' --model_prefix=' + target +
                                   ' --model_type=' + model_type +
                                   ' --character_coverage=1.0 '
                                   '--vocab_size=' + vocab_size +
                                   "--max_sentence_length=4096")


def train(subwords_folder, train_source_path, prefix=''):
    target_folder = subwords_folder
    model_name = prefix + vconfig.subwords_model_type
    model_prefix = osp.join(target_folder, model_name)

    train_model(train_source_path, model_prefix)


class SubwordReader:
    def __init__(self, model_path):

        self.sp = spm.SentencePieceProcessor()
        print("loading subword model :", model_path)
        self.sp.Load(model_path)

    def line_to_subwords(self, line):
        return self.sp.EncodeAsPieces(line)

    def subwords_to_line(self, l):
        return self.sp.DecodePieces(l)

    def lines_to_subwords(self, lines):
        for line in lines:
            yield self.line_to_subwords((line))
