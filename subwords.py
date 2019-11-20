from os import path as osp, makedirs

import sentencepiece as spm

import configuration
import paths

vconfig = configuration.VocabConfig()


def train_model(train_source_path, target):
    vocab_size = str(vconfig.subwords_vocab_size)
    model_type = vconfig.subwords_model_type
    print("Train subwords model")

    spm.SentencePieceTrainer.Train('--input=' + train_source_path +
                                   ' --model_prefix=' + target +
                                   ' --model_type=' + model_type +
                                   ' --character_coverage=1.0 '
                                   '--vocab_size=' + vocab_size +
                                   "--max_sentence_length=4096")


def train():
    target_folder = paths.subwords_folder
    model_type = vconfig.subwords_model_type
    model_prefix = osp.join(target_folder, model_type + ".en")
    train_source_path = paths.subwords_data_path

    train_model(train_source_path, model_prefix)


class SubwordReader:
    def __init__(self):
        folder = paths.subwords_folder
        model_type = vconfig.subwords_model_type
        model_prefix = osp.join(folder, model_type)
        model_path = model_prefix + ".en.model"

        self.sp = spm.SentencePieceProcessor()
        print("Loading subword model :", model_path)
        self.sp.Load(model_path)

    def line_to_subwords(self, line):
        return self.sp.EncodeAsPieces(line)

    def subwords_to_line(self, l):
        return self.sp.DecodePieces(l)


def main():
    try:
        makedirs(paths.subwords_folder)
    except FileExistsError:
        pass
    if not vconfig.load_subwords:
        train()
    else:
        print("Loading subwords")


if __name__ == '__main__':
    main()
