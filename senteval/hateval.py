import os
import torch
import numpy as np
from time import time
from nltk.tokenize import TweetTokenizer
import csv
from utils.progress_bar import progress_bar

from senteval.tools.classifier_task import Classifier_task
from utils.helpers import prepare_sentences

class HatEval(Classifier_task):

    def __init__(self, taskpath, seed=111):
        self.sorting_key = lambda z: len(z[0])
        self.dict_label = {0: 0, 1: 1}
        self.classifier_input_multiplier = 1
        self.task_name = "HatEval"
        self.eval_metrics = ['acc', 'f1', 'conf']
        self.f1_excluded_classes = ()
        super(HatEval, self).__init__(taskpath, seed)


    def loadFiles(self, path: str, file_type: str):
        assert os.path.isdir(path), "Directory %s not found" % path
        assert file_type in ["train", "dev", "test"], "File type must be 'train', 'dev' or 'test'"

        tt = TweetTokenizer()
        sent, labels = [], []
        with open(os.path.join(path, "hateval2019_en_" + file_type + ".csv"), "r") as f:
            reader = csv.reader(f)
            for line in reader:
                if not line[0].isdigit():
                    continue
                tokens = tt.tokenize(line[1])
                for i in range(len(tokens)):
                    if tokens[i][0] == "@":
                        tokens[i] = "<USER>"
                    if "https://" in tokens[i]:
                        tokens[i] = "<LINK>"
                sent.append(tokens)

                labels.append(int(line[2]))
        return sent, labels

    def batch_data_to_sent_embeddings(self, models, batch_features, params):
        word_embedder, sentence_encoder, _ = models
        s = list(zip(*batch_features))[0]
        sents, mask = prepare_sentences(s, params.vocab)
        return sentence_encoder(word_embedder(sents, mask), mask)

    def prepare_sent_embeddings(self, current_data, params, batcher):
        enc_input = []
        sents = current_data[0]
        n_labels = len(sents)
        for ii in range(0, n_labels, params.batch_size):
            batch = sents[ii:ii + params.batch_size]
            s = batcher(params, batch)
            enc_input.append(torch.tensor(s))
            if ii % 200 == 0:
                progress_bar(ii, n_labels)
        return enc_input