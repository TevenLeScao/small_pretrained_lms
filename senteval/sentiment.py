import os
import torch
import numpy as np

from senteval.tools.classifier_task import Classifier_task
from utils.helpers import prepare_sentences, lines_to_word_lists

class SentimentAnalysis(Classifier_task):

    def __init__(self, taskpath, seed=111):
        self.sorting_key = lambda z: len(z[0])
        self.dict_label = {'neutral': 0, 'positive': 1, 'negative': 2}
        self.classifier_input_multiplier = 1
        self.task_name = "Sentiment Analysis"
        super(SentimentAnalysis, self).__init__(taskpath, seed)

    def loadFiles(self, path: str, file_type: str):
        assert os.path.isdir(path), "Directory %s not found" % path
        assert file_type in ["train", "dev", "test"], "File type must be 'train', 'dev' or 'test'"

        sent = open(os.path.join(path, file_type+"_sent.txt")).readlines()
        sent = lines_to_word_lists(sent)

        labels = open(os.path.join(path, file_type+"_labels.txt")).readlines()
        return sent, labels

    def batch_data_to_sent_embeddings(self, models, batch_features, params):
        word_embedder, sentence_encoder, _ = models
        s = list(zip(*batch_features))[0]
        sents, mask = prepare_sentences(s, params.vocab)
        return sentence_encoder(word_embedder(sents, mask), mask)

    def prepare_sent_embeddings(self, current_data, params, batcher):
        enc_input = []
        sents = current_data[0]
        n_labels = len(current_data)
        for ii in range(0, n_labels, params.batch_size):
            batch = sents[ii:ii + params.batch_size]
            enc_input.append(batcher(params, batch))
            if ii % 200 == 0:
                print("PROGRESS: %.2f%%" % (100 * ii / n_labels))
        return enc_input