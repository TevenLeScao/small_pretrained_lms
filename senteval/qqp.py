import os
import torch
import csv

from senteval.tools.classifier_task import Classifier_task
from utils.helpers import prepare_sentences, lines_to_word_lists

class QQP(Classifier_task):

    def __init__(self, taskpath, seed=111):
        self.sorting_key = lambda z: len(z[0])+len(z[1])
        self.dict_label = {'0': 0, '1': 1}
        self.classifier_input_multiplier = 1
        self.task_name = "Sentiment Analysis"
        super(QQP, self).__init__(taskpath, seed)

    def loadFiles(self, path: str, file_type: str):
        assert os.path.isdir(path), "Directory %s not found" % path
        assert file_type in ["train", "dev", "test"], "File type must be 'train', 'dev' or 'test'"
        if file_type == "test":
            print("***************************************************************************************************")
            print("WARNING: This is not the official testing set, since the official labels are not publicly available."
                  "This does not correspond to the official benchmarks."
                  "For simplicity, we will only use the dev set again")
            print("***************************************************************************************************")
            file_type = "dev"
        s1, s2, labels = [], [], []
        with open(os.path.join(path, file_type+".tsv")) as infile:
            next(infile)
            bad_rows = 0
            for row in infile:
                row = row.strip()
                try:
                    s1_now, s2_now, label_now = row.split('\t')[3:]
                    s1.append(s1_now.split())
                    s2.append(s2_now.split())
                    labels.append(label_now)
                except:
                    bad_rows += 1
        print("%s bad rows"%bad_rows)
        return s1, s2, labels

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
            if (ii // params.batch_size)%10 == 0 :
                print("PROGRESS: %.2f%%" % (100 * ii / n_labels))
        return enc_input