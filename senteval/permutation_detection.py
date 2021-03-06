import os
import torch
import numpy as np

from senteval.tools.classifier_task import Classifier_task
from utils.helpers import prepare_sentences, lines_to_word_lists
from utils.progress_bar import progress_bar

class PermutationDetection(Classifier_task):

    def __init__(self, taskpath, seed=111):
        self.sorting_key = lambda z: len(z[0])
        self.dict_label = {'original': 0, 'altered': 1}
        self.classifier_input_multiplier = 1
        self.task_name = "Permutation detection"
        super(PermutationDetection, self).__init__(taskpath, seed)

    def loadFiles(self, path: str, file_type: str):
        assert os.path.isdir(path), "Directory %s not found" % path
        assert file_type in ["train", "dev", "test"], "File type must be 'train', 'dev' or 'test'"

        sent = open(os.path.join(path, file_type+"_sents.txt")).read().splitlines(keepends=False)
        sent = list(lines_to_word_lists(sent))
        labels = open(os.path.join(path, file_type+"_labels.txt")).read().splitlines(keepends=False)
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
            if ii % 20000 == 0:
                progress_bar(ii, n_labels)
        print("")
        return enc_input