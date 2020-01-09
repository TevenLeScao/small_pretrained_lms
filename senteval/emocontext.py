'''emotional context analysis'''
import os
import torch
from senteval.tools.classifier_task import Classifier_task
from utils.helpers import lines_to_word_lists, prepare_sentences
from utils.progress_bar import progress_bar

class EmoContext(Classifier_task):

    def __init__(self, taskpath, seed=111):
        self.sorting_key = lambda z: len(z[0]) + len(z[1]) + len(z[2])
        self.dict_label = {'others': 0, 'happy': 1, 'sad': 2, 'angry': 3}
        self.classifier_input_multiplier = 6
        self.task_name = "EmoContext"
        self.eval_metrics = ['loss', 'acc', 'f1']
        self.f1_excluded_classes = (0,)
        super(EmoContext, self).__init__(taskpath, seed)

    def loadFiles(self, path: str, file_type: str):
        assert os.path.isdir(path), "Directory %s not found" % path
        assert file_type in ["train", "dev", "test"], "File type must be 'train', 'dev' or 'test'"

        s1 = open(os.path.join(path, "s1." + file_type)).read()
        s1 = list(lines_to_word_lists(s1))
        s1.remove([])
        s2 = open(os.path.join(path, "s2." + file_type)).read()
        s2 = list(lines_to_word_lists(s2))
        s2.remove([])
        s3 = open(os.path.join(path, "s3." + file_type)).read()
        s3 = list(lines_to_word_lists(s3))
        s3.remove([])
        labels = open(os.path.join(path, "label." + file_type)).read().split('\n')
        labels.remove("")

        return s1, s2, s3, labels

    def batch_data_to_sent_embeddings(self, models, batch_features, params):
        word_embedder, sentence_encoder, _ = models
        sents1, sents2, sents3 = list(zip(*batch_features))
        sents1, mask1 = prepare_sentences(sents1, params.vocab)
        sents2, mask2 = prepare_sentences(sents2, params.vocab)
        sents3, mask3 = prepare_sentences(sents3, params.vocab)
        enc1 = sentence_encoder(word_embedder(sents1, mask1), mask1)
        enc2 = sentence_encoder(word_embedder(sents2, mask2), mask2)
        enc3 = sentence_encoder(word_embedder(sents3, mask3), mask3)
        return torch.cat((enc1, enc2, enc3, enc1 * enc2, enc2 * enc3, enc3 * enc1), dim=1)

    def prepare_sent_embeddings(self, current_data, params, batcher):
        s1, s2, s3 = current_data
        enc_input = []
        n_labels = len(s1)
        for ii in range(0, n_labels, params.batch_size):
            batch1 = s1[ii:ii + params.batch_size]
            batch2 = s2[ii:ii + params.batch_size]
            batch3 = s3[ii:ii + params.batch_size]

            if len(batch1) == len(batch2) and len(batch1) > 0:
                enc1 = torch.tensor(batcher(params, batch1))
                enc2 = torch.tensor(batcher(params, batch2))
                enc3 = torch.tensor(batcher(params, batch3))
                enc_input.append(torch.cat((enc1, enc2, enc3, enc1 * enc2, enc2 * enc3, enc3 * enc1), dim=1))
            if ii % 200 == 0:
                progress_bar(ii, n_labels)
        print("")
        return enc_input
