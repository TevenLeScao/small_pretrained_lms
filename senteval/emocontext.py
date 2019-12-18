import os
import numpy as np
import logging
import copy

from senteval.tools.validation import SplitClassifier

from utils.helpers import lines_to_word_lists
from configuration import SANITY, GPU, TrainConfig as tconfig
from senteval.trainer import Trainer


class EmoContext(Trainer):

    def __init__(self, taskpath, seed=111):
        super(EmoContext, self).__init__()
        self.seed = seed

        self.dict_label = {'others': 0, 'happy': 1, 'sad': 2, 'angry': 3}
        self.nclasses = len(self.dict_label)
        self.ninputs = 3

        train1, train2, train3, train_labels = self.loadFiles(taskpath, "train")
        train_labels = np.array([self.dict_label[y] for y in train_labels])
        train = zip(train1, train2, train3, train_labels)

        valid1, valid2, valid3, valid_labels = self.loadFiles(taskpath, "dev")
        valid_labels = np.array([self.dict_label[y] for y in valid_labels])
        valid = zip(valid1, valid2, valid3, valid_labels)

        test1, test2, test3, test_labels = self.loadFiles(taskpath, "test")
        test_labels = np.array([self.dict_label[y] for y in test_labels])
        test = zip(test1, test2, test3, test_labels)

        # sort to reduce the batch width
        train_sorted = sorted(train, key=lambda z: len(z[0]) + len(z[1]) + len(z[2]))[1:]
        valid_sorted = sorted(valid, key=lambda z: len(z[0]) + len(z[1]) + len(z[2]))[1:]
        test_sorted = sorted(test, key=lambda z: len(z[0]) + len(z[1]) + len(z[2]))[1:]

        if SANITY:
            self.data = {"train": list(zip(*train_sorted[:100])),
                         "valid": list(zip(*valid_sorted[:100])),
                         "test": list(zip(*test_sorted[:100]))}
        else:
            self.data = {"train": list(zip(*train_sorted)),
                         "valid": list(zip(*valid_sorted)),
                         "test": list(zip(*test_sorted))}
        self.data_source = self.data
        self.samples = train1 + train2 + train3 + valid1 + valid2 + valid3 + test1 + test2 + test3
        self.training_samples = train1 + train2 + train3

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def do_train_prepare(self, params, prepare):
        return prepare(params, self.training_samples)

    def loadFiles(self, path: str, file_type: str):
        assert os.path.isdir(path), "Directory %s not found" % path
        assert file_type in ["train", "dev", "test"], "File type must be 'train', 'dev' or 'test'"

        s1 = open(os.path.join(path, "s1." + file_type)).read()
        s1 = list(lines_to_word_lists(s1))[:-1]
        s2 = open(os.path.join(path, "s2." + file_type)).read()
        s2 = list(lines_to_word_lists(s2))[:-1]
        s3 = open(os.path.join(path, "s3." + file_type)).read()
        s3 = list(lines_to_word_lists(s3))[:-1]
        labels = open(os.path.join(path, "label." + file_type)).read().split('\n')[:-1]

        return s1, s2, s3, labels

    def run(self, params, batcher):
        if params.train_encoder:
            tconfig.resume_training = False
            params.sentence_encoder.__init__()
            if GPU:
                params.sentence_encoder = params.sentence_encoder.cuda()
            self.train(params, frozen_models=("embedder",))
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1, input2, input3, labels = self.data[key]
            enc_input = []
            n_labels = len(labels)
            for ii in range(0, n_labels, params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]
                batch3 = input3[ii:ii + params.batch_size]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    enc3 = batcher(params, batch3)
                    enc_input.append(np.hstack((enc1, enc2, enc3)))
                if (ii * params.batch_size) % (200 * params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
                    try:
                        self.X[key] = np.vstack((self.X[key], *enc_input))
                    except ValueError:
                        self.X[key] = np.vstack(enc_input)
                    enc_input = []
            self.X[key] = np.vstack((self.X[key], *enc_input))
            self.y[key] = np.array(labels)

        config = {'nclasses': self.nclasses, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        validf1, testf1 = clf.run(excluded_classes=(0,))
        logging.debug('Valid f1 : {0} Test f1 : {1} for EmoContext\n'
                      .format(validf1, testf1))
        return {'validf1': validf1, 'testf1': testf1,
                'nvalid': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
