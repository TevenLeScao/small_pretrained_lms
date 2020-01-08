import os
import numpy as np
from nltk.tokenize import TweetTokenizer
import csv
import logging
import copy

from senteval.tools.validation import SplitClassifier

from configuration import SANITY, GPU, TrainConfig as tconfig, ModelConfig as mconfig
from senteval.trainer import Trainer


class HatEval(Trainer):

    def __init__(self, taskpath, seed=111):
        super(HatEval, self).__init__()
        self.seed = seed

        trainsent, trainlabels = self.loadFiles(taskpath, "train")
        validsent, validlabels = self.loadFiles(taskpath, "dev")
        testsent, testlabels = self.loadFiles(taskpath, "test")

        # sort to reduce the batch width
        train_sorted = sorted(list(zip(trainsent, trainlabels)), key=lambda z: len(z[0]))
        valid_sorted = sorted(list(zip(validsent, validlabels)), key=lambda z: len(z[0]))
        test_sorted = sorted(list(zip(testsent, testlabels)), key=lambda z: len(z[0]))

        if SANITY:
            self.data = {"train": list(zip(*train_sorted[:100])),
                         "valid": list(zip(*valid_sorted[:100])),
                         "test": list(zip(*test_sorted[:100]))}
        else:
            self.data = {"train": list(zip(*train_sorted)),
                         "valid": list(zip(*valid_sorted)),
                         "test": list(zip(*test_sorted))}

        self.nclasses = 2
        self.ninputs = 1
        self.data_source = self.data
        self.samples = trainsent + validsent + testsent
        self.training_samples = trainsent

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def do_train_prepare(self, params, prepare):
        return prepare(params, self.training_samples)

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

    def run(self, params, batcher):
        if params.train_encoder:
            tconfig.resume_training = False
            params.sentence_encoder.__init__(word_dim=mconfig.width, sentence_dim=mconfig.sentence_width,
                                             depth=mconfig.encoder_depth)
            if GPU:
                params.sentence_encoder = params.sentence_encoder.cuda()
            self.train(params, frozen_models=("embedder",))
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            sents, labels = self.data[key]
            enc_input = []
            n_labels = len(labels)
            for ii in range(0, n_labels, params.batch_size):
                batch = sents[ii:ii + params.batch_size]
                encoding = batcher(params, batch)
                enc_input.append(encoding)
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
                  'cudaEfficient': GPU,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        validf1, testf1 = clf.run(excluded_classes=(0,))
        logging.debug('Valid f1 : {0} Test f1 : {1} for HatEval\n'
                      .format(validf1, testf1))
        return {'validf1': validf1, 'testf1': testf1,
                'nvalid': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}
