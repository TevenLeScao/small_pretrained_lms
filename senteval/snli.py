# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SNLI - Entailment
'''
from __future__ import absolute_import, division, unicode_literals

import json
from time import time
import codecs
import os
import os.path as osp
import io
import copy
import logging
import numpy as np
from contextlib import nullcontext

import torch

from senteval.tools.validation import SplitClassifier

from models.sentence_encoders import SentenceEncoder
from models.structure import WordEmbedder, StandardMLP
from utils.helpers import prepare_sentences, batch_iter, word_lists_to_lines, makedirs
from utils.progress_bar import progress_bar
from configuration import SANITY, GPU, TrainConfig as tconfig


class SNLI(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : SNLI Entailment*****\n\n')
        self.seed = seed

        train1 = self.loadFile(os.path.join(taskpath, 's1.train'))
        train2 = self.loadFile(os.path.join(taskpath, 's2.train'))
        trainlabels = io.open(os.path.join(taskpath, 'labels.train'),
                              encoding='utf-8').read().splitlines()

        valid1 = self.loadFile(os.path.join(taskpath, 's1.dev'))
        valid2 = self.loadFile(os.path.join(taskpath, 's2.dev'))
        validlabels = io.open(os.path.join(taskpath, 'labels.dev'),
                              encoding='utf-8').read().splitlines()

        test1 = self.loadFile(os.path.join(taskpath, 's1.test'))
        test2 = self.loadFile(os.path.join(taskpath, 's2.test'))
        testlabels = io.open(os.path.join(taskpath, 'labels.test'),
                             encoding='utf-8').read().splitlines()

        # sort data (by s2 first) to reduce padding
        sorted_train = sorted(zip(train2, train1, trainlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))

        sorted_valid = sorted(zip(valid2, valid1, validlabels),
                              key=lambda z: (len(z[0]), len(z[1]), z[2]))

        sorted_test = sorted(zip(test2, test1, testlabels),
                             key=lambda z: (len(z[0]), len(z[1]), z[2]))

        if SANITY:
            sorted_train = sorted_train[0:100]
            sorted_valid = sorted_valid[0:100]
            sorted_test = sorted_test[0:100]

        train2, train1, trainlabels = map(list, zip(*sorted_train))
        valid2, valid1, validlabels = map(list, zip(*sorted_valid))
        test2, test1, testlabels = map(list, zip(*sorted_test))

        self.training_samples = train1 + train2
        self.samples = train1 + train2 + valid1 + valid2 + test1 + test2
        self.data = {'train': (train1, train2, trainlabels),
                     'valid': (valid1, valid2, validlabels),
                     'test': (test1, test2, testlabels)
                     }
        self.n_classes = 3

        self.dico_label = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def do_train_prepare(self, params, prepare):
        return prepare(params, self.training_samples)

    def loadFile(self, fpath):
        with codecs.open(fpath, 'rb', 'latin-1') as f:
            return [line.split() for line in
                    f.read().splitlines()]

    def epoch_loop(self, data, models, params, validation=False):
        epoch_losses = []
        epoch_accuracies = []
        if validation:
            for model in models:
                model.eval()
            context = torch.no_grad()
        else:
            context = nullcontext()
        word_embedder, sentence_encoder, classifier = models
        with context:
            for batch_num, (sents1, sents2, labels) in enumerate(batch_iter(list(zip(*data)), tconfig.batch_size)):
                sents1, mask1 = prepare_sentences(sents1, params.vocab)
                sents2, mask2 = prepare_sentences(sents2, params.vocab)
                if GPU:
                    labels = torch.LongTensor(labels).cuda()
                else:
                    labels = torch.LongTensor(labels)
                enc1, enc2 = sentence_encoder(word_embedder(sents1, mask1), mask1), \
                             sentence_encoder(word_embedder(sents2, mask2), mask2)
                classifier_input = torch.cat((enc1, enc2, enc1 * enc2, (enc1 - enc2).abs()), dim=1)
                predictions = classifier(classifier_input)
                loss = classifier.predictions_to_loss(predictions, labels)
                acc = classifier.predictions_to_acc(predictions, labels)
                epoch_accuracies.append(acc.item())
                epoch_losses.append(loss.item())
                if not validation:
                    loss = loss / tconfig.accumulate
                    loss.backward()
                    for model in models:
                        model.step()
                progress_bar(batch_num, (len(data[0]) // tconfig.batch_size) + 1,
                             msg=progress_bar_msg(validation, epoch_losses, epoch_accuracies))
        if validation:
            for model in models:
                model.train()
        return np.mean(epoch_losses), np.mean(epoch_accuracies)

    def train(self, params, word_embedder: WordEmbedder, sentence_encoder: SentenceEncoder):
        start_time = time()
        training_history = {'time': [], 'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
        best_valid = 0
        start_epoch = 0
        classifier = StandardMLP(params, sentence_encoder.sentence_dim * 4, self.n_classes)
        if GPU:
            classifier = classifier.cuda()
        models = {"embedder": word_embedder, "encoder": sentence_encoder, "classifier": classifier}
        if tconfig.load_models:
            try:
                for key, model in models.items():
                    print("reloaded {}".format(key))
                    model.load_params(osp.join(params.current_xp_folder, key))
                training_history = json.load(open(osp.join(params.current_xp_folder, "training_history.json"), 'r'))
                best_valid = min(training_history['valid_loss'])
                start_epoch = len(training_history['valid_loss'])
            except FileNotFoundError:
                print("Could not find models to load")

        sub_reader = params.get("reader")
        if sub_reader is not None:
            self.data['train'] = (list(sub_reader.lines_to_subwords(word_lists_to_lines(self.data['train'][0]))),
                                  list(sub_reader.lines_to_subwords(word_lists_to_lines(self.data['train'][1]))),
                                  [self.dico_label[value] for value in self.data['train'][2]])
            print(len(self.data['train'][0]))
            self.data['valid'] = (list(sub_reader.lines_to_subwords(word_lists_to_lines(self.data['valid'][0]))),
                                  list(sub_reader.lines_to_subwords(word_lists_to_lines(self.data['valid'][1]))),
                                  [self.dico_label[value] for value in self.data['valid'][2]])
            print(len(self.data['valid'][0]))

        for epoch in range(start_epoch, tconfig.max_epoch):
            print("epoch {}".format(epoch))
            train_loss, train_acc = self.epoch_loop(self.data['train'], models.values(), params, validation=False)
            valid_loss, valid_acc = self.epoch_loop(self.data['valid'], models.values(), params, validation=True)
            elapsed_time = time() - start_time
            update_training_history(training_history, elapsed_time, train_loss, train_acc, valid_loss, valid_acc)
            json.dump(training_history, open(osp.join(params.current_xp_folder, "training_history.json"), 'w'))
            if valid_acc > best_valid:
                best_valid = valid_acc
                for key, model in models.items():
                    model.save(osp.join(params.current_xp_folder, key))
            else:
                print("updating LR")
                for key, model in models.items():
                    model.load_params(osp.join(params.current_xp_folder, key))
                    model.update_learning_rate(tconfig.lr_decay)
                    if model.get_current_learning_rate() < tconfig.min_lr:
                        break

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1, input2, mylabels = self.data[key]
            enc_input = []
            n_labels = len(mylabels)
            for ii in range(0, n_labels, params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    enc_input.append(np.hstack((enc1, enc2, enc1 * enc2,
                                                np.abs(enc1 - enc2))))
                if (ii * params.batch_size) % (20000 * params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = [self.dico_label[y] for y in mylabels]

        config = {'nclasses': 3, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'nhid': params.nhid, 'noreg': True}

        config_classifier = copy.deepcopy(params.classifier)
        config_classifier['max_epoch'] = 15
        config_classifier['epoch_size'] = 1
        config['classifier'] = config_classifier

        clf = SplitClassifier(self.X, self.y, config)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for SNLI\n'
                      .format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.data['valid'][0]),
                'ntest': len(self.data['test'][0])}


def progress_bar_msg(validation, epoch_losses, epoch_accuracies):
    letter = "v" if validation else "t"
    return "{:.4f} {}. loss | {:.3f} {}. acc.   ".format(np.mean(epoch_losses), letter, np.mean(epoch_accuracies), letter)


def update_training_history(history, time, train_loss, train_acc, valid_loss, valid_acc):
    history['time'].append(time)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['valid_loss'].append(valid_loss)
    history['valid_acc'].append(valid_acc)
