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
from typing import Dict

import torch

from senteval.tools.validation import SplitClassifier

from models.sentence_encoders import SentenceEncoder
from models.structure import StandardMLP
from models.word_embedders import WordEmbedder
from utils.helpers import\
    make_masks, batch_iter, word_lists_to_lines, makedirs, progress_bar_msg, update_training_history
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

    def epoch_loop(self, data, models: Dict, params, frozen_models=(), validation=False):
        epoch_losses = []
        epoch_accuracies = []
        for key, model in models.items():
            if validation or key in frozen_models:
                model.eval()
        if validation:
            context = torch.no_grad()
        else:
            context = nullcontext()
        word_embedder, sentence_encoder, classifier = models.values()
        with context:
            for batch_num, (sents1, sents2, labels) in enumerate(batch_iter(list(zip(*data)), tconfig.batch_size)):
                sents1, mask1 = make_masks(params.tokenize(params, sents1))
                sents2, mask2 = make_masks(params.tokenize(params, sents2))
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
                    for key, model in models.items():
                        if key not in frozen_models:
                            model.step()
                progress_bar(batch_num, (len(data[0]) // tconfig.batch_size) + 1,
                             msg=progress_bar_msg(validation, epoch_losses, epoch_accuracies))
        for key, model in models.items():
            if validation or key in frozen_models:
                model.train()
        return np.mean(epoch_losses), np.mean(epoch_accuracies)

    def train(self, params, frozen_models=()):
        start_time = time()
        training_history = {'time': [], 'train_loss': [], 'train_score': [], 'valid_loss': [], 'valid_score': []}
        best_valid = 0
        start_epoch = 0
        # to make sure we reload with the proper updated learning rate
        restart_memory = 0
        classifier = StandardMLP(params, params.sentence_encoder.sentence_dim * 4, self.n_classes)
        if GPU:
            classifier = classifier.cuda()
        models = {"embedder": params.word_embedder, "encoder": params.sentence_encoder, "classifier": classifier}
        for frozen in frozen_models:
            for param in models[frozen].parameters():
                param.requires_grad = False
        if tconfig.resume_training:
            try:
                for key, model in models.items():
                    model.load_params(osp.join(params.current_xp_folder, key))
                    print("reloaded {}".format(key))
                training_history = json.load(open(osp.join(params.current_xp_folder, "training_history.json"), 'r'))
                best_valid = min(training_history['valid_loss'])
                start_epoch = len(training_history['valid_loss'])
            except FileNotFoundError:
                print("Could not find models to load")

        self.data['train'] = (params.tokenize(params, self.data['train'][0]),
                              params.tokenize(params, self.data['train'][1]),
                              [self.dico_label[value] for value in self.data['train'][2]])
        self.data['valid'] = (params.tokenize(params, self.data['valid'][0]),
                              params.tokenize(params, self.data['valid'][1]),
                              [self.dico_label[value] for value in self.data['valid'][2]])

        for epoch in range(start_epoch, tconfig.max_epoch):
            print("epoch {}".format(epoch))
            train_loss, train_acc = self.epoch_loop(self.data['train'], models, params, frozen_models=frozen_models, validation=False)
            valid_loss, valid_acc = self.epoch_loop(self.data['valid'], models, params, frozen_models=frozen_models, validation=True)
            elapsed_time = time() - start_time
            update_training_history(training_history, elapsed_time, train_loss, train_acc, valid_loss, valid_acc)
            json.dump(training_history, open(osp.join(params.current_xp_folder, "training_history.json"), 'w'))
            if valid_acc >= best_valid:
                best_valid = valid_acc
                restart_memory = 0
                for key, model in models.items():
                    if key not in frozen_models:
                        model.save(osp.join(params.current_xp_folder, key))
            else:
                print("updating LR and re-loading model")
                restart_memory += 1
                for key, model in models.items():
                    if key not in frozen_models:
                        model.load_params(os.path.join(params.current_xp_folder, key))
                        model.update_learning_rate(tconfig.lr_decay ** restart_memory)
                if max(model.get_current_learning_rate() for model in models.values()) < tconfig.min_lr:
                    print("min lr {} reached, stopping training".format(tconfig.min_lr))
                    break

    def run(self, params, batcher):
        if params.train_encoder:
            tconfig.resume_training = False
            self.train(params, frozen_models=("encoder",))
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
                    enc_input.append(np.hstack((enc1, enc2, enc1, enc2, enc1 * enc2, np.abs(enc1 - enc2))))
                if (ii * params.batch_size) % (20000 * params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" %
                                 (100 * ii / n_labels))
                    # we add the training data bit by bit to not overwhelm the memory with one big stacking operation
                    try:
                        self.X[key] = np.vstack((self.X[key], *enc_input))
                    except ValueError:
                        self.X[key] = np.vstack(enc_input)
                    enc_input = []
            self.X[key] = np.vstack((self.X[key], *enc_input))
            self.y[key] = np.array([self.dico_label[y] for y in mylabels])

        config = {'nclasses': 3, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': GPU,
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
