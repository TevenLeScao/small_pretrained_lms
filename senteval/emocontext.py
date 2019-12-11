import os
import torch
import numpy as np
from time import time
import json
import logging
import copy

from senteval.tools.validation import SplitClassifier

from models.sentence_encoders import SentenceEncoder
from models.structure import StandardMLP
from models.word_embedders import WordEmbedder
from utils.helpers import \
    make_masks, batch_iter, word_lists_to_lines, lines_to_word_lists, progress_bar_msg, update_training_history
from utils.progress_bar import progress_bar
from configuration import SANITY, GPU, TrainConfig as tconfig
from contextlib import nullcontext


class EmoContext(object):

    def __init__(self, taskpath, seed=111):
        self.seed = seed

        train1, train2, train3, trainlabels = self.loadFiles(taskpath, "train")
        train = zip(train1, train2, train3, trainlabels)

        valid1, valid2, valid3, validlabels = self.loadFiles(taskpath, "dev")
        valid = zip(valid1, valid2, valid3, validlabels)

        test1, test2, test3, testlabels = self.loadFiles(taskpath, "test")
        test = zip(test1, test2, test3, testlabels)

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

        self.dict_label = {'others': 0, 'happy': 1, 'sad': 2, 'angry': 3}
        self.nclasses = len(self.dict_label)
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
        s1 = list(lines_to_word_lists(s1))
        s2 = open(os.path.join(path, "s2." + file_type)).read()
        s2 = list(lines_to_word_lists(s2))
        s3 = open(os.path.join(path, "s3." + file_type)).read()
        s3 = list(lines_to_word_lists(s3))
        labels = open(os.path.join(path, "label." + file_type)).read().split('\n')

        return s1, s2, s3, labels

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
            for batch_num, (sents1, sents2, sents3, labels) in enumerate(
                    batch_iter(data, tconfig.batch_size)):
                sents1, mask1 = make_masks(params.tokenize(sents1))
                sents2, mask2 = make_masks(params.tokenize(sents2))
                sents3, mask3 = make_masks(params.tokenize(sents3))
                if GPU:
                    labels = torch.LongTensor(labels).cuda()
                else:
                    labels = torch.LongTensor(labels)
                enc1, enc2, enc3 = sentence_encoder(word_embedder(sents1, mask1), mask1), \
                                   sentence_encoder(word_embedder(sents2, mask2), mask2), \
                                   sentence_encoder(word_embedder(sents3, mask3), mask3)
                classifier_input = torch.cat((enc1, enc2, enc3, enc1 * enc2, enc2 * enc3, enc3 * enc1), dim=1)
                predictions = classifier(classifier_input)
                loss = classifier.predictions_to_loss(predictions, labels)
                if not validation:
                    loss = loss / tconfig.accumulate
                    loss.backward()
                    for model in models:
                        model.step()

                epoch_losses.append(loss.item())
                predictions = classifier(classifier_input)
                acc = classifier.predictions_to_acc(predictions, labels)
                epoch_accuracies.append(acc.item())

                progress_bar(batch_num, (len(data) // tconfig.batch_size) + 1,
                             msg="{:.4f} {} loss    ".format(np.mean(epoch_losses),
                                                             "validation" if validation else "training"))
        if validation:
            for model in models:
                model.train()
        return np.mean(epoch_losses), np.mean(epoch_accuracies)

    def train(self, params):
        start_time = time()
        training_history = {'time': [], 'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
        best_valid = 0
        start_epoch = 0
        # to make sure we reload with the proper updated learning rate
        restart_memory = 0
        classifier = StandardMLP(params, params.sentence_encoder.sentence_dim * 6, self.nclasses)
        if GPU:
            classifier = classifier.cuda()
        models = {"embedder": params.word_embedder, "encoder": params.sentence_encoder, "classifier": classifier}
        if tconfig.resume_training:
            try:
                for key, model in models.items():
                    print("reloaded {}".format(key))
                    model.load_params(os.path.join(params.current_xp_folder, key))
                training_history = json.load(open(os.path.join(params.current_xp_folder, "training_history.json"), 'r'))
                best_valid = min(training_history['valid_loss'])
                start_epoch = len(training_history['valid_loss'])
            except FileNotFoundError:
                print("Could not find models to load")

        sub_reader = params.get("reader")
        if sub_reader is not None:
            self.data_subwords = {}
            self.data_source = self.data_subwords
            for data_type in ['train', 'valid']:
                sub_list = [list(sub_reader.lines_to_subwords(word_lists_to_lines(self.data[data_type][i]))) \
                            for i in range(3)]
                label_list = [self.dict_label[value] for value in self.data[data_type][3]]
                self.data_subwords[data_type] = list(zip(sub_list[0], sub_list[1], sub_list[2], label_list))
            print(len(self.data_subwords['train'][0]))
            print(len(self.data['valid'][0]))

        for epoch in range(start_epoch, tconfig.max_epoch):
            print("epoch {}".format(epoch))
            train_loss, train_acc = self.epoch_loop(self.data_source['train'], models.values(), params,
                                                    validation=False)
            valid_loss, valid_acc = self.epoch_loop(self.data_source['valid'], models.values(), params, validation=True)
            elapsed_time = time() - start_time
            update_training_history(training_history, elapsed_time, train_loss, train_acc, valid_loss, valid_acc)
            json.dump(training_history, open(os.path.join(params.current_xp_folder, "training_history.json"), 'w'))
            if valid_acc > best_valid:
                best_valid = valid_acc
                restart_memory = 0
                for key, model in models.items():
                    model.save(os.path.join(params.current_xp_folder, key))
            else:
                print("updating LR and re-loading model")
                restart_memory += 1
                for key, model in models.items():
                    model.load_params(os.path.join(params.current_xp_folder, key))
                    model.update_learning_rate(tconfig.lr_decay ** restart_memory)
                if max(model.get_current_learning_rate() for model in models.values()) < tconfig.min_lr:
                    print("min lr {} reached, stopping training".format(tconfig.min_lr))
                    break

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        for key in self.data:
            if key not in self.X:
                self.X[key] = []
            if key not in self.y:
                self.y[key] = []

            input1, input2, input3, mylabels = self.data[key]
            enc_input = []
            n_labels = len(mylabels)
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
            self.y[key] = np.array([self.dict_label[y] for y in mylabels])

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
