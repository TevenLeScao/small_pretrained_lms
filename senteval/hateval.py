import os
import torch
import numpy as np
from time import time
from nltk.tokenize import TweetTokenizer
import csv
import json
import logging
import copy

from senteval.tools.validation import SplitClassifier

from models.sentence_encoders import SentenceEncoder
from models.structure import WordEmbedder, StandardMLP
from utils.helpers import \
    prepare_sentences, batch_iter, word_lists_to_lines, lines_to_word_lists, progress_bar_msg, update_training_history
from utils.progress_bar import progress_bar
from configuration import SANITY, GPU, TrainConfig as tconfig
from contextlib import nullcontext


class HatEval(object):

    def __init__(self, taskpath, seed=111):
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
            for batch_num, (sents, labels) in enumerate(batch_iter(data, tconfig.batch_size)):
                sents, mask = prepare_sentences(sents, params.vocab)
                if GPU:
                    labels = torch.LongTensor(labels).cuda()
                else:
                    labels = torch.LongTensor(labels)
                enc = sentence_encoder(word_embedder(sents, mask), mask)
                classifier_input = enc
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
        classifier = StandardMLP(params, params.sentence_encoder.sentence_dim, self.nclasses)
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
                sub_list = list(sub_reader.lines_to_subwords(word_lists_to_lines(self.data[data_type][0])))
                label_list = self.data[data_type][1]
                self.data_subwords[data_type] = list(zip(sub_list, label_list))
            print(len(self.data_subwords['train']))
            print(len(self.data['valid']))

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
                  'cudaEfficient': True,
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
