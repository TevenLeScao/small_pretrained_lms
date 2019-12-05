'''emotional context analysis'''
import os
import torch
import numpy as np
from time import time
from nltk.tokenize import TweetTokenizer
import csv
import json

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

        devsent, devlabels = self.loadFiles(taskpath, "dev")

        testsent, testlabels = self.loadFiles(taskpath, "test")

        # sort to reduce the batch width
        train_sorted = sorted(list(zip(trainsent, trainlabels)), key=lambda z: len(z[0]))
        dev_sorted = sorted(list(zip(devsent, devlabels)), key=lambda z: len(z[0]))
        test_sorted = sorted(list(zip(testsent, testlabels)), key=lambda z: len(z[0]))

        if SANITY:
            self.data = {"train": list(zip(*train_sorted[:100])),
                         "dev": list(zip(*dev_sorted[:100])),
                         "test": list(zip(*test_sorted[:100]))}
        else:
            self.data = {"train": list(zip(*train_sorted)),
                         "dev": list(zip(*dev_sorted)),
                         "test": list(zip(*test_sorted))}

        self.n_classes = 2
        self.data_source = self.data
        self.samples = trainsent + devsent + testsent
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
        classifier = StandardMLP(params, params.sentence_encoder.sentence_dim, self.n_classes)
        if GPU:
            classifier = classifier.cuda()
        models = {"embedder": params.word_embedder, "encoder": params.sentence_encoder, "classifier": classifier}
        if tconfig.load_models:
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
            for data_type in ['train', 'dev']:
                sub_list = list(sub_reader.lines_to_subwords(word_lists_to_lines(self.data[data_type][0])))
                label_list = self.data[data_type][1]
                self.data_subwords[data_type] = list(zip(sub_list, label_list))
            print(len(self.data_subwords['train']))
            print(len(self.data['dev']))

        for epoch in range(start_epoch, tconfig.max_epoch):
            print("epoch {}".format(epoch))
            train_loss, train_acc = self.epoch_loop(self.data_source['train'], models.values(), params,
                                                    validation=False)
            valid_loss, valid_acc = self.epoch_loop(self.data_source['dev'], models.values(), params, validation=True)
            elapsed_time = time() - start_time
            update_training_history(training_history, elapsed_time, train_loss, train_acc, valid_loss, valid_acc)
            json.dump(training_history, open(os.path.join(params.current_xp_folder, "training_history.json"), 'w'))
            if valid_acc > best_valid:
                best_valid = valid_acc
                for key, model in models.items():
                    model.save(os.path.join(params.current_xp_folder, key))
            else:
                print("updating LR")
                for key, model in models.items():
                    model.load_params(os.path.join(params.current_xp_folder, key))
                    model.update_learning_rate(tconfig.lr_decay)
                    if model.get_current_learning_rate() < tconfig.min_lr:
                        break

    def run(self, params, batcher):
        self.examples = {}
        for key in self.data_source:
            if key not in self.examples:
                self.examples[key] = []
            sents, labels = self.data_source[key]
            enc_input = []
            n_labels = len(labels)
            for ii in range(0, n_labels, params.batch_size):
                batch = sents[ii:ii + params.batch_size]
                enc_input.append(batcher(params, batch))
                if ii % 200 == 0:
                    print("PROGRESS (encoding %s): %.2f%%" % (key, 100 * ii / n_labels))
            labels = torch.LongTensor(labels)
            self.examples[key] = (torch.cat(enc_input, dim=0), labels)
        sentence_dim = self.examples['train'][0].shape[1]

        classifier = StandardMLP(params, sentence_dim, self.n_classes)
        train_data = self.examples['train']
        train_data = [(train_data[0][i], train_data[1][i]) for i in range(train_data[0].shape[0])]
        for embed, targets in batch_iter(train_data, params.batch_size):
            embed = torch.stack(embed)
            targets = torch.LongTensor(targets)
            if GPU:
                embed = embed.cuda()
                targets = targets.cuda()
            predictions = classifier(embed)
            loss = classifier.predictions_to_loss(predictions, targets)
            loss.backward()
            classifier.step()

        dev_embed, dev_labels = self.examples['dev']
        test_embed, test_labels = self.examples['test']
        if GPU:
            dev_embed, dev_labels = dev_embed.cuda(), dev_labels.cuda()
            test_embed, test_labels = test_embed.cuda(), test_labels.cuda()

        with torch.no_grad():
            dev_scores = classifier(dev_embed)
            test_scores = classifier(test_embed)
            dev_loss = classifier.predictions_to_loss(dev_scores, dev_labels).item()
            dev_acc = classifier.predictions_to_acc(dev_scores, dev_labels).item()
            dev_f1 = classifier.predictions_to_f1(dev_scores, dev_labels).item()
            test_loss = classifier.predictions_to_loss(test_scores, test_labels).item()
            test_acc = classifier.predictions_to_acc(test_scores, test_labels).item()
            test_f1 = classifier.predictions_to_f1(test_scores, test_labels).item()

        return {'devacc': dev_acc, 'acc': test_acc,
                'devloss': dev_loss, 'loss': test_loss,
                'devf1': dev_f1, 'f1': test_f1,
                'ndev': len(self.data['dev'][0]),
                'ntest': len(self.data['test'][0])}
