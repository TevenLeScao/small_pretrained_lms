'''emotional context analysis'''
import os
import torch
import numpy as np
from time import time
import json

from models.sentence_encoders import SentenceEncoder
from models.structure import WordEmbedder, StandardMLP
from utils.helpers import \
    prepare_sentences, batch_iter, word_lists_to_lines, lines_to_word_lists, progress_bar_msg, update_training_history
from utils.progress_bar import progress_bar
from configuration import SANITY, GPU, TrainConfig as tconfig, TransferConfig as transconfig
from contextlib import nullcontext


class EmoContext(object):

    def __init__(self, taskpath, seed=111):
        self.seed = seed

        train1, train2, train3, trainlabels = self.loadFiles(taskpath, "train")
        train = zip(train1, train2, train3, trainlabels)

        dev1, dev2, dev3, devlabels = self.loadFiles(taskpath, "dev")
        dev = zip(dev1, dev2, dev3, devlabels)

        test1, test2, test3, testlabels = self.loadFiles(taskpath, "test")
        test = zip(test1, test2, test3, testlabels)

        # sort to reduce the batch width
        train_sorted = sorted(train, key=lambda z: len(z[0]) + len(z[1]) + len(z[2]))[1:]
        dev_sorted = sorted(dev, key=lambda z: len(z[0]) + len(z[1]) + len(z[2]))[1:]
        test_sorted = sorted(test, key=lambda z: len(z[0]) + len(z[1]) + len(z[2]))[1:]

        if SANITY:
            self.data = {"train": list(zip(*train_sorted[:100])),
                         "dev": list(zip(*dev_sorted[:100])),
                         "test": list(zip(*test_sorted[:100]))}
        else:
            self.data = {"train": list(zip(*train_sorted)),
                         "dev": list(zip(*dev_sorted)),
                         "test": list(zip(*test_sorted))}

        self.dict_label = {'others': 0, 'happy': 1, 'sad': 2, 'angry': 3}
        self.n_classes = len(self.dict_label)
        self.data_source = self.data
        self.samples = train1 + train2 + train3 + dev1 + dev2 + dev3 + test1 + test2 + test3
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
                sents1, mask1 = prepare_sentences(sents1, params.vocab)
                sents2, mask2 = prepare_sentences(sents2, params.vocab)
                sents3, mask3 = prepare_sentences(sents3, params.vocab)
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
        classifier = StandardMLP(params, params.sentence_encoder.sentence_dim * 6, self.n_classes)
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
            for data_type in ['train', 'dev']:
                sub_list = [list(sub_reader.lines_to_subwords(word_lists_to_lines(self.data[data_type][i]))) \
                            for i in range(3)]
                label_list = [self.dict_label[value] for value in self.data[data_type][3]]
                self.data_subwords[data_type] = list(zip(sub_list[0], sub_list[1], sub_list[2], label_list))
            print(len(self.data_subwords['train'][0]))
            print(len(self.data['dev'][0]))

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
        print("******* Transferring to the EmoContext task ********")
        params["optim"] = transconfig.optim
        self.examples = {}
        for key in self.data_source:
            if key not in self.examples:
                self.examples[key] = []
            s1, s2, s3, labels = self.data_source[key]
            enc_input = []
            n_labels = len(labels)
            for ii in range(0, n_labels, params.batch_size):
                batch1 = s1[ii:ii + params.batch_size]
                batch2 = s2[ii:ii + params.batch_size]
                batch3 = s3[ii:ii + params.batch_size]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    enc3 = batcher(params, batch3)
                    enc_input.append(torch.cat((enc1, enc2, enc3, enc1 * enc2, enc2 * enc3, enc3 * enc1), dim=1))
                if ii % 200 == 0:
                    print("PROGRESS (encoding %s): %.2f%%" % (key, 100 * ii / n_labels))
            labels = torch.LongTensor([self.dict_label[y] for y in labels])
            self.examples[key] = (torch.cat(enc_input, dim=0), labels)
        sentence_dim = self.examples['train'][0].shape[1]

        classifier = StandardMLP(params, sentence_dim, self.n_classes)
        train_data = self.examples['train']
        train_data = [(train_data[0][i], train_data[1][i]) for i in range(train_data[0].shape[0])]
        print("Training a classifier layer")
        for epoch in range(transconfig.epoch):
            for embed, targets in batch_iter(train_data, transconfig.batch_size):
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
            dev_f1 = classifier.emocontext_f1(dev_scores, dev_labels, included_classes=(1, 2, 3))
            test_loss = classifier.predictions_to_loss(test_scores, test_labels).item()
            test_acc = classifier.predictions_to_acc(test_scores, test_labels).item()
            test_f1 = classifier.emocontext_f1(test_scores, test_labels, included_classes=(1, 2, 3))

        return {'devacc': dev_acc, 'acc': test_acc,
                'devloss': dev_loss, 'loss': test_loss,
                'devf1': dev_f1, 'f1': test_f1,
                'ndev': len(self.data['dev'][0]),
                'ntest': len(self.data['test'][0])}
