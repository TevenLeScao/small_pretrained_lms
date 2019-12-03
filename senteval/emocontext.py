'''emotional context analysis'''
import os
import io
import torch
import emojis
import numpy as np
from time import time
import json

from models.sentence_encoders import SentenceEncoder
from models.structure import WordEmbedder, StandardMLP
from utils.helpers import prepare_sentences, batch_iter, word_lists_to_lines, lines_to_word_lists
from utils.progress_bar import progress_bar
from configuration import GPU, TrainConfig as tconfig
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
        train_sorted = sorted(train, key=lambda z: len(z[0]) + len(z[1]) + len(z[2]))
        dev_sorted = sorted(dev, key=lambda z: len(z[0]) + len(z[1]) + len(z[2]))
        test_sorted = sorted(test, key=lambda z: len(z[0]) + len(z[1]) + len(z[2]))

        self.data = {"train": train_sorted,
                     "dev": dev_sorted,
                     "test": test_sorted}

        self.dict_label = {'other': 0, 'happy': 1, 'sad': 2}
        self.n_classes = 3
        self.data_source = self.data
        self.samples = train1 + train2 + train3 + dev1 + dev2 + dev3 + test1 + test2 + test3
        self.training_samples = train1 + train2 + train3

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def do_train_prepare(self, params, prepare):
        return prepare(params, self.training_samples)

    def loadFiles(self, path: str, file_type: str):
        assert os.path.isdir(path), "Directory %s not found"%path
        assert file_type in ["train", "dev", "test"], "File type must be 'train', 'dev' or 'test'"

        s1 = open(os.path.join(path, "s1."+file_type)).read()
        s1 = list(lines_to_word_lists(s1))
        s2 = open(os.path.join(path, "s2."+file_type)).read()
        s2 = list(lines_to_word_lists(s2))
        s3 = open(os.path.join(path, "s3."+file_type)).read()
        s3 = list(lines_to_word_lists(s3))
        labels = open(os.path.join(path, "label."+file_type)).read().split('\n')

        return s1, s2, s3, labels

    def epoch_loop(self, data, models, params, validation=False):
        epoch_losses = []
        if validation:
            for model in models:
                model.eval()
            context = torch.no_grad()
        else:
            context = nullcontext()
        word_embedder, sentence_encoder, classifier = models
        with context:
            for batch_num, (sents1, sents2, sents3, labels) in enumerate(
                    batch_iter(list(zip(*data)), tconfig.batch_size)):
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
                loss = classifier.decode_to_loss(classifier_input, labels)
                if not validation:
                    loss = loss / tconfig.accumulate
                    loss.backward()
                    for model in models:
                        model.step()
                epoch_losses.append(loss.item())
                progress_bar(batch_num, (len(data[0]) // tconfig.batch_size) + 1,
                             msg="{:.4f} {} loss    ".format(np.mean(epoch_losses),
                                                             "validation" if validation else "training"))
        if validation:
            for model in models:
                model.train()
        return np.mean(epoch_losses)

    def train(self, params, word_embedder: WordEmbedder, sentence_encoder: SentenceEncoder):
        start_time = time()
        training_history = {'time': [], 'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
        best_valid = 0
        start_epoch = 0
        classifier = StandardMLP(params, sentence_encoder.sentence_dim * 6, self.n_classes)
        if GPU:
            classifier = classifier.cuda()
        models = {"embedder": word_embedder, "encoder": sentence_encoder, "classifier": classifier}
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
                self.data_subwords[data_type] = (
                    (list(sub_reader.lines_to_subwords(word_lists_to_lines(self.data[data_type][i])))
                     for i in range(3)),
                    [self.dict_label[value] for value in self.data[data_type][3]])
            print(len(self.data['train'][0]))
            print(len(self.data['valid'][0]))

        for epoch in range(start_epoch, tconfig.max_epoch):
            print("epoch {}".format(epoch))
            train_loss, train_acc = self.epoch_loop(self.data_source['train'], models.values(), params, validation=False)
            valid_loss, valid_acc = self.epoch_loop(self.data_source['valid'], models.values(), params, validation=True)
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
