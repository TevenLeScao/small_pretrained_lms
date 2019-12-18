import os
from os import path as osp
import torch
import numpy as np
from time import time
import json
from typing import Dict
from contextlib import nullcontext

from models.structure import StandardMLP
from utils.helpers import \
    make_masks, batch_iter, progress_bar_msg, update_training_history
from utils.progress_bar import progress_bar
from configuration import GPU, TrainConfig as tconfig


class Trainer(object):

    def __init__(self):
        self.nclasses = None
        self.ninputs = None

    def epoch_loop(self, data, models: Dict, frozen_models=(), validation=False):
        epoch_losses = []
        epoch_f1s = []
        for key, model in models.items():
            if validation or key in frozen_models:
                model.eval()
        if validation:
            context = torch.no_grad()
        else:
            context = nullcontext()
        word_embedder, sentence_encoder, classifier = models.values()
        with context:
            for batch_num, batch in enumerate(batch_iter(data, tconfig.batch_size)):
                batch_sentences = [batch[i] for i in range(self.ninputs)]
                labels = batch[self.ninputs]
                batch_inputs = [make_masks(sents) for sents in batch_sentences]
                if GPU:
                    labels = torch.LongTensor(labels).cuda()
                else:
                    labels = torch.LongTensor(labels)
                encs = [sentence_encoder(word_embedder(sents, mask), mask) for sents, mask in batch_inputs]
                classifier_input = torch.cat(encs, dim=1)
                predictions = classifier(classifier_input)
                loss = classifier.predictions_to_loss(predictions, labels)
                if not validation:
                    loss = loss / tconfig.accumulate
                    loss.backward()
                    for key, model in models.items():
                        if key not in frozen_models:
                            try:
                                model.step()
                            except RuntimeError:
                                pass

                epoch_losses.append(loss.item())
                predictions = classifier(classifier_input)
                f1 = classifier.emocontext_f1(predictions, labels)
                epoch_f1s.append(f1)

                progress_bar(batch_num, (len(data) // tconfig.batch_size) + 1,
                             msg=progress_bar_msg(validation, epoch_losses, epoch_f1s))

        for key, model in models.items():
            if validation or key in frozen_models:
                model.train()
        return np.mean(epoch_losses), np.mean(epoch_f1s)

    def train(self, params, frozen_models=()):
        start_time = time()
        training_history = {'time': [], 'train_loss': [], 'train_score': [], 'valid_loss': [], 'valid_score': []}
        best_valid = 0
        start_epoch = 0
        # to make sure we reload with the proper updated learning rate
        restart_memory = 0
        classifier = StandardMLP(params, params.sentence_encoder.sentence_dim * self.ninputs, self.nclasses)
        if GPU:
            classifier = classifier.cuda()
        models = {"embedder": params.word_embedder, "encoder": params.sentence_encoder, "classifier": classifier}
        for frozen in frozen_models:
            for param in models[frozen].parameters():
                param.requires_grad = False
        if tconfig.resume_training:
            try:
                for key, model in models.items():
                    model.load_params(os.path.join(params.current_xp_folder, key))
                    print("reloaded {}".format(key))
                training_history = json.load(open(os.path.join(params.current_xp_folder, "training_history.json"), 'r'))
                best_valid = min(training_history['valid_loss'])
                start_epoch = len(training_history['valid_loss'])
            except FileNotFoundError:
                print("Could not find models to load")

        self.data_subwords = {}
        self.data_source = self.data_subwords
        for data_type in ['train', 'valid']:
            text_list = (params.tokenize(params, self.data[data_type][i]) for i in range(self.ninputs))
            label_list = self.data[data_type][self.ninputs]
            self.data_subwords[data_type] = list(zip(*text_list, label_list))

        for epoch in range(start_epoch, tconfig.max_epoch):
            print("epoch {}".format(epoch))
            train_loss, train_f1 = self.epoch_loop(self.data_source['train'], models,
                                                   frozen_models=frozen_models, validation=False)
            valid_loss, valid_f1 = self.epoch_loop(self.data_source['valid'], models,
                                                   frozen_models=frozen_models, validation=True)
            elapsed_time = time() - start_time
            update_training_history(training_history, elapsed_time, train_loss, train_f1, valid_loss, valid_f1)
            json.dump(training_history, open(os.path.join(params.current_xp_folder, "training_history.json"), 'w'))
            if valid_f1 >= best_valid:
                best_valid = valid_f1
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
                if max(model.get_current_learning_rate() for key, model in models.items() if key not in frozen_models) \
                        < tconfig.min_lr:
                    print("min lr {} reached, stopping training".format(tconfig.min_lr))
                    break
