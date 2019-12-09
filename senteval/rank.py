# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
Image-Caption Retrieval with COCO dataset
'''
from __future__ import absolute_import, division, unicode_literals

import os
import sys
import logging
import numpy as np
from time import time
import json
from contextlib import nullcontext
import os.path as osp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import torch

from senteval.tools.ranking import ImageSentenceRankingPytorch

from models.sentence_encoders import SentenceEncoder
from models.structure import WordEmbedder, StandardMLP
from utils.helpers import prepare_sentences, batch_iter, word_lists_to_lines, update_training_history, progress_bar_msg
from utils.progress_bar import progress_bar
from configuration import SANITY, GPU, TrainConfig as tconfig


class ImageCaptionRetrievalEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task: Image Caption Retrieval *****\n\n')

        # Get captions and image features
        self.seed = seed
        train, dev, test = self.loadFile(task_path)
        self.coco_data = {'train': train, 'dev': dev, 'test': test}
        self.training_samples = self.coco_data['train']['sent']
        self.samples = self.coco_data['train']['sent'] + \
                       self.coco_data['dev']['sent'] + \
                       self.coco_data['test']['sent']

    def do_prepare(self, params, prepare):
        prepare(params, self.samples)

    def do_train_prepare(self, params, prepare):
        return prepare(params, self.training_samples)

    def loadFile(self, fpath):
        coco = {}

        for split in ['train', 'valid', 'test']:
            list_sent = []
            list_img_feat = []
            if sys.version_info < (3, 0):
                with open(os.path.join(fpath, split + '.pkl')) as f:
                    cocodata = pickle.load(f)
            else:
                with open(os.path.join(fpath, split + '.pkl'), 'rb') as f:
                    cocodata = pickle.load(f, encoding='latin1')

            for imgkey in range(len(cocodata['features'])):
                assert len(cocodata['image_to_caption_ids'][imgkey]) >= 5, \
                    cocodata['image_to_caption_ids'][imgkey]
                for captkey in cocodata['image_to_caption_ids'][imgkey][0:5]:
                    sent = cocodata['captions'][captkey]['cleaned_caption']
                    sent += ' .'  # add punctuation to end of sentence in COCO
                    list_sent.append(sent.encode('utf-8').split())
                    list_img_feat.append(cocodata['features'][imgkey])
            assert len(list_sent) == len(list_img_feat) and \
                   len(list_sent) % 5 == 0
            list_img_feat = np.array(list_img_feat).astype('float32')
            coco[split] = {'sent': list_sent, 'imgfeat': list_img_feat}
        return coco['train'], coco['valid'], coco['test']

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

    def train(self, params):
        start_time = time()
        training_history = {'time': [], 'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
        best_valid = 0
        start_epoch = 0
        classifier = StandardMLP(params, params.sentence_encoder.sentence_dim * 4, self.n_classes)
        if GPU:
            classifier = classifier.cuda()
        models = {"embedder": params.word_embedder, "encoder": params.sentence_encoder, "classifier": classifier}
        if tconfig.resume_training:
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
        coco_embed = {'train': {'sentfeat': [], 'imgfeat': []},
                      'dev': {'sentfeat': [], 'imgfeat': []},
                      'test': {'sentfeat': [], 'imgfeat': []}}

        for key in self.coco_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            self.coco_data[key]['sent'] = np.array(self.coco_data[key]['sent'])
            self.coco_data[key]['sent'], idx_sort = np.sort(self.coco_data[key]['sent']), np.argsort(
                self.coco_data[key]['sent'])
            idx_unsort = np.argsort(idx_sort)

            coco_embed[key]['X'] = []
            nsent = len(self.coco_data[key]['sent'])
            for ii in range(0, nsent, params.batch_size):
                batch = self.coco_data[key]['sent'][ii:ii + params.batch_size]
                embeddings = batcher(params, batch)
                coco_embed[key]['sentfeat'].append(embeddings)
            coco_embed[key]['sentfeat'] = np.vstack(coco_embed[key]['sentfeat'])[idx_unsort]
            coco_embed[key]['imgfeat'] = np.array(self.coco_data[key]['imgfeat'])
            logging.info('Computed {0} embeddings'.format(key))

        config = {'seed': self.seed, 'projdim': 1000, 'margin': 0.2}
        clf = ImageSentenceRankingPytorch(train=coco_embed['train'],
                                          valid=coco_embed['dev'],
                                          test=coco_embed['test'],
                                          config=config)

        bestdevscore, r1_i2t, r5_i2t, r10_i2t, medr_i2t, \
        r1_t2i, r5_t2i, r10_t2i, medr_t2i = clf.run()

        logging.debug("\nTest scores | Image to text: \
            {0}, {1}, {2}, {3}".format(r1_i2t, r5_i2t, r10_i2t, medr_i2t))
        logging.debug("Test scores | Text to image: \
            {0}, {1}, {2}, {3}\n".format(r1_t2i, r5_t2i, r10_t2i, medr_t2i))

        return {'devacc': bestdevscore,
                'acc': [(r1_i2t, r5_i2t, r10_i2t, medr_i2t),
                        (r1_t2i, r5_t2i, r10_t2i, medr_t2i)],
                'ndev': len(coco_embed['dev']['sentfeat']),
                'ntest': len(coco_embed['test']['sentfeat'])}
