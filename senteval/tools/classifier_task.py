import os
import torch
import numpy as np
from time import time
import json

from models.sentence_encoders import SentenceEncoder
from models.word_embedders import WordEmbedder
from models.structure import StandardMLP
from utils.helpers import \
    prepare_sentences, batch_iter, word_lists_to_lines, lines_to_word_lists, progress_bar_msg, update_training_history
from utils.progress_bar import progress_bar
from configuration import SANITY, GPU, TrainConfig as tconfig, TransferConfig as transconfig
from contextlib import nullcontext

class Classifier_task(object):
    """
       An abstract class that facilitates classifying task training and evaluation.
       It needs the following to work:
        These parameters need to be specified in the __init__ function BEFORE the super.__init__ call.
         * sorting_key (lambda function) - a lambda function that sorts the data rows
         * dict_label (dict) - dictionary of the label names and their values (0-n)
         * classifier_input_multiplier (int) - multiplier for classifier layer, i.e. how many times the sentence embedding
            length is in the input. For example if the classifier layer inputs (enc1, enc2, enc1*enc2) where enc1,2 are
            sentence encodings, the multiplier is 3.
         * task_name (string) - name of the task for output reasons
        Furthermore we need the following functions:
         * loadFiles(taskpath, data_type): a function for loading the data. It should output n lists of input sentences
            split by words, where n is the number of inputs per example, and one list of labels (in string form)
         * batch_data_to_sent_embeddings(models, batch_features, params): Converts the batch features into sent embeddings.
            i.e. application of the word and sent embedders on the data. Outputs a tensor of the size
            sent_embedding_dim*classifier_input_multiplier
         * prepare_sent_embeddings(current_data, params, batcher): A function that converts the data into static embeddings.
            Should use the batcher. Current_data is the list of lists of (sub)words, output is a list of tensors ready
            for the classifier (i.e. of the length sent_embedding_dim*classifier_input_multiplier)
    """
    def __init__(self, taskpath, seed=111):
        assert self.sorting_key, "Sorting key not specified"
        assert self.classifier_input_multiplier, "Classifier input multiplier not specified"
        assert self.dict_label, "label dictionary not specified"
        self.seed = seed
        train = self.loadFiles(taskpath, "train")
        dev = self.loadFiles(taskpath, "dev")
        test = self.loadFiles(taskpath, "test")

        train_sorted = sorted(zip(*train), key=self.sorting_key)
        dev_sorted = sorted(zip(*dev), key=self.sorting_key)
        test_sorted = sorted(zip(*test), key=self.sorting_key)

        if SANITY:
            self.data = {"train": list(zip(*train_sorted[:100])),
                         "dev": list(zip(*dev_sorted[:100])),
                         "test": list(zip(*test_sorted[:100]))}
        else:
            self.data = {"train": list(zip(*train_sorted)),
                         "dev": list(zip(*dev_sorted)),
                         "test": list(zip(*test_sorted))}

        self.n_classes = len(self.dict_label)
        self.data_source = self.data
        self.samples = train[:-1] + dev[:-1] + test[:-1]
        self.samples = sum(self.samples, [])  # concatenate
        self.training_samples = sum(train[:-1], [])  # concatenate
        if not hasattr(self, 'eval_metrics'):
            self.eval_metrics = ['loss', 'acc']

    def loadFiles(self, taskpath, data_type: str):
        raise NotImplementedError("You need to implement a loading method that outputs the tuple with the format\
        (sentence 1, sentence 2, ... sentence n, label)")

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def do_train_prepare(self, params, prepare):
        return prepare(params, self.training_samples)

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
            for batch_num, batch_data in enumerate(
                    batch_iter(data, tconfig.batch_size)):
                (batch_features, labels) = batch_data
                if GPU:
                    labels = torch.LongTensor(labels).cuda()
                else:
                    labels = torch.LongTensor(labels)
                classifier_input = self.batch_data_to_sent_embeddings(models, batch_features, params)
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
        training_history = {'time': [], 'train_loss': [], 'train_score': [], 'valid_loss': [], 'valid_score': []}
        best_valid = 0
        start_epoch = 0
        # to make sure we reload with the proper updated learning rate
        restart_memory = 0
        classifier = StandardMLP(params, params.sentence_encoder.sentence_dim*self.classifier_input_multiplier, self.n_classes)
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
                sub_list = [list(sub_reader.lines_to_subwords(word_lists_to_lines(list_of_word_lists))) \
                            for list_of_word_lists in self.data[data_type][:-1]]
                label_list = [self.dict_label[value] for value in self.data[data_type][-1]]
                self.data_subwords[data_type] = list(zip(zip(*sub_list), label_list))
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
        print("******* Transferring to the {} task ********".format(self.task_name))
        params["optim"] = transconfig.optim
        self.examples = {}
        for key in self.data_source:
            if key not in self.examples:
                self.examples[key] = []
            current_data = self.data_source[key][:-1]
            labels = self.data_source[key][-1]
            print("Encoding %s"%key)
            enc_input = self.prepare_sent_embeddings(current_data, params, batcher)
            labels = torch.LongTensor([self.dict_label[y] for y in labels])
            self.examples[key] = (torch.cat(enc_input, dim=0), labels)
        sentence_dim = self.examples['train'][0].shape[1]

        classifier = StandardMLP(params, sentence_dim, self.n_classes)
        train_data = self.examples['train']
        train_data = [(train_data[0][i], train_data[1][i]) for i in range(train_data[0].shape[0])]
        print("Training a classifier layer")
        for epoch in range(transconfig.epoch):
            progress_bar(epoch, transconfig.epoch)
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

        output = {}
        with torch.no_grad():
            dev_scores = classifier(dev_embed)
            test_scores = classifier(test_embed)
            if "loss" in self.eval_metrics:
                dev_loss = classifier.predictions_to_loss(dev_scores, dev_labels).item()
                test_loss = classifier.predictions_to_loss(test_scores, test_labels).item()
                output['devloss'] = dev_loss
                output['loss'] = test_loss
            if "acc" in self.eval_metrics:
                dev_acc = classifier.predictions_to_acc(dev_scores, dev_labels).item()
                test_acc = classifier.predictions_to_acc(test_scores, test_labels).item()
                output['devacc'] = dev_acc
                output['testacc'] = test_acc
            if 'f1' in self.eval_metrics:
                if not hasattr(self, 'f1_excluded_classes'):
                    self.f1_excluded_classes = ()
                excluded_classes_index = (self.dict_label[lab] for lab in self.f1_excluded_classes)
                dev_f1 = classifier.emocontext_f1(dev_scores, dev_labels, excluded_classes=excluded_classes_index)
                f1 = classifier.emocontext_f1(test_scores, test_labels, excluded_classes=excluded_classes_index)
                output['f1 eval classes excluded'] = \
                    'None' if self.f1_excluded_classes == () else str(self.f1_excluded_classes)
                output['devf1'] = dev_f1
                output['f1'] = f1
            if 'mcc' in self.eval_metrics:
                dev_mcc = classifier.predictions_to_mcc(dev_scores, dev_labels)
                mcc = classifier.predictions_to_mcc(test_scores, test_labels)
                output['devmcc'] = dev_mcc
                output['mcc'] = mcc

        return output
