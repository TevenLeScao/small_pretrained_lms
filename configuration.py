import os.path as osp
import torch

SANITY = False
GPU = torch.cuda.is_available()
EXPERIMENT_NAME = "first_long_training"


class Paths:
    code_path = osp.dirname(osp.realpath(__file__))
    project_path = osp.join(code_path, "..")
    data_path = osp.join(project_path, "data")
    senteval_data_path = osp.join(data_path, "senteval")
    experiment_path = osp.join(project_path, "experiments", EXPERIMENT_NAME)
    results_path = osp.join(project_path, "results", EXPERIMENT_NAME)


class VocabConfig:
    freq_cutoff = 2

    subwords = True
    load_subwords = False
    subwords_model_type = "bpe"
    subwords_vocab_size = 10000
    subwords_joint_vocab_size = 37000

    vocab_size = subwords_vocab_size if subwords else 50000


class ModelConfig:
    if SANITY:
        # model = "transformer"
        # depth = 1
        # width = 8
        # d_ff = 8
        # n_head = 2
        # sentence_width = 16
        model = "bert"
        depth = 8
        width = 512
        d_ff = 1024
        n_head = 8
        sentence_width = 1024
    else:
        model = "bert"
        depth = 1
        width = 64
        d_ff = 64
        n_head = 1
        sentence_width = 1024


class TrainConfig:
    load_models = True
    lr = 0.001
    weight_decay = 0.00001
    batch_size = 64
    clip_grad = 5.0
    lr_decay = 0.7
    max_epoch = 100
    patience = 3
    max_num_trial = 3
    accumulate = 8
    min_lr = 0.00001
