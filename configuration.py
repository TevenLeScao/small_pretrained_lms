import os.path as osp
import torch

SANITY = False
GPU = torch.cuda.is_available()
EXPERIMENT_NAME = "SNLI2EmoHateSNLI"


class Paths:
    code_path = osp.dirname(osp.realpath(__file__))
    project_path = osp.join(code_path, "..")
    data_path = osp.join(project_path, "data")
    senteval_data_path = osp.join(data_path, "senteval")
    semeval_data_path = osp.join(data_path, "semeval")
    experiment_path = osp.join(project_path, "experiments", EXPERIMENT_NAME)
    results_path = osp.join(project_path, "results", EXPERIMENT_NAME)
    direct_reload_path = osp.join(experiment_path, "SNLI")


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
        model = "bert"
        depth = 1
        width = 64
        d_ff = 64
        n_head = 1
        sentence_width = 1024
    else:
        model = "bert"
        depth = 6
        width = 256
        d_ff = 512
        n_head = 8
        sentence_width = 1024


class TrainConfig:
    lr = 0.001
    weight_decay = 0.00001
    batch_size = 64
    clip_grad = 5.0
    lr_decay = 0.6
    accumulate = 8
    min_lr = 0.00001
    if SANITY:
        max_epoch = 10
        resume_training = False
    else:
        max_epoch = 100
        resume_training = False
