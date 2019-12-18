import torch

SANITY = True
GPU = torch.cuda.is_available()
EXPERIMENT_NAME = "SNLI2EmoHateSNLI_2048" if not SANITY else "sanity"


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
        encoder_depth = 3
    else:
        model = "bert"
        depth = 6
        width = 256
        d_ff = 512
        n_head = 8
        sentence_width = 2048
        encoder_depth = 3


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
        resume_training = True


class TransferConfig:
    lr = 0.001
    weight_decay = 0.00001
    batch_size = 64
    epoch = 10
    optim = "adam,lr=%f,weight_decay=%f" % (lr, weight_decay)
