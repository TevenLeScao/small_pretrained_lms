import torch

SANITY = False
GPU = torch.cuda.is_available() and not SANITY


class VocabConfig:
    freq_cutoff = 2
    vocab_size = 50000
    max_len_corpus = 1000

    subwords = True
    load_subwords = False
    subwords_model_type = "bpe"
    subwords_vocab_size = 10000
    subwords_joint_vocab_size = 37000


class ModelConfig:
    if SANITY:
        model = "transformer"
        width = 8
        d_ff = 8
        n_head = 2
        depth = 1
    else:
        model = "transformer"
        width = 64
        d_ff = 64
        n_head = 8
        depth = 3



class TrainConfig:
    lr = 0.001
    weight_decay = 0.00001
    batch_size = 8
    clip_grad = 5.0
    lr_decay = 0.2
    max_epoch = 100
    patience = 3
    max_num_trial = 3
    accumulate = 8
