import torch

SANITY = False
GPU = torch.cuda.is_available() and not SANITY


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
        model = "transformer"
        depth = 1
        width = 8
        d_ff = 8
        n_head = 2
        sentence_width = 16
    else:
        model = "transformer"
        depth = 3
        width = 256
        d_ff = 256
        n_head = 8
        sentence_width = 1024



class TrainConfig:
    lr = 0.001
    weight_decay = 0.00001
    batch_size = 16
    clip_grad = 5.0
    lr_decay = 0.2
    max_epoch = 100
    patience = 3
    max_num_trial = 3
    accumulate = 8
