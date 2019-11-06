class VocabConfig:
    freq_cutoff = 2
    vocab_size = 50000
    max_len_corpus = 1000

    subwords = False
    load_subwords = False
    subwords_model_type = "bpe"
    subwords_vocab_size = 8000
    subwords_joint_vocab_size = 37000


class TrainConfig:
    lr = 1
    weight_decay = 0.00001
    batch_size = 16
    clip_grad = 5.0
    lr_decay = 0.2
    max_epoch = 200
    patience = 3
    max_num_trial = 3
    accumulate = 8
