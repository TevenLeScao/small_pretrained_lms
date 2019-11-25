import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from models.general_model import GeneralModel
from configuration import TrainConfig as tconfig, VocabConfig as vconfig

default_vocab_size = vconfig.subwords_vocab_size if vconfig.subwords else vconfig.vocab_size


class SentimentTransformer(GeneralModel):

    def __init__(self, depth=3, width=64, d_ff=64, n_head=8, vocab_size=default_vocab_size):
        super(SentimentTransformer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, width)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, n_head, d_ff), depth)
        self.regressor = BasicRegressor(width)
        self.optimizer = Adam(self.parameters(), lr=tconfig.lr, weight_decay=tconfig.weight_decay)
        self.initialize()

    def forward(self, sents, sent_mask):
        embedded = self.embeddings(sents)
        encoded = self.encoder(embedded, src_key_padding_mask=sent_mask)
        estimate = self.regressor(encoded, sent_mask)
        return estimate

    def decode_to_loss(self, sents, sent_mask, targets):
        estimates = self(sents, sent_mask)
        return F.mse_loss(estimates, targets)

    def initialize(self):
        # Initialize parameters with Glorot
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform(param)


class BasicRegressor(nn.Module):

    def __init__(self, width):
        super(BasicRegressor, self).__init__()
        self.linear = nn.Linear(width, 1)
        self.activation = nn.Sigmoid()

    def forward(self, encoded, sent_mask):
        encoded[sent_mask.T] = float("-inf")
        pooled = encoded.max(dim=0)[0]
        prediction = self.activation(self.linear(pooled).squeeze(-1))
        return prediction
