from transformers import BertConfig, BertModel, BertForSequenceClassification, BertTokenizer
import torch
from torch import nn, functional
from configuration import VocabConfig as vconfig, ModelConfig as mconfig

# from utils import read_corpus, prepare_sentences
from models.structure import *
from models.sentence_encoders import SentenceEncoder


# just to keep in mind this is the way to do it
def change_pooler(bert_model: BertModel, pooler: SentenceEncoder):
    bert_model.pooler = pooler


class TransformerWordEmbedder(WordEmbedder):

    def __init__(self, vocab_size=vconfig.vocab_size, depth=mconfig.depth, width=mconfig.width, n_head=mconfig.n_head,
                 d_ff=mconfig.d_ff):
        super(TransformerWordEmbedder, self).__init__()
        self.config = BertConfig(vocab_size, width, depth, n_head, d_ff)
        self.bert = BertModel(self.config)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=tconfig.lr, weight_decay=tconfig.weight_decay)

    def forward(self, sentences, sent_mask):
        return self.bert(input_ids=sentences, attention_mask=sent_mask)[0]
