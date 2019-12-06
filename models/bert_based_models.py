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
                 d_ff=mconfig.d_ff, load_bert=False):
        super(TransformerWordEmbedder, self).__init__()
        if load_bert:
            self.config = BertConfig.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.config = BertConfig(vocab_size, width, depth, n_head, d_ff)
            self.bert = BertModel(self.config)
        self.embedder = self.bert
        self.optimizer = torch.optim.Adam(self.parameters(), lr=tconfig.lr, weight_decay=tconfig.weight_decay)
        self.pretrained_bert = load_bert

    def forward(self, sentences, sent_mask):
        return self.bert(input_ids=sentences, attention_mask=sent_mask)[0]

    def to_huggingface_format(self, pooler: SentenceEncoder, classifier):
        formatted_model = BertForSequenceClassification(self.config)
        formatted_model.bert = self.bert
        formatted_model.bert.pooler = pooler
        formatted_model.classifier = classifier
        return formatted_model

