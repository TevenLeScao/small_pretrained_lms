from transformers import *
import torch
from torch import nn, functional
from configuration import VocabConfig as vconfig, ModelConfig as mconfig, TrainConfig as tconfig

# from utils import read_corpus, prepare_sentences
from models.sentence_encoders import SentenceEncoder

# just to keep in mind this is the way to do it
from models.structure import GeneralModel

import warnings

MODEL_CLASSES = {
    'bert': {"config": BertConfig, "model": BertModel, "tokenizer": BertTokenizer,
             "classifier": BertForSequenceClassification},
    'xlnet': {"config": XLNetConfig, "model": XLNetModel, "tokenizer": XLNetTokenizer,
              "classifier": XLNetForSequenceClassification},
    'xlm': {"config": XLMConfig, "model": XLMModel, "tokenizer": XLMTokenizer,
            "classifier": XLMForSequenceClassification},
    'roberta': {"config": RobertaConfig, "model": RobertaModel, "tokenizer": RobertaTokenizer,
                "classifier": RobertaForSequenceClassification},
    'distilbert': {"config": DistilBertConfig, "model": DistilBertModel, "tokenizer": DistilBertTokenizer,
                   "classifier": DistilBertForSequenceClassification},
}


def change_pooler(bert_model: BertModel, pooler: SentenceEncoder):
    bert_model.pooler = pooler


class WordEmbedder(GeneralModel):

    def __init__(self):
        super(WordEmbedder, self).__init__()
        self.vocab_size = vconfig.subwords_vocab_size if vconfig else vconfig.vocab_size
        self.embedder = None

    def forward(self, one_hot_sentences: torch.Tensor, sent_mask: torch.Tensor) -> torch.Tensor:
        pass

    def main_module(self):
        return self.embedder


class BertWordEmbedder(WordEmbedder):

    def __init__(self, vocab_size=vconfig.vocab_size, depth=mconfig.depth, width=mconfig.width, n_head=mconfig.n_head,
                 d_ff=mconfig.d_ff, model=mconfig.model, pretrained=None):
        super(BertWordEmbedder, self).__init__()
        if pretrained:
            self.config = BertConfig.from_pretrained(pretrained)
            self.bert = BertModel.from_pretrained(pretrained)
        else:
            self.config = BertConfig(vocab_size, width, depth, n_head, d_ff)
            self.bert = BertModel(self.config)
        self.embedding_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.depth = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.d_ff = self.config.intermediate_size

        self.embedder = self.bert
        self.optimizer = torch.optim.Adam(self.parameters(), lr=tconfig.lr, weight_decay=tconfig.weight_decay)
        self.pretrained_bert = pretrained

    def forward(self, sentences, sent_mask):
        return self.bert(input_ids=sentences, attention_mask=sent_mask)[0]

    def to_huggingface_format(self, pooler: SentenceEncoder, classifier):
        formatted_model = BertForSequenceClassification(self.config)
        formatted_model.bert = self.bert
        formatted_model.bert.pooler = pooler
        formatted_model.classifier = classifier
        return formatted_model


class TransformerWordEmbedder(WordEmbedder):

    def __init__(self, vocab_size=vconfig.vocab_size, depth=mconfig.depth, width=mconfig.width, n_head=mconfig.n_head,
                 d_ff=mconfig.d_ff, model=mconfig.model, pretrained=None):
        super(TransformerWordEmbedder, self).__init__()
        self.model = model
        if pretrained:
            self.config = MODEL_CLASSES[model]["config"].from_pretrained(pretrained)
            self.embedder = MODEL_CLASSES[model]["model"].from_pretrained(pretrained)
        else:
            # unfortunately config calls are different for each class
            if model == "bert":
                self.config = BertConfig(vocab_size, width, depth, n_head, d_ff)
            elif model == "xlnet":
                self.config = XLNetConfig(vocab_size, width, depth, n_head, d_ff)
            elif model == "xlm":
                self.config = XLMConfig(vocab_size, width, depth, n_head)
            elif model == "roberta":
                warnings.warn("Cannot adapt Roberta model size")
                self.config = RobertaConfig()
            elif model == "distilbert":
                self.config = DistilBertConfig(vocab_size_or_config_json_file=vocab_size, n_layers=depth,
                                               n_heads=n_head, dim=width, hidden_dim=d_ff)
            self.embedder = MODEL_CLASSES[model]["model"](self.config)
        self.embedding_size = get_embedding_size(self.config)

        self.embedder = self.embedder
        self.optimizer = torch.optim.Adam(self.parameters(), lr=tconfig.lr, weight_decay=tconfig.weight_decay)
        self.pretrained_model_name = pretrained

    def forward(self, sentences, sent_mask):
        return self.embedder(input_ids=sentences, attention_mask=sent_mask)[0]

    # untested
    def to_huggingface_format(self, pooler: SentenceEncoder, classifier):
        formatted_model = MODEL_CLASSES[self.model]["classifier"](self.config)
        if self.model in ["xlnet", "xlm"]:
            formatted_model.transformer = self.embedder
            formatted_model.transformer.pooler = pooler
        else:
            formatted_model.__setattr__(self.model, self.embedder)
            formatted_model.__getattr__(self.model).pooler = self.embedder
        formatted_model.classifier = classifier
        return formatted_model


def get_embedding_size(config):
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    elif hasattr(config, "d_model"):
        return config.d_model
    elif hasattr(config, "emb_dim"):
        return config.emb_dim
    elif hasattr(config, "dim"):
        return config.dim
    else:
        raise AttributeError("Unknown model config type")
