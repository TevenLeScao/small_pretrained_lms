# from utils import read_corpus, prepare_sentences
from models.general_model import GeneralModel
import transformers
import torch
from torch import nn, functional
from configuration import VocabConfig as vconfig, ModelConfig as mconfig

sm = torch.nn.Softmax()
vconfig.max_len_corpus = 512  # Otherwise the pre-trained model is unusable
assert vconfig.vocab_size <= 30522  # uses this many words


class BertBasedClassifier(GeneralModel):

    def __init__(self, num_labels):
        super(BertBasedClassifier, self).__init__()
        self.config = transformers.BertConfig(vocab_size_or_config_json_file=vconfig.vocab_size,
                                              num_hidden_layers=mconfig.depth,
                                              num_attention_heads=mconfig.n_head,
                                              max_position_embeddings=vconfig.max_len_corpus,
                                              num_labels=num_labels
                                              )
        # self.model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', config=self.config)
        self.error = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, sentences, sent_mask):
        outputs = self.model(sentences, attention_mask=sent_mask)
        logits = outputs[0]
        return sm(logits)

    def decode_to_loss(self, sentences, sent_mask, targets):
        try:
            assert (targets % 1 == 0).min()  # check if targets are integers
        except AssertionError:
            raise TypeError("BertBasedClassifier only accepts integer targets")
        targets = targets - torch.ones(size=targets.size())
        targets = targets.long()
        outputs = self.model(sentences.t(), attention_mask=sent_mask, labels=targets)
        loss = outputs[0]
        return loss
