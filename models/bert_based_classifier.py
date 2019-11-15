#from utils import read_corpus, prepare_sentences
from models.general_model import GeneralModel
import transformers
import torch
from torch import nn,functional

sm = torch.nn.Softmax()

class Bert_based_classifier(GeneralModel):

    def __init__(self,num_labels):
        super(Bert_based_classifier, self).__init__()
        self.error = nn.CrossEntropyLoss()
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.model.config.num_labels = num_labels
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, sentences, sent_mask):
        outputs = self.model(sentences, attention_mask=sent_mask)
        logits = outputs[0]
        return sm(logits)

    def decode_to_loss(self, sentences, sent_mask, targets):
        outputs = self.model(sentences, attention_mask=sent_mask, labels = targets)
        loss = outputs[0]
        return loss