# from utils import read_corpus, prepare_sentences
from models.general_model import GeneralModel
import transformers
import torch
from torch import nn, functional
from configuration import VocabConfig, ModelConfig

sm = torch.nn.Softmax()
vocabConfig = VocabConfig()
modelConfig = ModelConfig()
vocabConfig.max_len_corpus = 512  # Otherwise the pre-trained model is unusable
assert vocabConfig.vocab_size <= 30522  # uses this many words


class BertBasedClassifier(GeneralModel):

    def __init__(self, num_labels):
        super(BertBasedClassifier, self).__init__()
        self.config = transformers.BertConfig(vocab_size_or_config_json_file=vocabConfig.vocab_size,
                                              num_hidden_layers=modelConfig.depth,
                                              num_attention_heads=modelConfig.n_head,
                                              max_position_embeddings=vocabConfig.max_len_corpus,
                                              num_labels=num_labels
                                              )
        self.error = nn.CrossEntropyLoss()
        self.model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', config=self.config)
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
