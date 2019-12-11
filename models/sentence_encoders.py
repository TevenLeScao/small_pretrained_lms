from models.structure import *
from configuration import ModelConfig as mconfig


class SentenceEncoder(GeneralModel):

    def __init__(self, word_dim, sentence_dim):
        super(SentenceEncoder, self).__init__()
        self.word_dim = word_dim
        self.sentence_dim = sentence_dim
        self.projection = None

    def forward(self, embedded_words: torch.Tensor, sent_mask: torch.Tensor) -> torch.Tensor:
        pass

    def main_module(self):
        return self.projection


class BOREP(SentenceEncoder):
    # after Wieting and Kiela

    def __init__(self, word_dim=mconfig.width, sentence_dim=mconfig.sentence_width):
        super(BOREP, self).__init__(word_dim, sentence_dim)
        self.projection = nn.Linear(word_dim, sentence_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=tconfig.lr, weight_decay=tconfig.weight_decay)

    def forward(self, embedded_words: torch.Tensor, sent_mask=None) -> torch.Tensor:
        projected = self.projection(embedded_words)
        return projected.max(dim=1, keepdim=False)[0]
        # return projected.mean(dim=1, keepdim=False)

    def redraw(self):
        self.projection = nn.Linear(self.word_dim, self.sentence_dim)


class RandomLSTM(SentenceEncoder):
    # after Wieting and Kiela

    def __init__(self, word_dim=mconfig.width, sentence_dim=mconfig.sentence_width):
        super(RandomLSTM, self).__init__(word_dim, sentence_dim)
        self.projection = nn.LSTM(word_dim, sentence_dim // 2, bidirectional=True, batch_first=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=tconfig.lr, weight_decay=tconfig.weight_decay)

    def forward(self, embedded_words: torch.Tensor, sent_mask: torch.Tensor) -> torch.Tensor:
        lengths = (-sent_mask + 1).sum(dim=1, keepdim=False)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(embedded_words, lengths,
                                                            batch_first=True, enforce_sorted=False)
        projected_sequence, (last_h, last_c) = self.projection(packed_sequence)
        unpacked_sequence, lengths = nn.utils.rnn.pad_packed_sequence(projected_sequence, batch_first=True)
        sent_mask = sent_mask.unsqueeze(2).expand(*sent_mask.shape, self.sentence_dim)
        unpacked_sequence = unpacked_sequence - sent_mask * 1e9
        return unpacked_sequence.max(dim=1, keepdim=False)[0]

    def redraw(self):
        self.projection = nn.LSTM(self.word_dim, self.sentence_dim // 2, bidirectional=True,  batch_first=True)

# TODO: echo-state networks