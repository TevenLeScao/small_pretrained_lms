import torch
from torch import nn

from configuration import TrainConfig as tconfig
from utils import load_partial_state_dict

class RegressionModel(nn.Module):

    def forward(self, *args, **kwargs):
        raise("No forward allowed directly on wrapper RegressionModel")

    def __init__(self):
        super(RegressionModel, self).__init__()
        self.encoder = None
        self.decoder = None
        self.num_accumulations = 0
        self.accumulate = max(1, tconfig.accumulate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=tconfig.lr, weight_decay=tconfig.weight_decay)

    def step(self, loss, **kwargs):
        self.num_accumulations += 1
        nn.utils.clip_grad_norm(self.parameters(), tconfig.clip_grad)
        loss = loss / self.accumulate
        loss.backward()
        if self.num_accumulations % self.accumulate == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.num_accumulations = 0

    def initialize(self):

        for param in self.encoder.parameters():
            torch.nn.init.xavier_normal_(param)
        for param in self.decoder.parameters():
            torch.nn.init.xavier_normal_(param)

    def load_params(self, model_path, opt=True):
        dict_path = model_path + ".dict.pt"
        self.load_state_dict(torch.load(dict_path))
        if opt:
            opt_path = model_path + ".opt.pt"
            self.optimizer.load_state_dict(torch.load(opt_path))

    def save(self, path: str, opt=True):

        dict_path = path + ".dict.pt"
        torch.save(self.state_dict(), dict_path)
        if opt:
            opt_path = path + ".opt.pt"
            torch.save(self.optimizer.state_dict(), opt_path)
