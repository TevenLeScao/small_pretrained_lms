from typing import List

import torch
from torch import nn

from utils.helpers import get_optimizer

from configuration import TrainConfig as tconfig, VocabConfig as vconfig


class GeneralModel(nn.Module):

    def __init__(self):
        super(GeneralModel, self).__init__()
        self.optimizer = None
        self.num_accumulations = 0
        # accumulate must be > 0
        self.accumulate = max(1, tconfig.accumulate)

    def forward(self, *args, **kwargs):
        raise ("No forward allowed directly on wrapper GeneralModel")

    def step(self, **kwargs):
        self.num_accumulations += 1
        nn.utils.clip_grad_norm(self.parameters(), tconfig.clip_grad)
        # so that it's the mean of the gradients of the different batches not the sum
        if self.num_accumulations % self.accumulate == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.num_accumulations = 0

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


class WordEmbedder(GeneralModel):

    def __init__(self):
        super(WordEmbedder, self).__init__()
        self.vocab_size = vconfig.subwords_vocab_size if vconfig else vconfig.vocab_size

    def forward(self, one_hot_sentences: torch.Tensor, sent_mask: torch.Tensor) -> torch.Tensor:
        pass


def standard_mlp(params, inputdim, nclasses):
    dropout = 0. if "dropout" not in params else params["dropout"]
    if params["nhid"] == 0:
        return nn.Sequential(
            nn.Linear(inputdim, nclasses),
        ).cuda()
    else:
        return nn.Sequential(
            nn.Linear(inputdim, params["nhid"]),
            nn.Dropout(p=dropout),
            nn.Sigmoid(),
            nn.Linear(params["nhid"], nclasses),
        ).cuda()


class StandardMLP(GeneralModel):
    def __init__(self, params, inputdim, nclasses):
        super(StandardMLP, self).__init__()
        self.mlp = standard_mlp(params, inputdim, nclasses)
        optim = "adam" if "optim" not in params else params["optim"]
        optim_fn, optim_params = get_optimizer(optim)
        self.optimizer = optim_fn(self.mlp.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = tconfig.weight_decay
        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

    def forward(self, inp):
        return self.mlp(inp)

    def decode_to_loss(self, inp, target):
        return self.loss_fn(self(inp), target)
