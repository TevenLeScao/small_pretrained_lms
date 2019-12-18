import torch
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel
from sklearn.metrics import f1_score
from utils.helpers import get_optimizer

from configuration import GPU, TrainConfig as tconfig


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

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self, lr_decay):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= lr_decay

    def main_module(self):
        pass

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def __str__(self):
        s, n = self.get_network_description(self.main_module())
        if isinstance(self.main_module(), nn.DataParallel) or isinstance(self.main_module(), DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.main_module().__class__.__name__,
                                             self.main_module().module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.main_module().__class__.__name__)
        return 'Network structure: {}, with parameters: {:,d}\n{}'.format(net_struc_str, n, s)


def standard_mlp(params, inputdim, nclasses):
    dropout = 0. if "dropout" not in params else params["dropout"]
    if params["nhid"] == 0:
        mlp = nn.Sequential(
            nn.Linear(inputdim, nclasses),
        )
    else:
        mlp = nn.Sequential(
            nn.Linear(inputdim, params["nhid"]),
            nn.Dropout(p=dropout),
            nn.Sigmoid(),
            nn.Linear(params["nhid"], nclasses),
        )
    if GPU:
        return mlp.cuda()
    else:
        return mlp


class StandardMLP(GeneralModel):
    def __init__(self, params, inputdim, nclasses):
        super(StandardMLP, self).__init__()
        self.mlp = standard_mlp(params, inputdim, nclasses)
        self.nclasses = nclasses
        optim = "adam" if "optim" not in params else params["optim"]
        optim_fn, optim_params = get_optimizer(optim)
        self.optimizer = optim_fn(self.mlp.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = tconfig.weight_decay
        if GPU:
            self.loss_fn = nn.CrossEntropyLoss().cuda()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn.size_average = False

    def forward(self, inp):
        return self.mlp(inp)

    def predictions_to_loss(self, predictions, target):
        return self.loss_fn(predictions, target)

    def predictions_to_acc(self, predictions, target):
        predictions = predictions.max(dim=1)[1]
        return (predictions == target).sum().double() / target.shape[0]

    def emocontext_f1(self, predictions, target, excluded_classes=(0,)):
        predictions = predictions.max(dim=1)[1]
        included_classes = set(range(self.nclasses)) - set(excluded_classes)
        true_pos = sum(((predictions == in_class) * (target == in_class)).sum().item() for in_class in included_classes)
        total_preds = sum((predictions == in_class).sum().item() for in_class in included_classes)
        total_targets = sum((target == in_class).sum().item() for in_class in included_classes)
        try:
            precision = true_pos / total_preds
            recall = true_pos / total_targets
            return float(2 * precision * recall / (precision + recall))
        except ZeroDivisionError:
            return 0

    def main_module(self):
        return self.mlp

class SimpleDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)