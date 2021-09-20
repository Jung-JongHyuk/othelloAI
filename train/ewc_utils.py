from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        datasetSize = self.getDatasetSize()
        self.model.eval()
        for taskIdx in range(len(self.dataset)):
            for input in self.dataset[taskIdx]:
                self.model.zero_grad()
                input = variable(input).view(1, input.shape[0], input.shape[1])
                output = self.model(input, taskIdx)[0].view(1, -1)
                label = output.max(1)[1].view(-1)

                actionSize = output.shape[1]
                oneHotLabel = torch.zeros((1, actionSize))
                if torch.cuda.is_available():
                    oneHotLabel = oneHotLabel.cuda()
                oneHotLabel[0, label] = 1
                loss = self.loss_pi(output, oneHotLabel)
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.grad != None:
                        precision_matrices[n].data += p.grad.data ** 2 / datasetSize

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices
    
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]
    
    def getDatasetSize(self):
        count = 0
        for task in range(len(self.dataset)):
            count += len(self.dataset[task])
        return count

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss