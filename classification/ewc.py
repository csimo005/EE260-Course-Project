import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from copy import deepcopy

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            output = self.model(torch.from_numpy(input).float()).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices
    
    def angular_dist(self, w_i, w_j):
        return torch.acos(torch.sum(w_i*w_j))

    def MHE(self, weights, s, dist='euclidean'):
        if dist == 'euclidean':
            dist = lambda W, w_i: torch.norm(W-w_i, 2, dim=1)
        elif dist == 'angular':
            dist = lambda w_i, w_j: self.angular_dist(w_i, w_j)
        else:
            raise NotImplementedError

        if s > 0:
            fs = lambda x: torch.pow(x+1e-5, -s)
        else:
            fs = lambda x: torch.log(torch.pow(x, -1))

        energy = torch.zeros(1, requires_grad=True)
        for i in range(weights.shape[0]-1):
            energy = energy +torch.sum(fs(dist(weights[i+1:], weights[i])))

        energy = (2/(weights.shape[0]*(weights.shape[0]-1)))*energy
        return energy
        

    def MHE_halfspace(self, weights, s, dist='euclidean'):
        weights = torch.cat((weights, -weights), 0)
        return self.MHE(weights, s, dist=dist)

    def penalty_1(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

    def penalty_2(self, model: nn.Module, s, dist='euclidean'):
        hidden_loss = torch.zeros(1, requires_grad=True)
        output_loss = torch.zeros(1, requires_grad=True)
        for n, p in model.named_parameters():
            weights = p
            weights = weights/torch.norm(weights,2,dim=1).view(-1,1)

            ## hard scaling code
            # median = torch.median(self._precision_matrices[n])
            # idx = self._precision_matrices[n] > median
            # idx = idx.int()
            # idx = 10*idx
            # idx[idx==0] = 1
            # mask = self._precision_matrices[n] * idx
            # weights = mask * weights

            weights 10*self._precision_matrices[n]*weights
            if n == 'fc3.weight':
                output_loss = self.MHE(weights, s, dist=dist)
            else:
                hidden_loss = hidden_loss + self.MHE_halfspace(weights, s, dist=dist)
        return hidden_loss + output_loss