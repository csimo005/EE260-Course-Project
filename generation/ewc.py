import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from copy import deepcopy
import numpy as np

INPUT_DIM = 28 * 28     # size of each input
HIDDEN_DIM = 256        # hidden dimension
LATENT_DIM = 75         # latent vector dimension
N_CLASSES = 10          # number of classes in the data


def calculate_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    RCL = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD
    
def idx2onehot(y, batch_size, n=N_CLASSES):
    #y = y.numpy()
    # One hot encoding buffer that you create out of the loop and just keep reusing
    onehot = np.zeros((batch_size, n))

    # In your for loop
    onehot[np.arange(np.size(y)),y] = 1

    return torch.Tensor(onehot)

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
        #loss_func = nn.MSELoss()
        #i=0
        for input_, target in self.dataset:

            self.model.zero_grad()
            #input(type(target))
            target = idx2onehot(target, 1, 10)
            #input(input_)
            input_ = torch.FloatTensor(input_).unsqueeze(0)
            input_ = input_.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            input_, target = input_.cuda(), target.cuda()
            reconstructed_x, z_mu, z_var = self.model(input_, target)
            #i += 1
            #print(i)
            #print(input_.is_cuda, reconstructed_x.is_cuda, z_mu.is_cuda, z_var.is_cuda)
            loss = calculate_loss(input_.cuda(), reconstructed_x, z_mu, z_var)
            loss = torch.log(loss)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

    def __call__(self, model: nn.Module):
        return self.penalty(model)
