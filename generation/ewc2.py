import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from copy import deepcopy

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
        for name, param in self.model.named_children():
            if type(param) is nn.Linear:
                weights = torch.cat((param._parameters['weight'], param._parameters['bias'].view(-1,1)), 1)
            elif type(param) is nn.Conv2d:
                weights = torch.cat((param._parameters['weight'].view(param.out_channels,-1), param._parameters['bias'].view(-1,1)), 0)
            else:
                raise NotImplementedError(type(param))
            precision_matrices[name] = torch.zeros(weights.shape)

        self.model.eval()
        #loss_func = nn.MSELoss()
        #i=0
        for input_, target in self.dataset:

            self.model.zero_grad()
            target = idx2onehot(target, 1, 10) 
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

            for name, param in self.model.named_children():
                if type(param) is nn.Linear:
                    weights = torch.cat((param._parameters['weight'], param._parameters['bias'].view(-1,1)), 1)
                elif type(param) is nn.Conv2d:
                    weights = torch.cat((param._parameters['weight'].view(param.out_channels,-1), param._parameters['bias'].view(-1,1)), 0)
                else:
                    raise NotImplementedError
        
                precision_matrices[name].data += param.grad.data ** 2 / len(self.dataset)

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
        for name, param in model.named_children():
            if type(param) is nn.Linear:
                weights = torch.cat((param._parameters['weight'], param._parameters['bias'].view(-1,1)), 1)
            elif type(param) is nn.Conv2d:
                weights = torch.cat((param._parameters['weight'].view(param.out_channels,-1), param._parameters['bias'].view(-1,1)), 0)
            else:
                raise NotImplementedError

            weights = weights/torch.norm(weights,2,dim=1).view(-1,1)

            weights = 10*self._precision_matrices[n]*weights
            hidden_loss = hidden_loss + self.MHE_halfspace(weights, s, dist=dist)
        return hidden_loss

    def __call__(self, model: nn.Module, s, dist='euclidean'):
        return self.penalty_2(model, s, dist=dist)
