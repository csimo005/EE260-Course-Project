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
    def __init__(self, model: nn.Module, dataset=None):

        self.model = model
        self.dataset = dataset

        if dataset is not None:
            self._precision_matrices = self._diag_fisher()
        else:
            self._precision_matrices = None

    def update_FIM(self, dataset: list):
        self.dataset = dataset
        self._precision_matrices = self._diag_fisher()

    def _diag_fisher(self):
        precision_matrices = {}
        for name, module in self.model.named_modules():
            if len(list(module.modules())) > 1:
                continue
            if type(module) is nn.Linear:
                if hasattr(module, 'bias'):
                    weights = torch.cat((module._parameters['weight'], module._parameters['bias'].view(-1,1)), 1)
                else:
                    weights = module._parameters['weight']
            elif type(module) is nn.Conv2d:
                if hasattr(module, 'bias'):
                    weights = torch.cat((module._parameters['weight'].view(module.out_channels,-1), module._parameters['bias'].view(-1,1)), 0)
                else:
                    weights = module._parameters['weight']
            else:
                raise NotImplementedError(type(param))
            precision_matrices[name] = None 

        self.model.eval()
        for input_, target in self.dataset:
            self.model.zero_grad()
            target = idx2onehot(target, 1, 10) 
            input_ = torch.FloatTensor(input_).unsqueeze(0)
            input_ = input_.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            input_, target = input_.cuda(), target.cuda()
            reconstructed_x, z_mu, z_var = self.model(input_, target)
            
            loss = calculate_loss(input_.cuda(), reconstructed_x, z_mu, z_var)
            loss = torch.log(loss)
            loss.backward()


            for name, module in self.model.named_modules():
                if len(list(module.modules())) > 1:
                    continue
                if type(module) is nn.Linear:
                    if hasattr(module, 'bias'):
                        grad = torch.cat((module._parameters['weight'].grad, module._parameters['bias'].grad.view(-1,1)), 1)
                    else:
                        grad = module._parameters['weight'].grad
                elif type(module) is nn.Conv2d:
                    if hasattr(module, 'bias'):
                        grad = torch.cat((module._parameters['weight'].grad.view(module.out_channels,-1), module._parameters['bias'].grad.view(-1,1)), 0)
                    else:
                        grad = module._parameters['weight'].grad.view(module.out_channels,-1)
                else:
                    raise NotImplementedError(type(param))
        
                if precision_matrices[name] is None:
                    precision_matrices[name] = grad.data ** 2 / len(self.dataset)
                else:
                    precision_matrices[name] += grad.data ** 2 / len(self.dataset)
        precision_matrices = {n:p.detach() for n,p in precision_matrices.items()}
        
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

        energy = None 
        for i in range(weights.shape[0]-1):
            if energy is None:
                energy = torch.sum(fs(dist(weights[i+1:], weights[i])))
            else:
                energy = energy + torch.sum(fs(dist(weights[i+1:], weights[i])))

        energy = (2/(weights.shape[0]*(weights.shape[0]-1)))*energy
        return energy
        

    def MHE_halfspace(self, weights, s, dist='euclidean'):
        weights = torch.cat((weights, -weights), 0)
        return self.MHE(weights, s, dist=dist)

    def mhe_only(self, model: nn.Module, s, dist='euclidean'):
        hidden_loss = None
        for name, module in self.model.named_modules():
            if len(list(module.modules())) > 1:
                continue
            if type(module) is nn.Linear:
                if hasattr(module, 'bias'):
                    weights = torch.cat((module._parameters['weight'], module._parameters['bias'].view(-1,1)), 1)
                else:
                    weights = module._parameters['weight']
            elif type(module) is nn.Conv2d:
                if hasattr(module, 'bias'):
                    weights = torch.cat((module._parameters['weight'].view(module.out_channels,-1), module._parameters['bias'].view(-1,1)), 0)
                else:
                    weights = module._parameters['weight']
            else:
                raise NotImplementedError

            weights = weights/torch.norm(weights,2,dim=1).view(-1,1)
            if hidden_loss is None:
                hidden_loss = self.MHE_halfspace(weights, s, dist=dist)
            else:
                hidden_loss = hidden_loss + self.MHE_halfspace(weights, s, dist=dist)
        return hidden_loss

    def mhe_ewc(self, model: nn.Module, s, dist='euclidean'):
        hidden_loss = None 
        for name, module in self.model.named_modules():
            if len(list(module.modules())) > 1:
                continue
            if type(module) is nn.Linear:
                if hasattr(module, 'bias'):
                    weights = torch.cat((module._parameters['weight'], module._parameters['bias'].view(-1,1)), 1)
                else:
                    weights = module._parameters['weight']
            elif type(module) is nn.Conv2d:
                if hasattr(module, 'bias'):
                    weights = torch.cat((module._parameters['weight'].view(module.out_channels,-1), module._parameters['bias'].view(-1,1)), 0)
                else:
                    weights = module._parameters['weight']
            else:
                raise NotImplementedError

            weights = weights/torch.norm(weights,2,dim=1).view(-1,1)
            weights = 10*self._precision_matrices[name]*weights

            if hidden_loss is None:
                hidden_loss = self.MHE_halfspace(weights, s, dist=dist)
            else:
                hidden_loss = hidden_loss + self.MHE_halfspace(weights, s, dist=dist)
        return hidden_loss

    def __call__(self, model: nn.Module, s, dist='euclidean'):
        if self._precision_matrices is not None:
            return self.mhe_ewc(model, s, dist=dist)
        else:
            return self.mhe_only(model, s, dist=dist)
