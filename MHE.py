import torch
import torch.nn as nn

def angular_dist(w_i, w_j):
    return torch.acos(torch.sum(w_i*w_j))

def MHE(weights, s, dist='euclidean', device='cpu'):
    if dist == 'euclidean':
        dist = lambda W, w_i: torch.norm(W-w_i, 2, dim=1)
    elif dist == 'angular':
        dist = lambda w_i, w_j: angular_dist(w_i, w_j)
    else:
        raise NotImplementedError

    if s > 0:
        fs = lambda x: torch.pow(x, -s)
    else:
        fs = lambda x: torch.log(torch.pow(x, -1))

    energy = torch.zeros(1, device=device, requires_grad=True)
    for i in range(weights.shape[0]-1):
        energy = energy +torch.sum(fs(dist(weights[i+1:], weights[i])))

    energy = (2/(weights.shape[0]*(weights.shape[0]-1)))*energy
    return energy
    

def MHE_halfspace(weights, s, dist='euclidean', device='cpu'):
    weights = torch.cat((weights, -weights), 0)
    return MHE(weights, s, dist=dist, device=device)

def MHE_loss(model, s, lh, lo, dist='euclidean', device='cpu'):
    hidden_loss = torch.zeros(1, device=device, requires_grad=True)
    output_loss = torch.zeros(1, device=device, requires_grad=True)

    for name, param in model.named_children():
        if type(param) is nn.Linear:
            weights = torch.cat((param._parameters['weight'], param._parameters['bias'].view(-1,1)), 1)
        elif type(param) is nn.Conv2d:
            weights = torch.cat((param._parameters['weight'].view(param.out_channels,-1), param._parameters['bias'].view(-1,1)), 0)
        else:
            raise NotImplementedError

        weights = weights/torch.norm(weights,2,dim=1).view(-1,1)

        if name == '_output':
            output_loss = MHE(weights, s, dist=dist, device=device)
        else:
            hidden_loss = hidden_loss + MHE_halfspace(weights, s, dist=dist, device=device)
    return lh*hidden_loss + lo*output_loss
