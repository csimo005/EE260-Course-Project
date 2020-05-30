import torch
import torch.nn as nn

def euclidean_dist(w_i, w_j):
    return torch.norm(w_i-w_j, 2)

def angular_dist(w_i, w_j):
    return torch.acos(torch.sum(w_i*w_j))

def MHE(weights, s, dist_func='euclidean'):
    if dist_func == 'euclidean':
        dist = lambda w_i, w_j: euclidean_dist(w_i, w_j)
    elif dist_func == 'angular':
        dist = lambda w_i, w_j: angular_dist(w_i, w_j)
    else:
        raise NotImplementedError

    if s > 0:
        fs = lambda x: torch.pow(x, -s)
    else:
        fs = lambda x: torch.log(torch.pow(x, -1))

    energy = torch.zeros(1)
    for i in range(weights.shape[0]):
        for j in range(i, weights.shape[0]): #Assumes that what ever distance metric is symmetric
            energy += fs(dist(weights[i:i+1], weights[j:j+1]))

    energy = (2/(weights.shape[0]*(weights.shape[0]-1)))*energy
    return energy
    

def MHE_halfspace(weights, s):
    weights = torch.cat((weights, -weights), 0)
    return MHW(weights, s)

def MHE_loss(model, s, lh, lo):
    hidden_loss = torch.zeros(1)
    output_loss = torch.zeros(1)

    for name, param in state_dict.named_children():
        if type(param) is nn.Linear:
            weights = torch.cat((param._parameters['weight'], param._parameters['bias'].view(-1,1)), 0)
        elif type(param) is nn.Conv2d:
            weights = torch.cat((param._parameters['weight'].view(param.out_channels,-1), param._parameters['bias'].view(-1,1)), 0)
        else:
            raise NotImplementedError

        if name == '_output':
            output_loss = MHE(weights, s)
        else:
            hidden_loss += MHE_halfspace(weights, s)

    return lh*hidden_loss + lo*output_loss
