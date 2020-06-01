import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self):

    def forward(self, x):

class Generator(nn.Module):
    def __init__(self):

    def forward(self, x):

def train(model, crit, optim, dataset, embedding_sz, device, epoch):
    for i, data in enumerate(dataset):
        img, lbl = data
        img = img.to(device)
        z = torch.empty(img.shape[0], embedding_sz, device=device).normal_()

        # Real images
        optim['D'].zero_grad()
        score_real = Discriminator(img)
        l_real = optim(score_real, torch.ones(img.shape[0]))

        # Fake images
        G_z = Generator(z)
        score_fake = Discriminator(G_z.detach())
        l_fake = optim(score_fake, torch.zeros(img.shape[0]))

        l_disc = 0.5*(l_fake + l_real)
        l_disc.backward()
        optim['D'].step()

        # Generator Training
        optim['G'].zero_grad()
        score_fake = Discriminator(G_z)
        l_gen = optim(score_fake, torch.ones(img.shape[0]))
        l_gen.backward()
        optim['G'].step()

def test():
    print('Test not implemented')

def main(device):

    model = {'G': Generator(),
             'D': Discriminator{}}
    optim = {'G': optim.ADAM(model['G'].parameters(), lr=0.001),
             'D': optim.ADAM(model['D'].parameters(), lr=0.001)}
    crit  = nn.MSELoss()

    model['G'].to(device)
    model['D'].to(device)
    for epoch in range(epochs):
        train()
        test()

    model['G'].to('cpu')
    model['D'].to('cpu')
    torch.save({'G': model['G'].state_dict(),
                'D': model['D'].state_dict()}, 'GAN_checkpt')



if __name__ == '__name__':
    main(device='cuda')
