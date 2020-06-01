import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ewc import EWC
import numpy as np
from dataset import *
from gen_model import *
import matplotlib
from PIL import Image
import copy

def train(model, optimizer, criterion, trainloader, epoch, device, regularizer=None, lmbda=0):
    model.train()
    print_it = 10
    running_loss = 0.
    for batch_idx, (data, target) in enumerate(trainloader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        reconstructed_x, z_mu, z_var = model(data, target)
        reconstructed_x = reconstructed_x.type(torch.DoubleTensor)
        data = data.type(torch.DoubleTensor)
        z_mu, z_var = z_mu.type(torch.DoubleTensor), z_var.type(torch.DoubleTensor)
        
        data, reconstructed_x, z_mu, z_var = data.cuda(), reconstructed_x.cuda(), z_mu.cuda(), z_var.cuda()
       
        loss = criterion(data, reconstructed_x, z_mu, z_var)
        if regularizer is not None:
            loss = loss + lmbda*regularizer(model)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx%print_it == print_it-1:
            print("Training [%d, %d] Loss: %.4f" % (epoch+1, (batch_idx+1)*data.shape[0], running_loss/print_it))
    return
    
def test(model, testloader, save_prefix, device):
    model.eval()
    for i, (data, target) in enumerate(testloader):
        name = target.topk(1)[1]
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            reconstructed_x, _, _ = model(data, target)
        reconstructed_x = reconstructed_x.view(-1, 28, 28).cpu().numpy()
        reconstructed_x = (reconstructed_x*255).astype(np.uint8)

        for j in range(reconstructed_x.shape[0]):
            im = Image.fromarray(reconstructed_x[j])
            im = im.convert("L")
            im.save(save_prefix + "{}_{}.png".format(str(name[j].item()), str(i*data.shape[0]+j)))
    return

def idx2onehot(y, N_CLASSES):
    onehot = torch.zeros(y.shape[0], N_CLASSES)
    #onehot[torch.arange(y.shape[0]), y] = 1
    for i in range(y.shape[0]):
        onehot[i,y[i]] = 1
    return onehot

def collate(samples):
    img = torch.zeros(len(samples), *samples[0][0].shape)
    lbl = torch.zeros(len(samples), 10)

    for i in range(len(samples)):
        img[i] = torch.tensor(samples[0][0], dtype=torch.float32)
        lbl[i, samples[0][1]] = 1
    return img, lbl

def VAE_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    RCL = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD


def main(epochs, batch_sz,  lr, device):
    ## Get full MNIST dataset
    file_dict = {'train':['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'], 
    'test':['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']}
    path = 'data'
    data = load_data(path, file_dict)
    
    BATCH_SIZE = batch_sz
    EPOCH = epochs
    LR = lr
    
    mnist_full_tr = mnist_dataset(data, train=True)
    mnist_full_te = mnist_dataset(data, train=False)
    
    tasks = {}
    for i in range(5):
        tasks[i+1] = [x for x in range(2*i,2*i+2)]
    
    active_cl = {}
    for i in range(5):
        active_cl[i+1] = []
        for j in range(i+1):
            active_cl[i+1].extend(tasks[j+1])
    
    task_masks = {}
    for i in range(5):
        task_masks[i+1] = torch.LongTensor([x for x in range(2*i+2,10)])
    
    trainloaders = {}
    for i in range(5):
        task_data = get_task_dataset(tasks[i+1], mnist_full_tr)
        trainloaders[i+1] = DataLoader(task_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    
    testloaders = {}
    for i in range(5):
        task_data = get_task_dataset(active_cl[i+1], mnist_full_te)
        testloaders[i+1] = DataLoader(task_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    
    
    
    INPUT_DIM = 28 * 28     # size of each input
    HIDDEN_DIM = 256        # hidden dimension
    LATENT_DIM = 75         # latent vector dimension
    N_CLASSES = 10          # number of classes in the data
    model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, N_CLASSES)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Train on tasks sequentially
    WEIGHT = 5
    ewc = None
    for t in range(5):
        trainloader = trainloaders[t+1]
        testloader = testloaders[t+1]
        active_classes = active_cl[t+1]
        mask = task_masks[t+1]

        # Train loop for task t
        print('Training on task %d ....'%(t+1))
        for epoch in range(EPOCH):
            train(model, optimizer, VAE_loss, trainloader, epoch, device, ewc, WEIGHT)
        if not os.path.exists('task_%d_recon/' % (t+1)):
            os.mkdir('task_%d_recon/' % (t+1))
        test(model, testloader, 'task_%d_recon/' % (t+1), device)

        # Get EWC agent
        sample_idx = random.sample(range(len(trainloader.dataset)), 200)
        old_task = []
        for idx in sample_idx:
            old_task.append(trainloader.dataset[idx])
        ewc = EWC(model, old_task)
    
    
    if not os.path.exists('generated_images/'):
        os.mkdir('generated_images/')

    model.eval()
    for c in range(10):
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        y = torch.zeros(BATCH_SIZE, N_CLASSES, device=device)
        y[:,c]=1

        z = torch.cat((z, y), dim=1)
        with torch.no_grad():
            reconstructed_img = model.decoder(z)
        reconstructed_img = reconstructed_img.view(-1,28,28).cpu().numpy()
        reconstructed_img = (reconstructed_img*255).astype(np.uint8)

        for j in range(reconstructed_img.shape[0]):
            im = Image.fromarray(reconstructed_img[j])
            im = im.convert("L")
            im.save("generated_images/{}_{}.png".format(str(c), str(j)))
        
if __name__ == '__main__':
    main(25, 100, 1e-3, 'cuda:0')
