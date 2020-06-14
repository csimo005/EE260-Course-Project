import torch
import torchvision
import torchvision.transforms as tranforms
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from ewc_mhe import EWC
import numpy as np
from dataset import *
from gen_model import *
import matplotlib
from PIL import Image
import copy

from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR


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
            running_loss = 0.
    return
    
def test(model, testloader, save_prefix, learned, testing, device):
    model.eval()

    psnr = 0.
    ssim = 0.

    for i, (data, target) in enumerate(testloader):
        name = target.topk(1)[1]
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            reconstructed_x, _, _ = model(data, target)
        reconstructed_x = reconstructed_x.cpu().numpy()
        reconstructed_x = (reconstructed_x*255).astype(np.uint8)
        original_x = (data.cpu().numpy()*255).astype(np.uint8)

        for j in range(reconstructed_x.shape[0]):
            psnr += PSNR(original_x[j,0], reconstructed_x[j,0])
            ssim += SSIM(original_x[j,0], reconstructed_x[j,0])
            im = Image.fromarray(reconstructed_x[j,0])
            im = im.convert("L")
            im.save(save_prefix + "{}_{}.png".format(str(name[j].item()), str(i*data.shape[0]+j)))

    psnr = psnr/len(testloader)
    ssim = ssim/len(testloader)
    print('[%d, %d] PSNR: %.4f, SSIM %.4f' % (learned, testing, psnr, ssim))
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
        img[i] = samples[i][0]
        lbl[i, samples[i][1]] = 1
    return img, lbl

def VAE_loss(x, reconstructed_x, mean, log_var):
    RCL = F.binary_cross_entropy(reconstructed_x, x, reduction='mean')
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/x.shape[0]

    return RCL + KLD

def create_tasks(trainset, testset, batch_sz):
    train_idx = [np.zeros((0,), dtype=np.uint32) for i in range(10)]
    for i in range(len(trainset)):
        train_idx[trainset.targets[i]] = np.append(train_idx[trainset.targets[i]], [i])

    test_idx = [np.zeros((0,), dtype=np.uint32) for i in range(10)]
    for i in range(len(testset)):
        test_idx[testset.targets[i]] = np.append(test_idx[testset.targets[i]], [i])

    trainloader = [None]*2
    testloader = [None]*2
    for i in range(2):
        train_id = np.zeros((0,), dtype=np.uint32)
        test_id = np.zeros((0,), dtype=np.uint32)
        for j in range(5):
            train_id = np.concatenate((train_id, train_idx[i*5+j]))
            test_id = np.concatenate((test_id, test_idx[i*5+j]))
        print(train_id.shape)
        trainloader[i] = DataLoader(Subset(trainset, train_id), 
                                    batch_size=batch_sz, shuffle=True, num_workers=2, collate_fn=collate)
        testloader[i] = DataLoader(Subset(testset, test_id),
                                   batch_size=batch_sz, shuffle=True, num_workers=2, collate_fn=collate)

    return trainloader, testloader

def main(epochs, batch_sz,  lr, device, prefix):
    BATCH_SIZE = batch_sz
    EPOCH = epochs
    LR = lr
    
    INPUT_DIM = 28 * 28     # size of each input
    HIDDEN_DIM = 256        # hidden dimension
    LATENT_DIM = 75         # latent vector dimension
    N_CLASSES = 10          # number of classes in the data

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST('./', train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST('./', train=False, transform=transform, download=True)
    trainloader, testloader = create_tasks(trainset, testset, BATCH_SIZE)

    model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, N_CLASSES)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Train on tasks sequentially
    WEIGHT = 0.5
    ewc = EWC(model) 
    regularizer = lambda model: ewc(model, 2)
    old_task = []
    for t in range(2):
        # Train loop for task t
        print('Training on task %d ....'%(t+1))
        for epoch in range(EPOCH):
            train(model, optimizer, VAE_loss, trainloader[t], epoch, device, regularizer, WEIGHT)

        for i in range(t+1):
            output_path = prefix + 'task_%d/task_%d/' % (t+1,i+1)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            test(model, testloader[i], output_path, t, i, device)

        # Get EWC agent
        sample_idx = random.sample(range(len(testloader[t].dataset)), 200)
        old_task = []
        for idx in sample_idx:
           old_task.append(testloader[t].dataset[idx])
        ewc.update_FIM(old_task)
    
    if not os.path.exists(prefix + 'generated_images/'):
        os.makedirs(prefix + 'generated_images/')

    model.eval()
    for c in range(10):
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        y = torch.zeros(BATCH_SIZE, N_CLASSES, device=device)
        y[:,c]=1

        z = torch.cat((z, y), dim=1)
        with torch.no_grad():
            reconstructed_img = model.decoder(z)
        reconstructed_img = reconstructed_img.cpu().numpy()
        reconstructed_img = (reconstructed_img*255).astype(np.uint8)
        reconstructed_img = np.reshape(reconstructed_img, (-1,28,28))

        for j in range(reconstructed_img.shape[0]):
            im = Image.fromarray(reconstructed_img[j])
            im = im.convert("L")
            im.save(prefix + "generated_images/{}_{}.png".format(str(c), str(j)))
        
if __name__ == '__main__':
    main(25, 100, 1e-3, 'cuda:0', 'ewc_mhe_images/')
