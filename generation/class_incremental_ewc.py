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
## Get full MNIST dataset
file_dict = {'train':['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'], 
	        'test':['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']}
path = 'data'
data = load_data(path, file_dict)

BATCH_SIZE = 1
EPOCH = 1
LR = 1e-3

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
	trainloaders[i+1] = DataLoader(task_data, batch_size=BATCH_SIZE, shuffle=True)

testloaders = {}
for i in range(5):
	task_data = get_task_dataset(active_cl[i+1], mnist_full_te)
	testloaders[i+1] = DataLoader(task_data, batch_size=BATCH_SIZE, shuffle=False)

def calculate_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    RCL = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD

INPUT_DIM = 28 * 28     # size of each input
HIDDEN_DIM = 256        # hidden dimension
LATENT_DIM = 75         # latent vector dimension
N_CLASSES = 10          # number of classes in the data    
model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, N_CLASSES)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)

def idx2onehot(y, batch_size, n=N_CLASSES):
    y = y.cpu().numpy()
    # One hot encoding buffer that you create out of the loop and just keep reusing
    onehot = np.zeros((batch_size, n))

    # In your for loop
    onehot[np.arange(np.size(y)),y] = 1

    return torch.FloatTensor(onehot)

# Train on tasks sequentially
WEIGHT = 5
for t in range(5):
	trainloader = trainloaders[t+1]
	testloader = testloaders[t+1]
	active_classes = active_cl[t+1]
	mask = task_masks[t+1]
	# Get EWC agent
	if t > 0:
		sample_idx = random.sample(range(len(trainloader.dataset)), 200)
		old_task = []
		for idx in sample_idx:
			old_task.append(trainloader.dataset[idx])
		ewc = EWC(model, old_task)
	# Train loop for task t
	print('Training on task %d ....'%(t+1))
	model.train()
	for epoch in range(EPOCH):
		running_loss = 0.
		for batch_idx, (data, target) in enumerate(trainloader):
		    data, target = data.cuda(), target.cuda()   
		    target = idx2onehot(target.view(-1, 1), BATCH_SIZE, 10)
		    #input(target)
		    #print(data)
		    target = target.type(torch.FloatTensor).cuda()
		    data = data.type(torch.FloatTensor).cuda()
		    optimizer.zero_grad()
		    #print('main loop...', data.shape, target.shape)
		    reconstructed_x, z_mu, z_var = model(data, target)
		    reconstructed_x = reconstructed_x.type(torch.DoubleTensor)
		    data = data.type(torch.DoubleTensor)
		    z_mu, z_var = z_mu.type(torch.DoubleTensor), z_var.type(torch.DoubleTensor)
		    
		    data, reconstructed_x, z_mu, z_var = data.cuda(), reconstructed_x.cuda(), z_mu.cuda(), z_var.cuda()
		    
		    if t == 0:
		        loss = calculate_loss(data, reconstructed_x, z_mu, z_var)
		    else:
		        loss = calculate_loss(data, reconstructed_x, z_mu, z_var) + WEIGHT*ewc.penalty(model)
		        #print(loss.is_cuda)
		    running_loss += loss.item()
		    loss.backward()
		    optimizer.step()
		print("Epoch: %d  -  Loss: %.4f"%(epoch+1, running_loss/batch_idx))
	print('Testing ....')
	model.eval()
	for i, (data, target) in enumerate(testloader):
	    name = target
	    data, target = data.cuda(), target.cuda()
	    target = idx2onehot(target.view(-1, 1), BATCH_SIZE, 10)
	    target = target.type(torch.FloatTensor).cuda()
	    data = data.type(torch.FloatTensor).cuda()
	    reconstructed_x, _, _ = model(data, target)
	    reconstructed_x = reconstructed_x.view(28, 28).cpu().data
	    reconstructed_x = (reconstructed_x.cpu().detach().numpy().astype(np.uint8))
	    im = Image.fromarray(reconstructed_x)
	    im = im.convert("L")
	    im.save("{}_{}.png".format(str(name), str(i)))
	    

for l in range(1000):
    z = torch.randn(1, LATENT_DIM).cuda()
    
    y = torch.randint(0, N_CLASSES, (1, 1)).to(dtype=torch.long)
    real_label = copy.deepcopy(y)
    y = idx2onehot(y, 1, 10).cuda().type(z.dtype)
    z = torch.cat((z, y), dim=1)
    reconstructed_img = model.decoder(z)
    img = reconstructed_img.view(28, 28).cpu().data
    matplotlib.image.imsave('new_{}_{}.png'.format(str(label), str(l)), array)
    #plt.figure()
    #plt.imshow(img, cmap='gray')
    #plt.show()
		
		
