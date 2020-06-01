import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ewc import EWC
import numpy as np
from dataset import *

## Get full MNIST dataset
file_dict = {'train':['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'], 
	        'test':['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']}
path = 'data'
data = load_data(path, file_dict)

BATCH_SIZE = 128
EPOCH = 10
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

# Define model
class mlp(nn.Module):	
	def __init__(self):
		super(mlp, self).__init__()
		self.fc_1 = nn.Linear(784, 400)
		self.fc_2 = nn.Linear(400, 400)
		self.fc_3 = nn.Linear(400, 10)

	def forward(self, x):
		x = F.relu(self.fc_1(x))
		x = F.relu(self.fc_2(x))
		x = self.fc_3(x)
		return x

model = mlp()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Train on tasks sequentially
WEIGHT = 1e4
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
			old_task.append(trainloader.dataset[idx][0])
		ewc = EWC(model, old_task)
	# Train loop for task t
	print('Training on task %d ....'%(t+1))
	model.train()
	for epoch in range(EPOCH):
		running_loss = 0.
		for batch_idx, (data, target) in enumerate(trainloader):
			optimizer.zero_grad()
			output = model(data.float())
			output = output[:, active_classes]
			if t == 0:
				loss = criterion(output, target.long())
			else:
				loss = criterion(output, target.long()) + WEIGHT*ewc.penalty(model)
			running_loss += loss.item()
			loss.backward()
			model.fc_3.weight.grad[mask] = 0
			optimizer.step()
		print("Epoch: %d  -  Loss: %.4f"%(epoch+1, running_loss/batch_idx))
	print('Testing ....')
	model.eval()
	correct = 0
	for data, target in testloader:
		output = model(data.float())
		output = output[:, active_classes]
		pred = output.argmax(dim=1, keepdim=True)  
		correct += pred.eq(target.view_as(pred)).sum().item()
	print(100. * correct / len(testloader.dataset))
