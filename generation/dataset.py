import torch
import gzip
import os
import random
import numpy as np
from torch.utils.data import Dataset, Subset

class mnist_dataset(Dataset):
	def __init__(self, data, train):
		if train:
			self.X = data['train'][0]
			self.Y = data['train'][1]
		else:
			self.X = data['test'][0]
			self.Y = data['test'][1]

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		x = self.X[idx]
		y = self.Y[idx]
		return x, y

def load_data(path, file_dict):
	dataset = {}
	img_w = img_h = 28
	num_cl = 10
	num_tr = 50000
	num_te = 10000

	img_tr = np.zeros([num_tr, img_h*img_w])
	lab_tr = np.zeros(num_tr, dtype=np.uint8)
	img_te = np.zeros([num_te, img_h*img_w])
	lab_te = np.zeros(num_te, dtype=np.uint8)

	for data in ['train', 'test']:

		path_img = os.path.join(path, file_dict[data][0])
		path_lab = os.path.join(path, file_dict[data][1])

		file_img = gzip.open(path_img)
		file_lab = gzip.open(path_lab)
		file_img.read(16)
		file_lab.read(8)

		if data == 'train':
			for i in range(num_tr):
				buf_img = file_img.read(img_h*img_w)
				buf_lab = file_lab.read(1)
				img_tr[i] = np.frombuffer(buf_img, dtype=np.uint8).astype(np.float32)/255.
				lab_tr[i] = np.frombuffer(buf_lab, dtype=np.uint8)
			dataset[data] = [img_tr, lab_tr]
		else:
			for i in range(num_te):
				buf_img = file_img.read(img_h*img_w)
				buf_lab = file_lab.read(1)
				img_te[i] = np.frombuffer(buf_img, dtype=np.uint8).astype(np.float32)/255.
				lab_te[i] = np.frombuffer(buf_lab, dtype=np.uint8)
			dataset[data] = [img_te, lab_te]

	return dataset

def get_task_idx(cls, dataset):
	#assert len(cls) > 1
	idx = []
	for i in range(len(dataset)):
		c = dataset[i][1]
		if c in cls:
			idx.append(i)
	return idx 

def get_task_dataset(cls, dataset):
    idx = get_task_idx(cls, dataset)
    return Subset(dataset, idx)
