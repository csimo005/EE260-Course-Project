import torch

def get_task_idx(cls, dataset):
	assert len(cls) > 1
    idx = []
    for i in range(len(dataset)):
        c = dataset[i][1].numpy()
        if c in cls:
            idx.append(i)
    return idx 

def get_task_dataset(cls, dataset):
    idx = get_task_idx(cls, dataset)
    return torch.utils.data.Subset(dataset, idx) 

### Get original dataset
dataset = 

### Define tasks in terms of classes
task = [1,2]

### Get task specific dataset
dataset_task = get_task_dataset(task, dataset)

d = {}
t = [[0,1],[2,3],[4,5]]
for i,task in enumerate(t):
	d{i} = get_task_dataset(task, dataset)

