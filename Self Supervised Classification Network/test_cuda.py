import torch
import numpy as np
#print(torch.version.cuda)
#print(torch.__version__)
from torch.utils.data import DataLoader

list_1 = [1,2,3,4,5,6,7,8,9,10]
list_2 = [10,11,12,13,14,15,16,17,18,19]
dataset_list = np.array([list_1, list_2])

dataset_loader = DataLoader(dataset_list.T, shuffle=False, batch_size=5)
for i in range(10):
    for i, data in enumerate(dataset_loader, 0):
        print(data)
    print("---")