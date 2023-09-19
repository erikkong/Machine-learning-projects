import torch
import torchvision
import torchvision.transforms as T
from torch import optim
from torch.utils.data import ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



class data_loader():
    def __init__(self, data_set= "CIFAR10"):
        
        if data_set == "CIFAR10":
            batch_size = 1000
            aug = T.Compose([T.ToTensor()])
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=aug)
            
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=False)
            
            #self.test_set =  torchvision.datasets.CIFAR10(root='./data', train=False,
            #                           download=True, transform=transform)
            #self.validation_set = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size,
            #                             shuffle=False, num_workers=2)
            self.classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        else:
            raise Exception("No other data defined.")


def data_augmentations(images):
    
    
    transformations_1 = T.Compose([
        T.RandomResizedCrop(32, scale=(0.08, 1), interpolation=Image.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p =0.8),
        #T.RandomGrayscale(p=0.2),
        #T.GaussianBlur(1),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])
    
    transformations_2 = T.Compose([
        T.RandomResizedCrop(32, scale=(0.08, 1), interpolation=Image.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p =0.8),
        T.RandomGrayscale(p=0.2),
        #T.GaussianBlur(0.1),
        #T.Solarization(0.2),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])
        
    T1 = transformations_1(images)
    T2 = transformations_2(images)

    return T1, T2


def test_augmentations():
    dl = data_loader()
    tl_1 = dl.train_loader_1
    tl_2 = dl.train_loader_2
    
    for epoch in range(10):
        dataloader_iterator = iter(tl_1)
        for i, data1 in enumerate(tl_2):
        
            try:
                data2 = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(tl_1)
                data2 = next(dataloader_iterator)
            plt.imshow(np.array(data1)[0][0].T)
            plt.show()
            print(np.array(data1)[0].shape, flush=True)
            print(np.array(data1)[1].shape, flush=True)
            print(np.array(data2)[0].shape, flush=True)
            print(np.array(data2)[1].shape, flush=True)
        print("-----------------")