import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torchlars import LARS




class Self_supervised_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resNet = resnet18()
        self.resNet.fc = nn.Identity()
        self.classifier = MLP_PH()
    
    def forward(self, x):
        x = self.resNet(x)
        x = self.classifier(x)
        
        return x
        
        
        
        
class MLP_PH(torch.nn.Module):
    def __init__(self, input_dim=512, output_dim=10):
        super().__init__()
        
        # Input is adapted to the ResNet-18 structure.
        input_size =input_dim
        hidden_size= [4096, 128]
        output_size = output_dim
     
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size[0]),
            nn.BatchNorm1d(hidden_size[0]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.BatchNorm1d(hidden_size[1]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size[1],output_size)
        )
        
    def forward(self, x):
        return F.normalize(self.layers(x), p=2, dim=1, eps=1e-7)
    

class MLP_CH(torch.nn.Module):
    def __init__(self, input_size, output_size = 1000):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size, bias = False)
        )
        



# --------------------------- Linear evaluation ---------------------------
def get_opt_and_lr_linear(total_steps, model, verbose=False):
    #iterations = np.arange(epochs*it_per_epoch)
    #learning_rates = end_value + ((start_value - end_value)/2)*(1 + np.cos(np.pi*(iterations)/len(iterations)))
    init_lr = 0.8
    final_lr = 0.0048
    
    
    base_optimizer = optim.SGD(model.parameters(), lr=init_lr)
    optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=total_steps, eta_min=final_lr)
    
    lrs = []
    for _ in range(total_steps):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    
    
    if verbose:
        plt.plot(range(len(lrs)), lrs)
        plt.show()
    
    
    return optimizer, lrs


# --------------------------- Unsupervised pre-training ---------------------------
def get_opt_and_lr_pre_train(total_steps, model, verbose=False):
    init_lr = 0.3
    max_lr = 4.8
    final_lr = 0.0048
    fin_div_fact =  init_lr/final_lr
    div_fact = max_lr/init_lr
    w_d = 10**(-6)
    base_optimizer = optim.SGD(model.parameters(), lr=init_lr, weight_decay= w_d)
    optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(base_optimizer, max_lr, total_steps=total_steps, pct_start=10/total_steps, anneal_strategy='cos', final_div_factor =fin_div_fact, div_factor = div_fact)

    lrs = []
    for _ in range(total_steps):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    
    if verbose:
        plt.plot(range(len(lrs)), lrs)
        plt.show()
    
    return optimizer, lrs
