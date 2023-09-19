import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import optim


class Loss(nn.Module):
    def __init__(self, temp_rows, temp_cols):
        super(Loss, self).__init__()
        self.temp_rows = temp_rows
        self.temp_cols = temp_cols
    
    def forward(self, output_1, output_2):
        N = len(output_1)
        C = len(output_1[0])
        
        log_y_x1 = torch.log((N/C)*F.normalize(F.softmax(output_1/self.temp_rows), p=1.0, dim=0))
        log_y_x2 = torch.log((N/C)*F.normalize(F.softmax(output_2/self.temp_rows), p=1.0, dim=0))
        
        y_x1 = F.normalize(F.softmax(output_1/self.temp_cols), p=1.0, dim=1)
        y_x2 = F.normalize(F.softmax(output_2/self.temp_cols), p=1.0, dim=1)

        l1 = -torch.sum(y_x2*log_y_x1)/N
        l2 = -torch.sum(y_x1*log_y_x2)/N
        
        loss = (l1 + l2)/2
        return loss