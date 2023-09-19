import torch
from models import *
from data import *
from loss import *
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
#from torchlars import LARS


def pre_train(epochs, model, train_loader, opt, lrs):
    criterion = Loss(temp_rows=0.1, temp_cols=0.05)
    model.train()
    saved_losses = []
    for epoch in range(epochs):
        loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            images, _ = data
            
            T1_aug_images, T2_aug_images = data_augmentations(images)
            
            opt.zero_grad()
            
            # Forward-pass:
            output_1 = model(T1_aug_images)
            output_2 = model(T2_aug_images)
            
            # Compute loss:
            loss = criterion(output_1, output_2)
            
            # Backwards pass:
            loss.backward()
            
            # Update optimizer:
            opt.step()
            for g in opt.param_groups:
                g['lr'] = lrs[epoch*len(train_loader)+ i]
            
            # Saving loss:
            loss = loss.item()
            
            
            print(f"Iter: {i} - Epoch {epoch} - Loss: {loss} - Current lr: {lrs[epoch*len(train_loader)+ i]}")
            saved_losses.append(loss)
            loss = 0
    plt.plot(range(len(saved_losses)), saved_losses) 
    plt.show()           
    # Saved the trained model
    torch.save(model.state_dict(), "./Saved_models/")    
    return model
    
    
def get_data():
    dl = data_loader()
    return dl.train_loader, dl.classes

def main():
    # -------------------------- Load data --------------------------
    train_loader, classes = get_data()
    
    # -------------------------- Load models -------------------------- 
    ss_model = Self_supervised_model()
    
    # -------------------------- Load optimizer --------------------------
    optimizer_pre_train, lrs_pre_train = get_opt_and_lr_pre_train(800, ss_model)
    #optimizer_pre_linear, scheduler_linear = get_opt_and_lr_linear(100, mlp_ph)
    
    # -------------------------- Pre-training --------------------------
    pre_train(5, ss_model, train_loader, optimizer_pre_train, lrs_pre_train)
    

if __name__ == "__main__":
    main()
    
    
    
    