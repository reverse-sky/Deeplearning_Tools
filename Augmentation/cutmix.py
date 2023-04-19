import torch
import numpy as np
import torch.nn as nn

class CutMixCriterion:
    def __init__(self, criterion = nn.CrossEntropyLoss(),device='cpu'):
        self.criterion = criterion.to(device)

    def __call__(self, preds, targets):   # __call__는 instance가 실행되었을 때 진행하는 함수 
        
        return self.criterion(preds,targets)
    

def cutmix(data,targets,alpha = 1.0):
    cut_data    = data.clone()  # if data is numpy, use copy()
    cut_targets = targets.clone()
    ##### cutmix start
    indices     = torch.randperm(data.size(0))           # shuffle index  
    shuffled_data    = cut_data[indices]                     # shuffle dataset 
    shuffled_targets = cut_targets[indices]                     # shuffle dataset  
    lambda_prob = np.random.beta(alpha, alpha)       # Chose beta distribution value

    ########### cutmix start 
    image_h, image_w = data.shape[2:]                 
    r_x = np.random.uniform(0, image_w) 
    r_y = np.random.uniform(0, image_h)
    r_w = image_w * np.sqrt(1 - lambda_prob)
    r_h = image_h * np.sqrt(1 - lambda_prob)
    x1  =  int(np.clip((r_x - r_w)/2, a_min=0,a_max=image_w))
    x2  =  int(np.clip((r_x + r_w)/2, a_min=0,a_max=image_w))
    y1  =  int(np.clip((r_y - r_h)/2, a_min=0,a_max=image_h))
    y2  =  int(np.clip((r_y + r_h)/2, a_min=0,a_max=image_h))
    cut_data[:,:,x1:x2,y1:y2]    = shuffled_data[:,:,x1:x2,y1:y2]    # cutmix data
    ########### cut mix end
    
    lamb = 1-(x2-x1)*(y2-y1)/(image_w*image_h)   # calculate label_ratio 
    cut_targets = lamb*targets + (1-lamb)*cut_targets  # calculate  
    return cut_data, cut_targets

def segmentation_cutmix(data,mask,alpha = 1.0):
    cut_data    = data.clone()  # if data is numpy, use copy()
    cut_mask    = mask.clone()
    ##### cutmix start
    indices     = torch.randperm(data.size(0))           # shuffle index  
    shuffled_data       = cut_data[indices]                     # shuffle dataset 
    shuffled_targets    = cut_mask[indices]                     # shuffle dataset  
    lambda_prob = np.random.beta(alpha, alpha)       # Chose beta distribution value

    ########### cutmix start 
    image_h, image_w = data.shape[2:]                 
    r_x = np.random.uniform(0, image_w) 
    r_y = np.random.uniform(0, image_h)
    r_w = image_w * np.sqrt(1 - lambda_prob)
    r_h = image_h * np.sqrt(1 - lambda_prob)
    x1  =  int(np.clip((r_x - r_w)/2, a_min=0,a_max=image_w))
    x2  =  int(np.clip((r_x + r_w)/2, a_min=0,a_max=image_w))
    y1  =  int(np.clip((r_y - r_h)/2, a_min=0,a_max=image_h))
    y2  =  int(np.clip((r_y + r_h)/2, a_min=0,a_max=image_h))
    cut_data[:,:,x1:x2,y1:y2] = shuffled_data[:,:,x1:x2,y1:y2]    # cutmix data
    cut_mask[:,:,x1:x2,y1:y2] = shuffled_targets[:,:,x1:x2,y1:y2] # cutmix mask 
    ########### cut mix end

    lamb = 1-(x2-x1)*(y2-y1)/(image_w*image_h)
    # aug_targets = lamb*mask + (1-lamb)*cut_mask  # not doing in segmentation task 
    return cut_data, cut_mask

