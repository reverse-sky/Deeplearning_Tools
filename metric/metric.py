import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm.auto import tqdm
from thop import profile
from time import localtime
import time
import os

def confusion_matrix(pred, target, class_name):
    
    cm_pred = np.zeros((len(class_name), len(class_name)))
    cm_target = np.zeros((len(class_name), len(class_name)))
    
    for i in range(len(class_name)):
        mask = target == i # 각 label마다 masking을 진행 
        pred_mask = pred[mask] # pred에서 mask에 해당하는 mask만 pred_mask로 가지고 옴 
        for j in range(len(class_name)):
            cm_pred[i, j] = np.sum(pred_mask==j) #
            cm_target[i, j] = np.sum(mask)
            
    return cm_pred, cm_target

def draw_matrix(pred,labels,df,class_name =[i for i in range(8)],save=False,path ="./"):
    pred,labels = confusion_matrix(pred,labels,class_name)
    f,ax = plt.subplots(1,2,figsize=(13,7))
    cm = np.round(pred/labels, 2)
    ax[1] = sns.heatmap(cm,annot=True,
                     annot_kws={
                    'fontsize': 15,
                    'fontweight': 'bold',
                    'fontfamily': 'serif'})
    ax[1].set_title("confusion_matrix")
    ax[1].set_yticklabels(class_name, rotation=0,fontsize =15)
    ax[1].set_xticklabels(class_name, rotation=0,fontsize =15)
    ax[0].hist(df,label= "Distribution",bins=[i for i in range(9)])
    ax[0].set_title("Distribution")
    plt.tight_layout()
    if save: plt.savefig(f'{path}.jpeg',dpi=200)
    plt.show()
    return

def create_folder(path="./submission/"):
    tm = localtime()
    folder_time = f"{tm.tm_mon}{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}"
    submission_path = path
    folder_path = submission_path+folder_time
    if os.path.isdir(folder_path):pass
    else:
        os.makedirs(folder_path)
        print("make folder path: {}".format(folder_path))
    return folder_path
