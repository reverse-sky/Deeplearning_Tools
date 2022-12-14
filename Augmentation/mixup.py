import torch
import torch.nn as nn
import numpy as np

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1

    if use_cuda: index = torch.randperm(x.size()[0]).cuda()
    else: index = torch.randperm(x.size()[0])

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
