import torch
import numpy as np
import random

def mixup(data, targets, alpha=1.):
    indices = torch.randperm(data.size(0))  # Shuffle index
    shuffled_data    = data[indices].clone()        # 
    shuffled_targets = targets[indices].clone()     # 
    lam = np.random.beta(alpha, alpha)      #
    mixed_data = lam * data + (1 - lam) * shuffled_data # mix data
    mixed_targets = lam * targets + (1 - lam) * shuffled_targets
    return mixed_data, mixed_targets
