import torch
import torch.nn as nn
import torch.nn.functional as F

def knowledge_distillation_loss(teacher_logits,student_logits,labels,criterion = F.cross_entropy):
    alpha = 0.1
    T = 10

    student_loss = criterion(input=student_logits,target = labels)  # Calculate the loss about the Student model
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (T * T)
    total_loss =  alpha*student_loss + (1-alpha)*distillation_loss
    return total_loss
