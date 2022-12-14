def knowledge_distillation_loss(logits,labels,teacher_logits):
    alpha = 0.1
    T = 10

    student_loss = F.cross_entropy(input=logits,target = labels)
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (T * T)
    total_loss =  alpha*student_loss + (1-alpha)*distillation_loss
    return total_loss
